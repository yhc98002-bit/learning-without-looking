"""Construction-artifact attackers for a packaged release. Each attacker tries to tell the two
sides of a pair apart from something other than the answer - image statistics, PNG metadata,
file size or self-supervised image features - and is scored by grouped cross-validation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.stats import rankdata


METADATA_FEATURE_NAMES = (
    "file_size",
    "mtime_ns",
    "width",
    "height",
    "bands",
    "png_chunk_count",
    "idat_chunk_count",
    "idat_bytes",
    "ancillary_count",
    "compression_ratio",
    "path_length",
    "filename_hex_mean",
    "filename_hex_std",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _image_stats(path: str | Path) -> np.ndarray:
    """Colour, frequency-band and edge summaries of one image."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    means = array.mean(axis=(0, 1))
    stds = array.std(axis=(0, 1))
    quantiles = np.quantile(array, [0.05, 0.5, 0.95], axis=(0, 1)).reshape(-1)
    gray = array.mean(axis=2)
    magnitude = np.log1p(np.abs(np.fft.rfft2(gray)))
    bands = []
    height, width = magnitude.shape
    for low, high in ((0.0, 0.15), (0.15, 0.35), (0.35, 0.65), (0.65, 1.0)):
        y0, y1 = int(height * low), max(int(height * high), int(height * low) + 1)
        x0, x1 = int(width * low), max(int(width * high), int(width * low) + 1)
        bands.append(float(magnitude[y0:y1, x0:x1].mean()))
    vertical_edges = float(np.abs(np.diff(gray, axis=0)).mean())
    horizontal_edges = float(np.abs(np.diff(gray, axis=1)).mean())
    return np.asarray([*means, *stds, *quantiles, *bands, vertical_edges, horizontal_edges], dtype=np.float32)


def _png_chunk_lengths(path: Path) -> list[tuple[str, int]]:
    chunks: list[tuple[str, int]] = []
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            return chunks
        while True:
            length_bytes = handle.read(4)
            if len(length_bytes) != 4:
                break
            length = struct.unpack(">I", length_bytes)[0]
            chunk = handle.read(4).decode("ascii", errors="replace")
            chunks.append((chunk, length))
            handle.seek(length + 4, 1)
            if chunk == "IEND":
                break
    return chunks


def _file_size_features(path: str | Path) -> np.ndarray:
    """PNG byte size as a single column, so a size signature is scored on its own."""
    return np.asarray([float(Path(path).stat().st_size)], dtype=np.float64)


def _metadata_features(path: str | Path) -> np.ndarray:
    path = Path(path)
    chunks = _png_chunk_lengths(path)
    counts = Counter(name for name, _ in chunks)
    idat_bytes = sum(length for name, length in chunks if name == "IDAT")
    ancillary_count = sum(count for name, count in counts.items() if name and name[0].islower())
    with Image.open(path) as image:
        width, height = image.size
        bands = len(image.getbands())
    stat = path.stat()
    filename_values = np.asarray([int(char, 16) for char in path.stem if char in "0123456789abcdef"], dtype=np.float32)
    filename_mean = float(filename_values.mean()) if len(filename_values) else 0.0
    filename_std = float(filename_values.std()) if len(filename_values) else 0.0
    raw_bytes = max(1, width * height * bands)
    return np.asarray(
        [
            float(stat.st_size),
            float(stat.st_mtime_ns),
            float(width),
            float(height),
            float(bands),
            float(len(chunks)),
            float(counts.get("IDAT", 0)),
            float(idat_bytes),
            float(ancillary_count),
            float(stat.st_size / raw_bytes),
            float(len(path.as_posix())),
            filename_mean,
            filename_std,
        ],
        dtype=np.float64,
    )


def _try_dinov2_features(paths: list[str], model_name: str, batch_size: int) -> tuple[np.ndarray | None, str]:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"unavailable: import failed: {exc}"
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"unavailable: model load failed: {exc}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = [Image.open(path).convert("RGB") for path in paths[start : start + batch_size]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            features.append(outputs.last_hidden_state[:, 0].detach().float().cpu().numpy())
            for image in images:
                image.close()
    return np.concatenate(features, axis=0), "pass"


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, computed from ranks."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def univariate_feature_diagnosis(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Score each feature column on its own, so a single leaking column is visible."""
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("features must be a 2D matrix aligned with labels")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature_names must cover every feature column")
    output = {}
    for index, name in enumerate(feature_names):
        value = auc(labels, features[:, index])
        output[name] = {
            "auc": value,
            "gate_statistic": max(value, 1.0 - value),
            "mean_side_a": float(features[labels == 0, index].mean()),
            "mean_side_b": float(features[labels == 1, index].mean()),
        }
    return output


def grouped_folds(pair_ids: list[str], n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation folds that keep both members of a pair on the same side of the split."""
    unique_pairs = sorted(set(pair_ids))
    if len(unique_pairs) < n_splits:
        raise ValueError(f"need at least {n_splits} pairs for grouped CV")
    random.Random(seed).shuffle(unique_pairs)
    assignments = {pair_id: index % n_splits for index, pair_id in enumerate(unique_pairs)}
    folds = []
    pair_array = np.asarray(pair_ids)
    for fold in range(n_splits):
        test = np.asarray([assignments[pair_id] == fold for pair_id in pair_array], dtype=bool)
        folds.append((~test, test))
    return folds


def _fit_fold_scores(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Fit a ridge discriminant on the train fold and score the test fold, oriented by train AUC."""
    x_train = np.asarray(features[train], dtype=np.float64)
    x_test = np.asarray(features[test], dtype=np.float64)
    y_train = labels[train]
    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    target = y_train.astype(np.float64) * 2 - 1
    regularization = 1e-2
    lhs = x_train.T @ x_train + regularization * np.eye(x_train.shape[1], dtype=np.float64)
    weights = np.linalg.solve(lhs, x_train.T @ target)
    train_scores = x_train @ weights
    train_auc = auc(y_train, train_scores)
    direction = 1 if math.isnan(train_auc) or train_auc >= 0.5 else -1
    return direction * (x_test @ weights), direction, train_auc


def _pair_bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    pair_ids: list[str],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Bootstrap over pairs; returns the folded interval first, then the directed one."""
    indices_by_pair: dict[str, np.ndarray] = {}
    pair_array = np.asarray(pair_ids)
    for pair_id in sorted(set(pair_ids)):
        indices_by_pair[pair_id] = np.flatnonzero(pair_array == pair_id)
    pairs = sorted(indices_by_pair)
    rng = random.Random(seed)
    values = []
    raw_values = []
    for _ in range(n_bootstrap):
        sampled = [rng.choice(pairs) for _ in pairs]
        indices = np.concatenate([indices_by_pair[pair_id] for pair_id in sampled])
        value = auc(labels[indices], scores[indices])
        if not math.isnan(value):
            # Both intervals are read off the same draws, so adding the directed one left the
            # folded quantiles unchanged.
            values.append(max(value, 1.0 - value))
            raw_values.append(value)
    if not values:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.quantile(raw_values, 0.025)),
        float(np.quantile(raw_values, 0.975)),
    )


def evaluate_features(
    features: np.ndarray,
    labels: np.ndarray,
    pair_ids: list[str],
    *,
    n_splits: int = 5,
    seed: int = 0,
    n_bootstrap: int = 1000,
    include_scores: bool = False,
) -> dict[str, Any]:
    """Out-of-fold AUC for one attacker, with a bootstrap interval over pairs."""
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    directions = []
    train_aucs = []
    for train, test in grouped_folds(pair_ids, n_splits=n_splits, seed=seed):
        fold_scores, direction, train_auc = _fit_fold_scores(features, labels, train, test)
        scores[test] = fold_scores
        directions.append(direction)
        train_aucs.append(train_auc)
    if np.isnan(scores).any():
        raise AssertionError("OOF scores are incomplete")
    directed_auc = auc(labels, scores)
    gate_statistic = max(directed_auc, 1.0 - directed_auc)
    ci_low, ci_high, raw_ci_low, raw_ci_high = _pair_bootstrap_ci(
        labels,
        scores,
        pair_ids,
        n_bootstrap=n_bootstrap,
        seed=seed + 997,
    )
    result = {
        "directed_oof_auc": directed_auc,
        "gate_statistic": gate_statistic,
        "pair_bootstrap_ci_95": [ci_low, ci_high],
        "directed_oof_auc_unfolded_ci_95": [raw_ci_low, raw_ci_high],
        "fold_train_auc": train_aucs,
        "fold_direction": directions,
        "n_members": len(labels),
        "n_pairs": len(set(pair_ids)),
        "n_splits": n_splits,
    }
    if include_scores:
        result["oof_scores"] = scores.tolist()
    return result


def build_packaged_member_table(
    release_dir: str | Path,
    key_file: str | Path,
) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    """Paths, side labels, pair ids and template ids for every member of a packaged release."""
    release_dir = Path(release_dir)
    rows = _read_jsonl(release_dir / "manifest.jsonl")
    keys = {str(row["pair_id"]): row for row in _read_jsonl(key_file)}
    paths: list[str] = []
    labels: list[int] = []
    pair_ids: list[str] = []
    templates: list[str] = []
    for row in rows:
        pair_id = str(row["pair_id"])
        key = keys[pair_id]
        member_key = {str(member["member_id"]): member for member in key["members"]}
        for member in row["members"]:
            private = member_key[str(member["member_id"])]
            paths.append(str(release_dir / str(member["image_path"])))
            labels.append(0 if private["source_side"] == "a" else 1)
            pair_ids.append(pair_id)
            templates.append(str(key["template_id"]))
    return paths, np.asarray(labels, dtype=np.int64), pair_ids, templates


def _evaluate_all_scopes(
    features: np.ndarray,
    labels: np.ndarray,
    pair_ids: list[str],
    templates: list[str],
    *,
    n_splits: int,
    seed: int,
    n_bootstrap: int,
    include_scores: bool = False,
) -> dict[str, Any]:
    output = {
        "pooled": evaluate_features(
            features,
            labels,
            pair_ids,
            n_splits=n_splits,
            seed=seed,
            n_bootstrap=n_bootstrap,
            include_scores=include_scores,
        ),
        "per_template": {},
    }
    template_array = np.asarray(templates)
    for template in sorted(set(templates)):
        selected = template_array == template
        output["per_template"][template] = evaluate_features(
            features[selected],
            labels[selected],
            list(np.asarray(pair_ids)[selected]),
            n_splits=n_splits,
            seed=seed,
            n_bootstrap=n_bootstrap,
            include_scores=include_scores,
        )
    return output


def compute_gate(attacks: dict[str, Any]) -> dict[str, Any]:
    """Reduce the attacker results to the pass/fail checks reported with a release."""
    point_failures = []
    expansion_required = []
    missing = []
    for attacker, scopes in attacks.items():
        if scopes is None:
            missing.append(attacker)
            continue
        for scope_name, result in [("pooled", scopes["pooled"]), *scopes["per_template"].items()]:
            label = f"{attacker}:{scope_name}"
            if result["gate_statistic"] > 0.55:
                point_failures.append(label)
            if result["pair_bootstrap_ci_95"][1] > 0.62:
                expansion_required.append(label)
    checks = {
        "all_attackers_available": not missing,
        "all_point_estimates_at_most_0_55": not point_failures,
        "no_ci_upper_above_0_62": not expansion_required,
    }
    return {
        "status": all(checks.values()),
        "checks": checks,
        "missing_attackers": missing,
        "point_failures": point_failures,
        "expansion_required": expansion_required,
    }


def diagnose_old_splits(input_jsonl: str | Path) -> dict[str, Any]:
    """Repeat the diagnosis over ten split seeds, on an unpackaged source manifest."""
    rows = _read_jsonl(input_jsonl)
    paths: list[str] = []
    labels = []
    pair_ids = []
    for row in rows:
        paths.extend([row["image_a_path"], row["image_b_path"]])
        labels.extend([0, 1])
        pair_ids.extend([str(row["pair_id"]), str(row["pair_id"])])
    labels_array = np.asarray(labels, dtype=np.int64)
    stat_features = np.stack([_image_stats(path) for path in paths])
    metadata_features = np.stack([_metadata_features(path) for path in paths])
    output: dict[str, Any] = {"n_pairs": len(rows), "splits": []}
    for seed in range(10):
        train, test = grouped_folds(pair_ids, n_splits=3, seed=seed)[0]
        split = {"seed": seed}
        for name, features in (("frequency_stat", stat_features), ("metadata", metadata_features)):
            scores, direction, train_auc = _fit_fold_scores(features, labels_array, train, test)
            test_auc = auc(labels_array[test], scores)
            split[name] = {
                "train_auc": train_auc,
                "train_selected_direction": direction,
                "directed_test_auc": test_auc,
                "gate_statistic": max(test_auc, 1.0 - test_auc),
            }
        output["splits"].append(split)
    output["single_shared_split_is_not_sufficient"] = True
    return output


def run(
    release_dir: str | Path,
    key_file: str | Path,
    output: str | Path,
    dinov2_model: str,
    batch_size: int,
    *,
    skip_dino: bool = False,
    old_input_jsonl: str | Path | None = None,
    n_splits: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 20260710,
    per_item_scores: str | Path | None = None,
) -> dict[str, Any]:
    """Run every attacker over a packaged release and write the metrics JSON."""
    paths, labels, pair_ids, templates = build_packaged_member_table(release_dir, key_file)
    include_scores = per_item_scores is not None
    attacks: dict[str, Any] = {}
    stat_features = np.stack([_image_stats(path) for path in paths])
    metadata_features = np.stack([_metadata_features(path) for path in paths])
    # File size duplicates metadata column 0 on purpose: on its own it cannot be diluted by the
    # other metadata columns, so a pure size signature is caught at the same thresholds.
    size_features = np.stack([_file_size_features(path) for path in paths])
    attacks["file_size"] = _evaluate_all_scopes(
        size_features, labels, pair_ids, templates, n_splits=n_splits, seed=seed, n_bootstrap=n_bootstrap,
        include_scores=include_scores,
    )
    attacks["frequency_stat"] = _evaluate_all_scopes(
        stat_features, labels, pair_ids, templates, n_splits=n_splits, seed=seed, n_bootstrap=n_bootstrap,
        include_scores=include_scores,
    )
    attacks["metadata"] = _evaluate_all_scopes(
        metadata_features, labels, pair_ids, templates, n_splits=n_splits, seed=seed, n_bootstrap=n_bootstrap,
        include_scores=include_scores,
    )

    dino_status = "skipped"
    if skip_dino:
        attacks["dinov2"] = None
    else:
        dino_features, dino_status = _try_dinov2_features(paths, dinov2_model, batch_size)
        attacks["dinov2"] = (
            _evaluate_all_scopes(
                dino_features, labels, pair_ids, templates, n_splits=n_splits, seed=seed, n_bootstrap=n_bootstrap,
                include_scores=include_scores,
            )
            if dino_features is not None
            else None
        )

    if include_scores:
        # Keep the out-of-fold scores so intervals can be recomputed without rescoring the
        # images. The rows follow the member order each scope was scored in.
        template_array = np.asarray(templates)
        sidecar = Path(per_item_scores)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("w", encoding="utf-8") as handle:
            for attacker, scopes in attacks.items():
                if scopes is None:
                    continue
                scope_rows = [("pooled", np.ones(len(paths), dtype=bool))]
                scope_rows += [
                    (template, template_array == template)
                    for template in sorted(set(templates))
                ]
                for scope_name, mask in scope_rows:
                    result = scopes["pooled"] if scope_name == "pooled" else scopes["per_template"][scope_name]
                    oof = result.pop("oof_scores", None)
                    if oof is None:
                        continue
                    indices = np.flatnonzero(mask)
                    for position, index in enumerate(indices):
                        handle.write(
                            json.dumps(
                                {
                                    "attacker": attacker,
                                    "scope": scope_name,
                                    "pair_id": pair_ids[index],
                                    "image_path": paths[index],
                                    "label": int(labels[index]),
                                    "oof_score": oof[position],
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )

    metrics: dict[str, Any] = {
        "release_dir": str(release_dir),
        "key_file": str(key_file),
        "n_pairs": len(set(pair_ids)),
        "n_members": len(paths),
        "split": f"{n_splits}-fold grouped CV by pair",
        "direction": "selected from each train fold only",
        "gate_statistic": "max(AUC, 1-AUC)",
        "bootstrap": f"{n_bootstrap} resamples over pairs",
        "dinov2_status": dino_status,
        "attacks": attacks,
    }
    metrics["gate"] = compute_gate(attacks)
    if old_input_jsonl:
        metrics["historical_split_diagnosis"] = diagnose_old_splits(old_input_jsonl)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dinov2-model", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--old-input-jsonl")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--per-item-scores", default=None,
                        help="optional JSONL path persisting per-member OOF scores")
    args = parser.parse_args()
    result = run(
        args.release_dir,
        args.key_file,
        args.output,
        args.dinov2_model,
        args.batch_size,
        skip_dino=args.skip_dino,
        old_input_jsonl=args.old_input_jsonl,
        n_splits=args.n_splits,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        per_item_scores=args.per_item_scores,
    )
    print(json.dumps({"gate": result["gate"], "dinov2_status": result["dinov2_status"]}, sort_keys=True))


if __name__ == "__main__":
    main()

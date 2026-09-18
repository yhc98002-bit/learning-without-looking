#!/usr/bin/env python3
"""Freeze the decontaminated Geometry3K training subset.

Drops every training row flagged by the two contamination filters, then writes the retained
row indices, the training JSONL and a summary with digests and before/after distributions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from lwl.corpora.decontamination import normalize_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def derived_category(question: str) -> str:
    """Assign a question to the first matching topic family; Geometry3K ships no category field."""
    normalized = normalize_text(question)
    if normalized in {"find x", "find y", "find z", "find a", "find b", "find c"}:
        return "underspecified_visual"
    families = (
        ("circle_arc_tangent", r"\b(circle|arc|chord|tangent|radius|diameter|sector|circumference)\b"),
        ("triangle_trigonometry", r"\b(triangle|sine|sin|cosine|cos|tangent ratio|tan|hypotenuse|pythagorean)\b"),
        ("quadrilateral_polygon", r"\b(parallelogram|quadrilateral|rectangle|rhombus|trapezoid|polygon)\b"),
        ("coordinate_geometry", r"\b(coordinate|slope|midpoint|distance formula|ordered pair)\b"),
        ("area_volume", r"\b(area|volume|surface area|perimeter)\b"),
        ("angle_lines", r"\b(angle|parallel|perpendicular|supplementary|complementary|degree)\b"),
        ("length_ratio_similarity", r"\b(length|similar|congruent|ratio|proportion|scale factor)\b"),
    )
    for name, pattern in families:
        if re.search(pattern, normalized):
            return name
    return "other"


def answer_type(answer: Any) -> str:
    value = str(answer).strip().replace(",", "")
    value = value.replace("$", "").replace("^\\circ", "").replace("\N{DEGREE SIGN}", "").strip()
    if re.fullmatch(r"[+-]?\d+", value):
        return "integer"
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)", value):
        return "decimal"
    if re.fullmatch(r"[+-]?\d+\s*/\s*\d+", value) or "\\frac" in value:
        return "fraction"
    if re.search(r"[A-Za-z\\]|sqrt|pi", value):
        return "expression_or_text"
    return "other"


def _difficulty_band(probability: float) -> str:
    if probability == 0:
        return "p=0"
    if probability < 0.25:
        return "0<p<0.25"
    if probability < 0.75:
        return "0.25<=p<0.75"
    if probability < 1:
        return "0.75<=p<1"
    return "p=1"


def _distribution(
    rows: list[dict[str, Any]], difficulty: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    lengths = [len(normalize_text(row["problem"]).split()) for row in rows]
    difficulties = [difficulty[int(row["row_index"])] for row in rows]
    probabilities = [float(item["p_sample"]) for item in difficulties]
    categories = Counter(derived_category(str(row["problem"])) for row in rows)
    answer_types = Counter(answer_type(row["answer"]) for row in rows)
    bands = Counter(_difficulty_band(value) for value in probabilities)

    def counts(counter: Counter[str]) -> dict[str, dict[str, float | int]]:
        return {
            key: {"count": value, "fraction": value / len(rows)}
            for key, value in sorted(counter.items())
        }

    return {
        "n": len(rows),
        "derived_category": counts(categories),
        "answer_type": counts(answer_types),
        "question_length_tokens": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "min": min(lengths),
            "max": max(lengths),
        },
        "base_model_difficulty": {
            "mean_p_sample": statistics.fmean(probabilities),
            "median_p_sample": statistics.median(probabilities),
            "greedy_accuracy": statistics.fmean(
                float(bool(item["greedy_correct"])) for item in difficulties
            ),
            "bands": counts(bands),
        },
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def freeze_subset(
    *,
    source_manifest: Path,
    layer1_filter: Path,
    train_test_filter: Path,
    base_difficulty_rows: Path,
    ids_output: Path,
    dataset_output: Path,
    summary_output: Path,
) -> dict[str, Any]:
    """Write the retained ids, the training JSONL and the summary; existing outputs are never replaced."""
    for output in (ids_output, dataset_output, summary_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite frozen subset artifact: {output}")

    source = _read_jsonl(source_manifest)
    train = [row for row in source if row["split"] == "train"]
    if len({int(row["row_index"]) for row in train}) != len(train):
        raise ValueError("Geometry3K train row indices are not unique")
    source_record_ids = {
        f"geometry3k:train:{row['row_index']}:image{image_index}"
        for row in train
        for image_index, _ in enumerate(row["images"])
    }
    layer1 = json.loads(layer1_filter.read_text(encoding="utf-8"))
    train_test = json.loads(train_test_filter.read_text(encoding="utf-8"))
    if not train_test.get("complete"):
        raise ValueError("train-vs-test decontamination manifest is not complete")
    layer1_remove = set(map(str, layer1["remove_train_record_ids"]))
    train_test_remove = set(map(str, train_test["remove_train_record_ids"]))
    unknown = (layer1_remove | train_test_remove) - source_record_ids
    if unknown:
        raise ValueError(f"decontamination filters contain unknown source IDs: {sorted(unknown)[:5]}")

    remove = layer1_remove | train_test_remove
    retained = [
        row
        for row in train
        if not any(
            f"geometry3k:train:{row['row_index']}:image{index}" in remove
            for index, _ in enumerate(row["images"])
        )
    ]
    retained_ids = sorted(int(row["row_index"]) for row in retained)
    if not retained_ids:
        raise ValueError("decontamination removed every Geometry3K training row")

    difficulty_rows = [
        row for row in _read_jsonl(base_difficulty_rows) if row.get("split") == "train"
    ]
    difficulty = {int(row["row_index"]): row for row in difficulty_rows}
    expected_indices = {int(row["row_index"]) for row in train}
    if set(difficulty) != expected_indices:
        raise ValueError(
            "base-difficulty rows do not exactly cover Geometry3K train: "
            f"missing={len(expected_indices - set(difficulty))}, extra={len(set(difficulty) - expected_indices)}"
        )

    _atomic_write(ids_output, json.dumps(retained_ids, indent=2) + "\n")
    dataset_rows = [
        {
            "images": [str(image["path"]) for image in row["images"]],
            "problem": row["problem"],
            "answer": row["answer"],
            "row_index": int(row["row_index"]),
        }
        for row in retained
    ]
    _atomic_write(
        dataset_output,
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in dataset_rows),
    )
    summary: dict[str, Any] = {
        "schema_version": "lwl.geo3k-pilot-filtered.v1",
        "status": "pass",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "layer1_filter": str(layer1_filter),
        "layer1_filter_sha256": _sha256(layer1_filter),
        "train_test_filter": str(train_test_filter),
        "train_test_filter_sha256": _sha256(train_test_filter),
        "base_difficulty_rows": str(base_difficulty_rows),
        "base_difficulty_rows_sha256": _sha256(base_difficulty_rows),
        "n_original_train": len(train),
        "n_layer1_remove": len(layer1_remove),
        "n_train_test_remove": len(train_test_remove),
        "n_remove_intersection": len(layer1_remove & train_test_remove),
        "n_remove_union": len(remove),
        "n_retained": len(retained),
        "ids_output": str(ids_output),
        "ids_sha256": _sha256(ids_output),
        "dataset_output": str(dataset_output),
        "dataset_sha256": _sha256(dataset_output),
        "distribution": {
            "original": _distribution(train, difficulty),
            "filtered": _distribution(retained, difficulty),
        },
        "category_note": (
            "hiyouga/geometry3k exposes no source category field; derived_category is a "
            "deterministic question-text taxonomy and is reported as a proxy"
        ),
    }
    _atomic_write(summary_output, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--layer1-filter", type=Path, required=True)
    parser.add_argument("--train-test-filter", type=Path, required=True)
    parser.add_argument("--base-difficulty-rows", type=Path, required=True)
    parser.add_argument("--ids-output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    summary = freeze_subset(**vars(args))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "n_original_train": summary["n_original_train"],
                "n_remove_union": summary["n_remove_union"],
                "n_retained": summary["n_retained"],
                "ids_sha256": summary["ids_sha256"],
                "dataset_sha256": summary["dataset_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

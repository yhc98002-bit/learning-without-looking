"""Stratified sample of ViRL39K used for the necessity audit.

Strata are (source, category, answer type, pass-rate bin, image-count bucket); the sample is
written as manifest rows with a content-addressed image index.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lwl.corpora.virl39k import answer_type
from lwl.rewards.answer import extract_answer_span


SCHEMA_VERSION = "lwl.virl39k-blind-sample.v1"


def pass_rate_bin(value: float) -> str:
    if value < 0:
        return "missing"
    if value < 0.2:
        return "00_20"
    if value < 0.4:
        return "20_40"
    if value < 0.6:
        return "40_60"
    if value < 0.8:
        return "60_80"
    return "80_100"


def image_count_bucket(count: int) -> str:
    if count == 1:
        return "1"
    if 2 <= count <= 4:
        return "2_4"
    return "5_plus"


def stratum_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row["source"]),
        str(row["category"]),
        answer_type(str(row["answer"])),
        pass_rate_bin(float(row["pass_rate_7b_base"])),
        image_count_bucket(len(row["image_paths"])),
    )


def _allocation(groups: dict[tuple[str, ...], list[dict[str, Any]]], sample_size: int) -> dict[tuple[str, ...], int]:
    """Proportional allocation with largest-remainder rounding, capped by stratum size."""
    population = sum(len(rows) for rows in groups.values())
    if not 1 <= sample_size <= population:
        raise ValueError("sample_size must be in [1, population]")
    ideals = {key: sample_size * len(rows) / population for key, rows in groups.items()}
    allocation = {key: min(len(groups[key]), math.floor(ideal)) for key, ideal in ideals.items()}
    remaining = sample_size - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda key: (ideals[key] - allocation[key], len(groups[key]), key),
        reverse=True,
    )
    for key in order:
        if remaining == 0:
            break
        if allocation[key] < len(groups[key]):
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("proportional allocation did not reach requested sample size")
    return allocation


def stratified_sample(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    """Draw the sample; each stratum is shuffled with a seed derived from the stratum key."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum_key(row)].append(row)
    allocation = _allocation(groups, sample_size)
    selected = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda row: str(row["qid"]))
        key_seed = hashlib.sha256(
            f"{seed}:{json.dumps(key, separators=(',', ':'))}".encode("utf-8")
        ).digest()
        random.Random(key_seed).shuffle(candidates)
        selected.extend(candidates[: allocation[key]])
    return sorted(selected, key=lambda row: str(row["qid"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _problem_with_image_markers(question: str, image_count: int) -> tuple[str, bool]:
    """Prepend missing <image> markers; returns the question and whether it was repaired."""
    marker_count = question.count("<image>")
    if marker_count == image_count:
        return question, False
    if marker_count == 0 and image_count:
        prefix = "\n".join("<image>" for _ in range(image_count))
        return f"{prefix}\n{question}", True
    raise ValueError(f"question has {marker_count} image markers for {image_count} images")


def build_manifest_rows(
    selected: list[dict[str, Any]],
    image_index_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write the sha256-named image index and return the manifest rows with their statistics."""
    image_index_dir = Path(image_index_dir)
    image_index_dir.mkdir(parents=True, exist_ok=False)
    digest_cache: dict[Path, str] = {}
    manifest_rows = []
    repaired = 0
    for row_index, row in enumerate(selected):
        problem, marker_repaired = _problem_with_image_markers(
            str(row["question"]), len(row["image_paths"])
        )
        repaired += int(marker_repaired)
        images = []
        for raw_path in row["image_paths"]:
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = digest_cache.get(path)
            if digest is None:
                digest = _sha256(path)
                digest_cache[path] = digest
            link = image_index_dir / f"{digest}{path.suffix.lower()}"
            if not link.exists():
                link.symlink_to(path.resolve())
            images.append({"path": str(path), "sha256": digest})
        answer_raw = str(row["answer"])
        answer = extract_answer_span(answer_raw).span.strip()
        if not answer:
            raise ValueError(f"empty canonical answer for ViRL39K qid {row['qid']}")
        manifest_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "split": "audit",
                "row_index": row_index,
                "qid": str(row["qid"]),
                "problem": problem,
                "answer": answer,
                "images": images,
                "metadata": {
                    "source": str(row["source"]),
                    "category": str(row["category"]),
                    "answer_type": answer_type(answer_raw),
                    "answer_raw": answer_raw,
                    "pass_rate_32b_trained": float(row["pass_rate_32b_trained"]),
                    "pass_rate_7b_base": float(row["pass_rate_7b_base"]),
                    "pass_rate_bin": pass_rate_bin(float(row["pass_rate_7b_base"])),
                    "image_count": len(images),
                    "image_count_bucket": image_count_bucket(len(images)),
                    "original_image_marker_count": str(row["question"]).count("<image>"),
                    "image_markers_repaired": marker_repaired,
                    "relative_image_paths": list(row["relative_image_paths"]),
                },
            }
        )
    stats = {
        "schema_version": SCHEMA_VERSION,
        "sample_size": len(manifest_rows),
        "unique_images": len(set(digest_cache.values())),
        "image_references": sum(len(row["images"]) for row in manifest_rows),
        "marker_repaired_rows": repaired,
        "max_images_per_item": max((len(row["images"]) for row in manifest_rows), default=0),
        "source_counts": dict(sorted(Counter(row["metadata"]["source"] for row in manifest_rows).items())),
        "category_counts": dict(sorted(Counter(row["metadata"]["category"] for row in manifest_rows).items())),
        "answer_type_counts": dict(
            sorted(Counter(row["metadata"]["answer_type"] for row in manifest_rows).items())
        ),
        "pass_rate_bin_counts": dict(
            sorted(Counter(row["metadata"]["pass_rate_bin"] for row in manifest_rows).items())
        ),
        "image_count_counts": dict(
            sorted(Counter(str(len(row["images"])) for row in manifest_rows).items())
        ),
    }
    return manifest_rows, stats

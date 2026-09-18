#!/usr/bin/env python3
"""Freeze the decontaminated ViRL39K training subset.

Removes every item with a flagged image record, canonicalizes each answer, and writes the
retained qids, the training JSONL, a sha256-named image index and a summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from lwl.corpora.virl39k import answer_type, load_rows
from lwl.corpora.virl39k_sample import image_count_bucket, pass_rate_bin
from lwl.rewards.answer import extract_answer_span


SCHEMA_VERSION = "lwl.virl39k-filtered-training.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _problem_with_image_markers(question: str, image_count: int) -> tuple[str, bool]:
    """Prepend missing <image> markers; returns the question and whether it was repaired."""
    marker_count = question.count("<image>")
    if marker_count == image_count:
        return question, False
    if marker_count == 0 and image_count:
        return f"{' '.join('<image>' for _ in range(image_count))}\n{question}", True
    raise ValueError(
        f"question has {marker_count} image markers for {image_count} images"
    )


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": len(rows),
        "image_references": sum(len(row["image_paths"]) for row in rows),
        "sources": dict(sorted(Counter(str(row["source"]) for row in rows).items())),
        "categories": dict(sorted(Counter(str(row["category"]) for row in rows).items())),
        "answer_types": dict(
            sorted(Counter(answer_type(str(row["answer"])) for row in rows).items())
        ),
        "pass_rate_7b_base_bins": dict(
            sorted(
                Counter(pass_rate_bin(float(row["pass_rate_7b_base"])) for row in rows).items()
            )
        ),
        "image_count_buckets": dict(
            sorted(Counter(image_count_bucket(len(row["image_paths"])) for row in rows).items())
        ),
    }


def freeze_subset(
    *,
    source_parquet: Path,
    image_root: Path,
    filter_manifest: Path,
    ids_output: Path,
    dataset_output: Path,
    image_index_dir: Path,
    summary_output: Path,
    expected_items: int = 38_870,
    expected_records: int = 42_908,
) -> dict[str, Any]:
    """Write the retained qids, the training JSONL, the image index and the summary.

    The expected counts pin the upstream ViRL39K revision; a mismatch stops the build.
    """
    for output in (ids_output, dataset_output, image_index_dir, summary_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite frozen ViRL39K artifact: {output}")

    rows = load_rows(source_parquet, image_root)
    if len(rows) != expected_items:
        raise ValueError(f"expected {expected_items} ViRL39K items, found {len(rows)}")
    record_to_qid = {
        f"virl39k:train:{row['qid']}:image{image_index}": str(row["qid"])
        for row in rows
        for image_index, _ in enumerate(row["image_paths"])
    }
    if len(record_to_qid) != expected_records:
        raise ValueError(
            f"expected {expected_records} ViRL39K image records, found {len(record_to_qid)}"
        )

    filtering = json.loads(filter_manifest.read_text(encoding="utf-8"))
    if filtering.get("complete") is not True or filtering.get("pending_layers"):
        raise ValueError("ViRL39K decontamination manifest is incomplete")
    if int(filtering.get("n_train_records", -1)) != expected_records:
        raise ValueError("decontamination manifest train-record count does not match source")
    remove_records = set(map(str, filtering.get("remove_train_record_ids", [])))
    inspect_records = set(map(str, filtering.get("inspect_only_train_record_ids", [])))
    unknown = (remove_records | inspect_records) - set(record_to_qid)
    if unknown:
        raise ValueError(f"decontamination manifest contains unknown record IDs: {sorted(unknown)[:5]}")
    remove_qids = {record_to_qid[record_id] for record_id in remove_records}
    inspect_qids = {record_to_qid[record_id] for record_id in inspect_records} - remove_qids
    retained = [row for row in rows if str(row["qid"]) not in remove_qids]
    if not retained:
        raise ValueError("decontamination removed every ViRL39K item")

    frozen_rows: list[dict[str, Any]] = []
    repaired_markers = 0
    image_digests: dict[Path, str] = {}
    for row_index, row in enumerate(retained):
        problem, repaired = _problem_with_image_markers(
            str(row["question"]), len(row["image_paths"])
        )
        repaired_markers += int(repaired)
        answer_raw = str(row["answer"])
        answer = extract_answer_span(answer_raw).span.strip()
        if not answer:
            raise ValueError(f"empty canonical answer for ViRL39K qid {row['qid']}")
        row_digests = []
        for raw_path in row["image_paths"]:
            path = Path(raw_path)
            digest = image_digests.get(path)
            if digest is None:
                digest = _sha256(path)
                image_digests[path] = digest
            row_digests.append(digest)
        frozen_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "split": "train",
                "row_index": row_index,
                "qid": str(row["qid"]),
                "problem": problem,
                "answer": answer,
                "images": list(row["image_paths"]),
                "metadata": {
                    "source": str(row["source"]),
                    "category": str(row["category"]),
                    "answer_raw": answer_raw,
                    "answer_type": answer_type(answer_raw),
                    "pass_rate_32b_trained": float(row["pass_rate_32b_trained"]),
                    "pass_rate_7b_base": float(row["pass_rate_7b_base"]),
                    "relative_image_paths": list(row["relative_image_paths"]),
                    "image_sha256": row_digests,
                    "image_markers_repaired": repaired,
                },
            }
        )

    temporary_index = image_index_dir.with_name(f".{image_index_dir.name}.partial.{os.getpid()}")
    try:
        temporary_index.mkdir(parents=True, exist_ok=False)
        by_digest: dict[str, Path] = {}
        for path, digest in image_digests.items():
            by_digest.setdefault(digest, path)
        for digest, path in sorted(by_digest.items()):
            (temporary_index / f"{digest}{path.suffix.lower()}").symlink_to(path.resolve())
        os.replace(temporary_index, image_index_dir)
    except Exception:
        shutil.rmtree(temporary_index, ignore_errors=True)
        raise

    retained_ids = [str(row["qid"]) for row in retained]
    _atomic_write(ids_output, json.dumps(retained_ids, indent=2) + "\n")
    _atomic_write(
        dataset_output,
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in frozen_rows),
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "source_parquet": str(source_parquet),
        "source_parquet_sha256": _sha256(source_parquet),
        "filter_manifest": str(filter_manifest),
        "filter_manifest_sha256": _sha256(filter_manifest),
        "candidate_language": "conservative contamination candidates",
        "item_removal_rule": (
            "remove an entire ViRL39K item when any of its image records is an automatic-remove candidate"
        ),
        "inspect_policy": "inspection-only candidates remain in the frozen subset",
        "n_original_items": len(rows),
        "n_original_records": len(record_to_qid),
        "n_remove_records": len(remove_records),
        "n_remove_items": len(remove_qids),
        "n_inspect_only_records": len(inspect_records),
        "n_inspect_only_items_retained": len(inspect_qids),
        "n_retained_items": len(retained),
        "n_retained_image_references": sum(len(row["image_paths"]) for row in retained),
        "n_retained_unique_images": len(set(image_digests.values())),
        "marker_repaired_rows": repaired_markers,
        "ids_output": str(ids_output),
        "ids_sha256": _sha256(ids_output),
        "dataset_output": str(dataset_output),
        "dataset_sha256": _sha256(dataset_output),
        "image_index_dir": str(image_index_dir),
        "image_index_manifest_sha256": hashlib.sha256(
            "".join(
                f"{digest} {path.resolve()}\n"
                for digest, path in sorted(by_digest.items())
            ).encode("utf-8")
        ).hexdigest(),
        "distribution": {
            "original": _distribution(rows),
            "filtered": _distribution(retained),
        },
    }
    _atomic_write(summary_output, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--filter-manifest", type=Path, required=True)
    parser.add_argument("--ids-output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--image-index-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    summary = freeze_subset(**vars(args))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("status", "n_remove_items", "n_retained_items", "dataset_sha256")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

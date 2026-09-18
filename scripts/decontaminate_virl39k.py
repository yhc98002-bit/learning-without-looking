#!/usr/bin/env python3
"""Build the contamination records for ViRL39K against the evaluation benchmarks.

Writes one enriched record per (item, image) for the training corpus and for every benchmark,
plus a summary carrying the pinned ViRL39K revision and the digest of each input.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lwl.corpora.decontamination import (
    enrich_records,
    load_layer1_records,
    load_virl39k_records,
    sha256_file,
    write_jsonl,
)


def build_records(
    *,
    virl_parquet: Path,
    virl_image_root: Path,
    mmstar_tsv: Path,
    mmstar_image_root: Path,
    mathvista_tsv: Path,
    blink_tsv: Path,
    mmvp_tsv: Path,
    hallusion_tsv: Path,
    mathverse_tsv: Path,
    mmmu_tsv: Path,
    train_output: Path,
    eval_output: Path,
    summary_output: Path,
) -> dict[str, object]:
    """Emit the training and evaluation record files; the item and record counts pin the corpus."""
    for output in (train_output, eval_output, summary_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite decontamination artifact: {output}")

    train_rows = enrich_records(load_virl39k_records(virl_parquet, virl_image_root))
    eval_rows = enrich_records(
        load_layer1_records(
            mmstar_tsv,
            mmstar_image_root,
            mathvista_tsv,
            blink_tsv,
            mmvp_tsv=mmvp_tsv,
            hallusion_tsv=hallusion_tsv,
            mathverse_tsv=mathverse_tsv,
            mmmu_tsv=mmmu_tsv,
        )
    )
    train_item_ids = {row["item_id"] for row in train_rows}
    if len(train_item_ids) != 38_870:
        raise ValueError(f"expected 38,870 ViRL39K items, found {len(train_item_ids)}")
    if len(train_rows) != 42_908:
        raise ValueError(f"expected 42,908 ViRL39K image records, found {len(train_rows)}")
    expected_eval_datasets = {
        "blink",
        "hallusionbench",
        "mathverse",
        "mathvista",
        "mmmu",
        "mmstar",
        "mmvp",
    }
    actual_eval_datasets = {row["dataset"] for row in eval_rows}
    if actual_eval_datasets != expected_eval_datasets:
        raise ValueError(
            "evaluation dataset mismatch: "
            f"missing={sorted(expected_eval_datasets - actual_eval_datasets)}, "
            f"extra={sorted(actual_eval_datasets - expected_eval_datasets)}"
        )

    write_jsonl(train_rows, train_output)
    write_jsonl(eval_rows, eval_output)
    input_paths = (
        virl_parquet,
        mmstar_tsv,
        mathvista_tsv,
        blink_tsv,
        mmvp_tsv,
        hallusion_tsv,
        mathverse_tsv,
        mmmu_tsv,
    )
    summary: dict[str, object] = {
        "schema_version": "lwl.virl39k-layer1-decon-record-summary.v1",
        "status": "pass",
        "virl_revision": "TIGER-Lab/ViRL39K@812ec617dea4bc8a4e751663b88e4ebb7de4d00e",
        "train_output": str(train_output),
        "train_sha256": sha256_file(train_output),
        "eval_output": str(eval_output),
        "eval_sha256": sha256_file(eval_output),
        "n_train_items": len(train_item_ids),
        "n_train_records": len(train_rows),
        "n_train_unique_images": len({row["image_sha256"] for row in train_rows}),
        "n_eval_items": len({(row["dataset"], row["item_id"]) for row in eval_rows}),
        "n_eval_records": len(eval_rows),
        "n_eval_unique_applicable_images": len(
            {
                row["image_sha256"]
                for row in eval_rows
                if row.get("image_applicable", True)
            }
        ),
        "train_source_counts": dict(
            sorted(Counter(str(row["source"]) for row in train_rows).items())
        ),
        "train_category_counts": dict(
            sorted(Counter(str(row["category"]) for row in train_rows).items())
        ),
        "eval_dataset_record_counts": dict(
            sorted(Counter(str(row["dataset"]) for row in eval_rows).items())
        ),
        "source_hashes": {str(path): sha256_file(path) for path in input_paths},
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--virl-parquet", type=Path, required=True)
    parser.add_argument("--virl-image-root", type=Path, required=True)
    parser.add_argument("--mmstar-tsv", type=Path, required=True)
    parser.add_argument("--mmstar-image-root", type=Path, required=True)
    parser.add_argument("--mathvista-tsv", type=Path, required=True)
    parser.add_argument("--blink-tsv", type=Path, required=True)
    parser.add_argument("--mmvp-tsv", type=Path, required=True)
    parser.add_argument("--hallusion-tsv", type=Path, required=True)
    parser.add_argument("--mathverse-tsv", type=Path, required=True)
    parser.add_argument("--mmmu-tsv", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_records(**vars(args))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("status", "n_train_items", "n_train_records", "n_eval_records")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

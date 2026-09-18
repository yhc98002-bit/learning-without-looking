"""Export the pinned Geometry3K release to PNG images and a JSONL manifest.

The summary records the source revision and a sha256 for every file written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_rows(rows: Iterable[dict[str, Any]], split: str, output_dir: Path) -> list[dict[str, Any]]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        images = row.get("images") or []
        if not images:
            raise ValueError(f"Geometry3K {split} row {row_index} has no images")
        image_records = []
        for image_index, source in enumerate(images):
            if not isinstance(source, Image.Image):
                raise TypeError(f"Geometry3K {split} row {row_index} image is not PIL.Image")
            path = split_dir / f"{row_index:05d}_{image_index}.png"
            source.convert("RGB").save(path, format="PNG", optimize=False, compress_level=9)
            image_records.append({"path": str(path), "sha256": sha256_file(path)})
        records.append(
            {
                "split": split,
                "row_index": row_index,
                "problem": str(row["problem"]),
                "answer": str(row["answer"]),
                "images": image_records,
            }
        )
    return records


def export_dataset(dataset_root: Path, output_dir: Path, manifest_path: Path, summary_path: Path) -> dict[str, Any]:
    if output_dir.exists() or manifest_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite Geometry3K caption export")
    source_paths = {
        split: dataset_root / f"geometry3k-{split}.arrow"
        for split in ("train", "test")
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for split, path in source_paths.items():
        dataset = Dataset.from_file(str(path))
        records.extend(export_rows(dataset, split, output_dir))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    image_count = sum(len(record["images"]) for record in records)
    unique_hashes = {image["sha256"] for record in records for image in record["images"]}
    summary = {
        "schema_version": "lwl.geometry3k-caption-export.v1",
        "source_revision": "hiyouga/geometry3k@fd21e533e1e50d0662a2bf7b223e60511bd5f8b7",
        "source_arrow_sha256": {split: sha256_file(path) for split, path in source_paths.items()},
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_counts": {
            split: sum(record["split"] == split for record in records)
            for split in source_paths
        },
        "n_rows": len(records),
        "n_images": image_count,
        "n_unique_image_hashes": len(unique_hashes),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.dataset_root, args.output_dir, args.manifest, args.summary), sort_keys=True))


if __name__ == "__main__":
    main()

"""Cheap construction-artifact score for a source manifest: how far the two sides of a pair
differ in file size and path length, written back onto every row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def metadata_features(row: dict[str, Any]) -> dict[str, float]:
    image_a = Path(row["image_a_path"])
    image_b = Path(row["image_b_path"])
    return {
        "size_abs_delta": float(abs(image_a.stat().st_size - image_b.stat().st_size)),
        "same_suffix": float(image_a.suffix.lower() == image_b.suffix.lower()),
        "path_len_abs_delta": float(abs(len(str(image_a)) - len(str(image_b)))),
    }


def simple_artifact_score(row: dict[str, Any]) -> float:
    """Worst of the size and path-length differences, each clipped to 1.0."""
    feats = metadata_features(row)
    size_penalty = min(1.0, feats["size_abs_delta"] / 100_000.0)
    path_penalty = min(1.0, feats["path_len_abs_delta"] / 20.0)
    return max(size_penalty, path_penalty)


def score_manifest(input_jsonl: str | Path, output_jsonl: str | Path) -> None:
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_jsonl).open("r", encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            row["artifact_gate_score"] = simple_artifact_score(row)
            row["artifact_gate_features"] = metadata_features(row)
            dst.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()
    score_manifest(args.input_jsonl, args.output_jsonl)
    print(args.output_jsonl)


if __name__ == "__main__":
    main()

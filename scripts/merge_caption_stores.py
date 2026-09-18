#!/usr/bin/env python3
"""Merge caption-store shards into one store covering exactly the images of a release manifest.

The merged store and its summary are written through temporary files, so a failed merge leaves
no partial output behind.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lwl.captions.store import merge_caption_rows


def _release_hashes(path: Path) -> set[str]:
    """Image hashes of every member in a released pair manifest."""
    hashes = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            hashes.update(str(member["image_sha256"]) for member in row["members"])
    return hashes


def _publish_artifacts(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output: Path,
    summary_path: Path,
) -> None:
    for final_path in (output, summary_path):
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite caption merge artifact: {final_path}")
        final_path.parent.mkdir(parents=True, exist_ok=True)

    output_partial = Path(f"{output}.partial")
    summary_partial = Path(f"{summary_path}.partial")
    for partial_path in (output_partial, summary_partial):
        if partial_path.exists():
            raise FileExistsError(f"stale partial artifact requires inspection: {partial_path}")

    published_output = False
    try:
        with output_partial.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        summary_partial.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(output_partial, output)
        published_output = True
        os.replace(summary_partial, summary_path)
    except BaseException:
        output_partial.unlink(missing_ok=True)
        summary_partial.unlink(missing_ok=True)
        if published_output:
            output.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = merge_caption_rows(args.shards, _release_hashes(args.release_manifest))
    summary.update(
        {
            "release_manifest": str(args.release_manifest),
            "input_shards": [str(path) for path in args.shards],
        }
    )
    _publish_artifacts(rows, summary, args.output, args.summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

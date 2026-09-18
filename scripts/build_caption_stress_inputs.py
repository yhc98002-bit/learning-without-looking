#!/usr/bin/env python3
"""Build the caption-stress inputs for the constructed scenes.

For each family this writes a release directory (manifest.jsonl plus hard-linked images) and a
private key file, shaped for lwl.captions.qa_pairs.build_caption_qa_rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from lwl.paths import data_path, resolve_data_reference, results_root

FAMILIES = {
    "hier_coord_v1": ("n8", "n12", "n20"),
}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_family(data_dir: Path, family: str, cells: tuple[str, ...]) -> dict:
    """Package the discovery-level causal pairs of one family; image hashes are re-checked on disk."""
    release_dir = data_dir / f"caption_stress_{family}"
    key_path = data_dir / f"caption_stress_key_{family}.jsonl"
    for path in (release_dir, key_path):
        if path.exists():
            raise FileExistsError(path)
    release_rows, key_rows = [], []
    seen_hashes: set[str] = set()
    per_cell: dict[str, int] = {}
    for cell in cells:
        manifest = data_dir / f"manifest_{family}_{cell}_l3.jsonl"
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["role"] == "invariance":
                continue
            pair_id = str(row["pair_id"])
            members, key_members = [], []
            # source_side is the file-suffix side: answer_{side} describes image_{side}_path, so
            # each answer travels with its own image. The attack key's semantic
            # side_assignment_swapped is deliberately not applied - applying it here would invert
            # the answers of every swapped pair.
            for side in ("a", "b"):
                src = resolve_data_reference(row[f"image_{side}_path"], manifest.parent)
                if not src.is_file():
                    raise FileNotFoundError(src)
                digest = sha256_file(src)
                if digest != row[f"image_{side}_sha256"]:
                    raise ValueError(
                        f"on-disk sha mismatch for {pair_id} side {side}")
                if digest in seen_hashes:
                    raise ValueError(f"duplicate image hash in {family}: {digest}")
                seen_hashes.add(digest)
                rel = Path("images") / f"{pair_id}_{side}.png"
                link_or_copy(src, release_dir / rel)
                members.append({"member_id": f"{pair_id}_{side}",
                                "image_path": str(rel),
                                "image_sha256": digest})
                key_members.append({"member_id": f"{pair_id}_{side}",
                                    "source_side": side,
                                    "answer": str(row[f"answer_{side}"])})
            release_rows.append({"pair_id": pair_id,
                                 "question": str(row["question"]),
                                 "members": members})
            key_rows.append({"pair_id": pair_id,
                             "source_pair_id": pair_id,
                             "category": family,
                             "template_id": row["template_id"],
                             "members": key_members})
            per_cell[cell] = per_cell.get(cell, 0) + 1
    (release_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in release_rows), encoding="utf-8")
    key_path.write_text(
        "".join(json.dumps(r) + "\n" for r in key_rows), encoding="utf-8")
    return {"pairs": len(release_rows),
            "per_cell": per_cell,
            "release_dir": str(release_dir),
            "release_manifest_sha256": sha256_file(release_dir / "manifest.jsonl"),
            "key": str(key_path),
            "key_sha256": sha256_file(key_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_path("scenes", "development"))
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILIES),
                        default=sorted(FAMILIES))
    parser.add_argument("--report", type=Path,
                        default=results_root() / "caption_stress_inputs.json")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    summary = {"schema_version": "lwl.scene-caption-stress-inputs.v1",
               "side_binding": "file-suffix (answers travel with their own image; "
                               "semantic swap NOT applied)",
               "population": "discovery-level causal pairs (role != invariance)",
               "families": {}}
    for family, cells in ((f, FAMILIES[f]) for f in args.families):
        summary["families"][family] = build_family(args.data_dir, family, cells)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary["families"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

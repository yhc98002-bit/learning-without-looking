#!/usr/bin/env python3
"""Package one attacker set per scene family from its discovery-level pairs: a manifest of
pair ids with release-relative image paths, and a separate key giving each member's semantic
side. Images are hard-linked where the filesystem allows it, so a set adds no duplicate bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.paths import data_path, repo_relative, resolve_data_reference

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_path("scenes", "development"))
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILIES),
                        default=sorted(FAMILIES))
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    summary = {}
    for family, cells in ((f, FAMILIES[f]) for f in args.families):
        release_dir = args.data_dir / f"attacker_release_{family}"
        if release_dir.exists():
            raise FileExistsError(release_dir)
        release_rows, key_rows = [], []
        for cell in cells:
            manifest = args.data_dir / f"manifest_{family}_{cell}_l3.jsonl"
            for line in manifest.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["role"] == "invariance":
                    continue  # the attacker set uses the counterfactual pairs only
                swapped = bool(row["provenance"]["semantic_side_assignment_swapped"])
                members, key_members = [], []
                for side in ("a", "b"):
                    member_id = f"{row['pair_id']}_{side}"
                    src = resolve_data_reference(row[f"image_{side}_path"], manifest.parent)
                    rel = Path("images") / src.name
                    link_or_copy(src, release_dir / rel)
                    members.append({"member_id": member_id,
                                    "image_path": str(rel)})
                    semantic = {"a": "b", "b": "a"}[side] if swapped else side
                    key_members.append({"member_id": member_id,
                                        "source_side": semantic})
                release_rows.append({"pair_id": row["pair_id"], "members": members})
                key_rows.append({"pair_id": row["pair_id"],
                                 "template_id": row["template_id"],
                                 "members": key_members})
        (release_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in release_rows), encoding="utf-8")
        key_path = args.data_dir / f"attacker_key_{family}.jsonl"
        if key_path.exists():
            raise FileExistsError(key_path)
        key_path.write_text(
            "".join(json.dumps(r) + "\n" for r in key_rows), encoding="utf-8")
        summary[family] = {"pairs": len(release_rows),
                           "release_dir": repo_relative(release_dir),
                           "key": repo_relative(key_path)}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

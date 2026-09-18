#!/usr/bin/env python3
"""Place the upstream ViRL39K images where the released training rows and dose mixtures expect them.

The images are not redistributed. This reads images.zip from the pinned ViRL39K revision (or a
local copy given with --archive), extracts every image the filtered rows reference to
data/virl39k/images/<file>, and checks each against the SHA-256 recorded in those rows.

    python scripts/fetch_virl39k.py                          # downloads images.zip, about 1.8 GB
    python scripts/fetch_virl39k.py --archive images.zip     # an archive already on disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.paths import data_path

REPO_ID = "TIGER-Lab/ViRL39K"
REVISION = "812ec617dea4bc8a4e751663b88e4ebb7de4d00e"
ARCHIVE = "images.zip"


def wanted_images(rows_path: Path) -> dict[str, str]:
    """Archive member -> SHA-256, for every image the rows reference."""
    wanted: dict[str, str] = {}
    with rows_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row["metadata"]
            members, digests = metadata["relative_image_paths"], metadata["image_sha256"]
            if len(members) != len(digests):
                raise ValueError(f"row {row['qid']} records {len(members)} images and {len(digests)} digests")
            for member, digest in zip(members, digests):
                if wanted.setdefault(member, digest) != digest:
                    raise ValueError(f"{member} is recorded with two different digests")
    return wanted


def place_images(archive: zipfile.ZipFile, wanted: dict[str, str], dest: Path) -> dict[str, list[str] | int]:
    """Extract and verify each wanted member under `dest`; files already in place are kept."""
    present, placed, missing, mismatched = 0, 0, [], []
    names = set(archive.namelist())
    for member in sorted(wanted):
        target = dest / member
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == wanted[member]:
            present += 1
            continue
        if member not in names:
            missing.append(member)
            continue
        blob = archive.read(member)
        if hashlib.sha256(blob).hexdigest() != wanted[member]:
            mismatched.append(member)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.partial.{os.getpid()}")
        partial.write_bytes(blob)
        os.replace(partial, target)
        placed += 1
    return {"wanted": len(wanted), "already_present": present, "placed": placed,
            "missing_from_archive": missing, "sha256_mismatch": mismatched}


def download_archive(cache_dir: Path | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install -r requirements-cpu.txt")
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=ARCHIVE,
                                revision=REVISION, cache_dir=cache_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=Path, default=data_path("training", "virl39k", "filtered_rows.jsonl"),
                        help="rows whose images to place, with their recorded SHA-256")
    parser.add_argument("--archive", type=Path, default=None,
                        help=f"a local {ARCHIVE}; by default it is downloaded from {REPO_ID}@{REVISION}")
    parser.add_argument("--dest", type=Path, default=data_path("virl39k"),
                        help="directory the archive's images/ folder is placed in")
    parser.add_argument("--cache-dir", type=Path, default=None, help="download cache for the archive")
    args = parser.parse_args()

    wanted = wanted_images(args.rows)
    archive_path = args.archive or download_archive(args.cache_dir)
    with zipfile.ZipFile(archive_path) as archive:
        summary = place_images(archive, wanted, args.dest)
    failed = summary["missing_from_archive"] or summary["sha256_mismatch"]
    for key in ("missing_from_archive", "sha256_mismatch"):
        summary[f"{key}_examples"] = summary[key][:10]
        summary[key] = len(summary[key])
    summary["dest"] = str(args.dest)
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

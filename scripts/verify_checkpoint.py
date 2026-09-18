#!/usr/bin/env python3
"""Verify a merged checkpoint from disk: check the shard index against the files that
are present, hash every shard, and write a checksum manifest and a JSON record.

The shards are re-stat'ed after hashing, so a checkpoint still being written out is
reported rather than accepted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path: Path) -> tuple[int, int]:
    stat = path.stat()
    if stat.st_size <= 0:
        raise ValueError(f"checkpoint file is empty: {path}")
    return stat.st_size, stat.st_mtime_ns


def verify_checkpoint(checkpoint: Path) -> tuple[dict[str, Any], str]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError(f"checkpoint must be a real directory: {checkpoint}")
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"checkpoint index is absent: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise ValueError("checkpoint index must be a JSON object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index has no nonempty weight_map")
    shard_names = sorted(set(weight_map.values()))
    if any(
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or not name.endswith(".safetensors")
        for name in shard_names
    ):
        raise ValueError("checkpoint index contains an unsafe shard path")
    referenced = {checkpoint / name for name in shard_names}
    actual = set(checkpoint.glob("*.safetensors"))
    if actual != referenced:
        missing = sorted(path.name for path in referenced - actual)
        extra = sorted(path.name for path in actual - referenced)
        raise ValueError(f"checkpoint shard set mismatch: missing={missing}, extra={extra}")

    files = [index_path, *sorted(referenced)]
    before = {path: _snapshot(path) for path in files}
    records = [
        {
            "file": path.name,
            "size_bytes": before[path][0],
            "sha256": _sha256(path),
        }
        for path in files
    ]
    after = {path: _snapshot(path) for path in files}
    if before != after:
        raise RuntimeError("checkpoint changed during SHA256 verification")
    checksums = "".join(
        f"{record['sha256']}  {record['file']}\n" for record in records
    )
    payload = {
        "schema_version": "lwl.merged-checkpoint-verification.v1",
        "status": "pass",
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "checkpoint": str(checkpoint),
        "index_sha256": records[0]["sha256"],
        "tensor_count": len(weight_map),
        "shard_count": len(shard_names),
        "total_weight_bytes": sum(record["size_bytes"] for record in records[1:]),
        "files_stable_during_hash": True,
        "files": records,
        "checksum_manifest_sha256": hashlib.sha256(checksums.encode("ascii")).hexdigest(),
    }
    return payload, checksums


def _atomic_write(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verification artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-sha256", type=Path, required=True)
    args = parser.parse_args()
    payload, checksums = verify_checkpoint(args.checkpoint)
    _atomic_write(args.output_sha256, checksums)
    payload["checksum_manifest"] = str(args.output_sha256.resolve())
    _atomic_write(args.output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

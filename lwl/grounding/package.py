"""Package a source manifest into a distributable release: opaque pair and member ids, salted
content-addressed image names, shuffled member order and a separate answer key.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import secrets
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


# Schema tag written into every packaged manifest row.
SCHEMA_VERSION = "lwl.grounding-release.v1"
# Every released file gets this timestamp so that modification times carry no signal.
FIXED_MTIME = 946684800


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _opaque_id(salt: bytes, *parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(salt)
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:16]


def _png_bytes(path: Path, mode: str) -> bytes:
    with Image.open(path) as source:
        image = source.convert(mode)
        image.load()
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _content_name(content: bytes, salt: bytes) -> str:
    return hashlib.sha256(content + salt).hexdigest()[:16] + ".png"


def _write_content_addressed(directory: Path, content: bytes, salt: bytes) -> Path:
    path = directory / _content_name(content, salt)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"salted filename collision at {path}")
        return path
    path.write_bytes(content)
    return path


def _assert_private_path(path: Path, release_dir: Path, label: str) -> None:
    """Fail if a file that must stay private would be written inside the release directory."""
    try:
        path.resolve().relative_to(release_dir.resolve())
    except ValueError:
        return
    raise ValueError(f"{label} must be stored outside the release directory: {path}")


def _load_or_create_salt(path: Path) -> bytes:
    if path.exists():
        salt = path.read_bytes()
        if len(salt) < 16:
            raise ValueError("packaging salt must contain at least 16 bytes")
        return salt
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_bytes(32)
    path.write_bytes(salt)
    path.chmod(0o600)
    return salt


def package_manifest(
    source_manifest: str | Path,
    release_dir: str | Path,
    key_file: str | Path,
    salt_file: str | Path,
) -> dict[str, Any]:
    """Write the release directory and its answer key, and return a summary of both."""
    source_manifest = Path(source_manifest)
    release_dir = Path(release_dir)
    key_file = Path(key_file)
    salt_file = Path(salt_file)
    _assert_private_path(key_file, release_dir, "answer key")
    _assert_private_path(salt_file, release_dir, "salt")
    if release_dir.exists() and any(release_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty release directory: {release_dir}")

    image_dir = release_dir / "images"
    mask_dir = release_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    salt = _load_or_create_salt(salt_file)

    release_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    seen_member_ids: set[str] = set()
    for source in _read_jsonl(source_manifest):
        source_pair_id = str(source["pair_id"])
        pair_id = _opaque_id(salt, "pair", source_pair_id)
        if pair_id in seen_pair_ids:
            raise ValueError(f"duplicate packaged pair id: {pair_id}")
        seen_pair_ids.add(pair_id)

        members: list[dict[str, Any]] = []
        key_members: list[dict[str, Any]] = []
        for side in ("a", "b"):
            image_content = _png_bytes(Path(source[f"image_{side}_path"]), "RGB")
            mask_content = _png_bytes(Path(source[f"changed_region_mask_{side}"]), "L")
            image_path = _write_content_addressed(image_dir, image_content, salt)
            mask_path = _write_content_addressed(mask_dir, mask_content, salt)
            member_id = _opaque_id(salt, "member", source_pair_id, side)
            if member_id in seen_member_ids:
                raise ValueError(f"duplicate packaged member id: {member_id}")
            seen_member_ids.add(member_id)
            members.append(
                {
                    "member_id": member_id,
                    "image_path": image_path.relative_to(release_dir).as_posix(),
                    "mask_path": mask_path.relative_to(release_dir).as_posix(),
                    "image_sha256": hashlib.sha256(image_content).hexdigest(),
                    "mask_sha256": hashlib.sha256(mask_content).hexdigest(),
                }
            )
            key_members.append(
                {
                    "member_id": member_id,
                    "source_side": side,
                    "answer": str(source[f"answer_{side}"]),
                }
            )

        order_rng = random.Random(int(_opaque_id(salt, "order", source_pair_id), 16))
        order_rng.shuffle(members)
        release_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pair_id": pair_id,
                "question": str(source["question"]),
                "members": members,
            }
        )
        key_rows.append(
            {
                "pair_id": pair_id,
                "source_pair_id": source_pair_id,
                "category": source.get("category"),
                "template_id": source.get("template_id"),
                "catch_twin_id": source.get("catch_twin_id"),
                "members": key_members,
            }
        )

    order_rng = random.Random(int(hashlib.sha256(salt + b"manifest-order").hexdigest(), 16))
    order_rng.shuffle(release_rows)
    key_by_pair = {row["pair_id"]: row for row in key_rows}
    key_rows = [key_by_pair[row["pair_id"]] for row in release_rows]
    manifest_path = release_dir / "manifest.jsonl"
    _write_jsonl(manifest_path, release_rows)
    _write_jsonl(key_file, key_rows)

    release_files = [path for path in release_dir.rglob("*") if path.is_file()]
    for path in release_files + [key_file]:
        os.utime(path, (FIXED_MTIME, FIXED_MTIME))

    return {
        "source_manifest": str(source_manifest),
        "release_dir": str(release_dir),
        "manifest": str(manifest_path),
        "key_file": str(key_file),
        "salt_file": str(salt_file),
        "n_pairs": len(release_rows),
        "n_release_files": len(release_files),
        "fixed_mtime": FIXED_MTIME,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--salt-file", required=True)
    args = parser.parse_args()
    result = package_manifest(args.source_manifest, args.release_dir, args.key_file, args.salt_file)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Checks on a packaged grounding release: opaque ids and paths, equal image dimensions,
truthful change masks, stripped PNG metadata and no file-size signature that would give the
edited side away.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.stats import ks_2samp

from lwl.evaluation.metrics import match_tier


BASE_FORBIDDEN_TOKENS = {"side", "answer", "template", "left", "right"}
TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _error(errors: list[dict[str, str]], code: str, message: str, pair_id: str = "") -> None:
    errors.append({"code": code, "pair_id": pair_id, "message": message})


def png_chunks(path: Path) -> list[str]:
    """Chunk types of a PNG, in file order."""
    chunks: list[str] = []
    with path.open("rb") as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise ValueError(f"not a PNG file: {path}")
        while True:
            length_bytes = handle.read(4)
            if not length_bytes:
                break
            if len(length_bytes) != 4:
                raise ValueError(f"truncated PNG length: {path}")
            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = handle.read(4)
            if len(chunk_type) != 4:
                raise ValueError(f"truncated PNG chunk type: {path}")
            chunks.append(chunk_type.decode("ascii", errors="replace"))
            handle.seek(length + 4, 1)
            if chunk_type == b"IEND":
                break
    return chunks


def _has_forbidden_token(value: str, template_tokens: set[str]) -> bool:
    tokens = {token for token in TOKEN_SPLIT_RE.split(value.lower()) if token}
    if tokens & BASE_FORBIDDEN_TOKENS:
        return True
    return bool(tokens & template_tokens)


def _template_tokens(keys: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for row in keys:
        for token in TOKEN_SPLIT_RE.split(str(row.get("template_id", "")).lower()):
            if len(token) >= 4 and not re.fullmatch(r"v\d+", token):
                tokens.add(token)
    return tokens


def _member_key_map(key_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(member["member_id"]): member for member in key_row.get("members", [])}


def lint_package(
    release_dir: str | Path,
    key_file: str | Path,
    *,
    gross_ks_threshold: float = 0.8,
) -> dict[str, Any]:
    """Run every check over a packaged release and its answer key, and report the findings."""
    release_dir = Path(release_dir)
    manifest_path = release_dir / "manifest.jsonl"
    key_file = Path(key_file)
    rows = _read_jsonl(manifest_path)
    keys = _read_jsonl(key_file)
    key_by_pair = {str(row["pair_id"]): row for row in keys}
    errors: list[dict[str, str]] = []
    template_tokens = _template_tokens(keys)
    mtimes: set[int] = set()
    chunk_inventory: dict[str, list[str]] = {}
    sizes: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    order_counts = {"ab": 0, "ba": 0}
    seen_pair_ids: set[str] = set()
    seen_member_ids: set[str] = set()
    files_seen: set[Path] = set()

    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if pair_id in seen_pair_ids:
            _error(errors, "duplicate_pair_id", f"duplicate pair id {pair_id}", pair_id)
        seen_pair_ids.add(pair_id)
        if not re.fullmatch(r"[0-9a-f]{16}", pair_id):
            _error(errors, "nonopaque_id", f"pair id is not 16 lowercase hex: {pair_id}", pair_id)
        if _has_forbidden_token(pair_id, template_tokens):
            _error(errors, "leaky_id", f"forbidden token in pair id {pair_id}", pair_id)

        members = row.get("members", [])
        if len(members) != 2:
            _error(errors, "member_count", f"expected two members, found {len(members)}", pair_id)
            continue
        key_row = key_by_pair.get(pair_id)
        if key_row is None:
            _error(errors, "missing_key", "pair missing from private key", pair_id)
            continue
        member_keys = _member_key_map(key_row)
        if set(member_keys) != {str(member.get("member_id", "")) for member in members}:
            _error(errors, "key_member_mismatch", "release and key member ids differ", pair_id)
            continue

        source_order = "".join(str(member_keys[str(member["member_id"])]["source_side"]) for member in members)
        if source_order in order_counts:
            order_counts[source_order] += 1
        else:
            _error(errors, "invalid_source_side", f"invalid member source order {source_order}", pair_id)

        answer_a = next(member["answer"] for member in member_keys.values() if member["source_side"] == "a")
        answer_b = next(member["answer"] for member in member_keys.values() if member["source_side"] == "b")
        if match_tier(answer_a, answer_b) > 0 or match_tier(answer_b, answer_a) > 0:
            _error(errors, "answers_cross_match", f"answers are not distinguishable: {answer_a!r} vs {answer_b!r}", pair_id)

        image_arrays: list[np.ndarray] = []
        mask_arrays: list[np.ndarray] = []
        image_sizes: list[tuple[int, int]] = []
        for member in members:
            member_id = str(member.get("member_id", ""))
            if member_id in seen_member_ids:
                _error(errors, "duplicate_member_id", f"duplicate member id {member_id}", pair_id)
            seen_member_ids.add(member_id)
            if not re.fullmatch(r"[0-9a-f]{16}", member_id):
                _error(errors, "nonopaque_id", f"member id is not 16 lowercase hex: {member_id}", pair_id)
            if _has_forbidden_token(member_id, template_tokens):
                _error(errors, "leaky_id", f"forbidden token in member id {member_id}", pair_id)

            for field, expected_dir in (("image_path", "images"), ("mask_path", "masks")):
                relative = Path(str(member.get(field, "")))
                value = relative.as_posix()
                if relative.is_absolute() or ".." in relative.parts:
                    _error(errors, "unsafe_path", f"unsafe {field}: {value}", pair_id)
                    continue
                if len(relative.parts) != 2 or relative.parts[0] != expected_dir:
                    _error(errors, "nonuniform_path", f"unexpected {field} layout: {value}", pair_id)
                if not re.fullmatch(r"[0-9a-f]{16}\.png", relative.name):
                    _error(errors, "nonopaque_filename", f"filename is not salted opaque hex: {value}", pair_id)
                if _has_forbidden_token(value, template_tokens):
                    _error(errors, "leaky_path", f"forbidden token in {field}: {value}", pair_id)
                path = release_dir / relative
                if not path.is_file():
                    _error(errors, "missing_file", f"missing {field}: {value}", pair_id)
                    continue
                files_seen.add(path)
                mtimes.add(path.stat().st_mtime_ns)
                try:
                    chunks = png_chunks(path)
                except ValueError as exc:
                    _error(errors, "invalid_png", str(exc), pair_id)
                    continue
                chunk_inventory[value] = chunks
                ancillary = [chunk for chunk in chunks if chunk and chunk[0].islower()]
                if ancillary:
                    _error(errors, "png_ancillary_chunk", f"ancillary chunks {ancillary} in {value}", pair_id)

            image_path = release_dir / str(member["image_path"])
            mask_path = release_dir / str(member["mask_path"])
            if not image_path.is_file() or not mask_path.is_file():
                continue
            with Image.open(image_path) as image:
                image_arrays.append(np.asarray(image.convert("RGB")))
                image_sizes.append(image.size)
            with Image.open(mask_path) as mask:
                mask_arrays.append(np.asarray(mask.convert("L")))
            key_member = member_keys[member_id]
            sizes[str(key_row.get("template_id", ""))][str(key_member["source_side"])].append(image_path.stat().st_size)

        if len(image_arrays) == 2 and len(mask_arrays) == 2:
            if image_sizes[0] != image_sizes[1]:
                _error(errors, "dimension_mismatch", f"pair image dimensions differ: {image_sizes}", pair_id)
            if any(mask.shape != image_arrays[0].shape[:2] for mask in mask_arrays):
                _error(errors, "mask_dimension_mismatch", "mask dimensions do not match images", pair_id)
            elif image_arrays[0].shape == image_arrays[1].shape:
                changed = np.any(image_arrays[0] != image_arrays[1], axis=2)
                allowed = (mask_arrays[0] > 0) | (mask_arrays[1] > 0)
                outside_count = int(np.count_nonzero(changed & ~allowed))
                if outside_count:
                    _error(errors, "untruthful_mask", f"{outside_count} changed pixels lie outside masks", pair_id)

    if len(mtimes) > 1:
        _error(errors, "mtime_mismatch", f"release assets have {len(mtimes)} distinct mtimes")
    if len(rows) >= 4 and (order_counts["ab"] == 0 or order_counts["ba"] == 0):
        _error(errors, "member_order_not_randomized", f"member order counts are {order_counts}")

    ks_by_template: dict[str, dict[str, Any]] = {}
    for template_id, sides in sorted(sizes.items()):
        values_a = sides.get("a", [])
        values_b = sides.get("b", [])
        if values_a and values_b:
            result = ks_2samp(values_a, values_b, alternative="two-sided", method="auto")
            gross = len(values_a) >= 4 and float(result.statistic) >= gross_ks_threshold and float(result.pvalue) <= 0.05
            ks_by_template[template_id] = {
                "n_a": len(values_a),
                "n_b": len(values_b),
                "ks_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "gross_separation": gross,
            }
            if gross:
                _error(errors, "gross_file_size_separation", f"template {template_id}: KS={result.statistic:.3f}, p={result.pvalue:.4g}")

    checks = {
        "manifest_nonempty": bool(rows),
        "pair_and_member_ids_unique_opaque": not any(error["code"] in {"duplicate_pair_id", "duplicate_member_id", "nonopaque_id"} for error in errors),
        "paths_uniform_and_nonleaky": not any(error["code"] in {"leaky_id", "leaky_path", "unsafe_path", "nonuniform_path", "nonopaque_filename"} for error in errors),
        "files_exist_and_are_png": not any(error["code"] in {"missing_file", "invalid_png"} for error in errors),
        "timestamps_equalized": len(mtimes) <= 1,
        "dimensions_match": not any(error["code"] in {"dimension_mismatch", "mask_dimension_mismatch"} for error in errors),
        "png_ancillary_chunks_absent": not any(error["code"] == "png_ancillary_chunk" for error in errors),
        "masks_truthful": not any(error["code"] == "untruthful_mask" for error in errors),
        "answers_distinguishable": not any(error["code"] == "answers_cross_match" for error in errors),
        "member_order_randomized": not any(error["code"] == "member_order_not_randomized" for error in errors),
        "file_size_separation_not_gross": not any(error["code"] == "gross_file_size_separation" for error in errors),
        "private_key_complete": not any(error["code"] in {"missing_key", "key_member_mismatch", "invalid_source_side"} for error in errors),
    }
    status = all(checks.values())
    return {
        "status": status,
        "checks": checks,
        "errors": errors,
        "stats": {
            "n_pairs": len(rows),
            "n_members": len(seen_member_ids),
            "n_asset_files": len(files_seen),
            "distinct_mtimes": len(mtimes),
            "member_order_counts": order_counts,
            "file_size_ks_by_template": ks_by_template,
            "png_chunk_inventory": chunk_inventory,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = lint_package(args.release_dir, args.key_file)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if result["status"] else 1)


if __name__ == "__main__":
    main()

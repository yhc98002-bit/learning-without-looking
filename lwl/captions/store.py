"""Content-addressed store of question-blind image captions.

Rows are keyed by the sha256 of the image and must carry the fixed caption prompt, greedy
decoding and no question or answer text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "lwl.caption-store.v1"
CAPTION_PROMPT = (
    "Describe the image in one concise paragraph. Include visible text, labels, "
    "numbers, colors, shapes, counts, and spatial relations that could matter for answering questions."
)
CAPTION_PROMPT_SHA256 = hashlib.sha256(CAPTION_PROMPT.encode("utf-8")).hexdigest()
CAPTION_DECODING = {"temperature": 0.0, "top_p": 1.0, "n": 1}


def validate_caption_row(row: dict[str, Any]) -> None:
    """Check one stored row.

    The prompt, decoding and token budget must match, and a row carrying a question or answer
    field is rejected, so a caption can never smuggle in the item it belongs to.
    """
    digest = str(row.get("image_sha256", ""))
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported caption-store schema for image {digest}")
    if not str(row.get("caption", "")).strip():
        raise ValueError(f"empty caption for image {digest}")
    if row.get("caption_prompt") != CAPTION_PROMPT:
        raise ValueError(f"caption prompt is not the question-blind prompt for image {digest}")
    if row.get("caption_prompt_sha256") != CAPTION_PROMPT_SHA256:
        raise ValueError(f"caption prompt hash mismatch for image {digest}")
    if row.get("decoding") != CAPTION_DECODING:
        raise ValueError(f"caption decoding is not greedy decoding for image {digest}")
    if int(row.get("max_new_tokens", 0)) < 384:
        raise ValueError(f"caption token budget is below 384 for image {digest}")
    if not str(row.get("caption_model_path", "")).strip():
        raise ValueError(f"caption model path is missing for image {digest}")
    forbidden = sorted({"question", "problem", "answer", "answer_a", "answer_b"} & row.keys())
    if forbidden:
        raise ValueError(f"caption row contains question/answer fields for image {digest}: {forbidden}")


def load_validated_caption_prefix(
    resume_from: Path,
    items: list[dict[str, Any]],
    *,
    model_path: str,
    max_new_tokens: int,
    model_revision: str | None = None,
    tensor_parallel_width: int | None = None,
) -> list[str]:
    """Return the lines of an earlier run that can be reused.

    Each line is checked against the item it must cover and the settings it was produced with.
    """
    raw_lines = resume_from.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise ValueError(f"caption resume source is empty: {resume_from}")
    if any(not line.strip() for line in raw_lines):
        raise ValueError(f"caption resume source contains a blank row: {resume_from}")
    if len(raw_lines) > len(items):
        raise ValueError("caption resume source is longer than the selected shard")

    seen: set[str] = set()
    for line_number, (line, item) in enumerate(zip(raw_lines, items), start=1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"caption resume line {line_number} is not an object")
        validate_caption_row(row)
        digest = str(row["image_sha256"])
        if digest in seen:
            raise ValueError(f"duplicate image in caption resume source: {digest}")
        seen.add(digest)
        expected = {
            "image_sha256": item["image_sha256"],
            "image_path": item["image_path"],
            "duplicate_paths": item["duplicate_paths"],
            "caption_model_path": model_path,
            "max_new_tokens": max_new_tokens,
        }
        if model_revision is not None:
            expected["caption_model_revision"] = model_revision
        if tensor_parallel_width is not None:
            expected["tensor_parallel_width"] = tensor_parallel_width
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"caption resume mismatch at line {line_number} for {key}: "
                    f"expected {value!r}, found {row.get(key)!r}"
                )
    return raw_lines


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(input_dir: str | Path) -> list[dict[str, Any]]:
    """One entry per distinct image hash under the directory, with the duplicate paths listed."""
    input_dir = Path(input_dir)
    by_hash: dict[str, list[Path]] = {}
    for path in sorted(item for item in input_dir.rglob("*") if item.is_file()):
        try:
            digest = sha256_file(path)
        except OSError:
            continue
        by_hash.setdefault(digest, []).append(path)
    return [
        {
            "image_sha256": digest,
            "image_path": str(paths[0]),
            "duplicate_paths": [str(path) for path in paths[1:]],
        }
        for digest, paths in sorted(by_hash.items())
    ]


def select_shard(items: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return [item for index, item in enumerate(items) if index % num_shards == shard_index]


def merge_caption_rows(
    shard_paths: Iterable[str | Path],
    expected_hashes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge shard files into one row per image hash; all shards must share one caption contract."""
    by_hash: dict[str, dict[str, Any]] = {}
    model_paths: set[str] = set()
    prompt_hashes: set[str] = set()
    token_budgets: set[int] = set()
    for shard_path in shard_paths:
        with Path(shard_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                validate_caption_row(row)
                digest = str(row["image_sha256"])
                caption = str(row["caption"]).strip()
                if digest in by_hash and str(by_hash[digest]["caption"]).strip() != caption:
                    raise ValueError(f"conflicting captions for image hash {digest}")
                by_hash.setdefault(digest, row)
                model_paths.add(str(row["caption_model_path"]))
                prompt_hashes.add(str(row["caption_prompt_sha256"]))
                token_budgets.add(int(row["max_new_tokens"]))
    if not by_hash:
        raise ValueError("caption shards contain no rows")
    if len(model_paths) != 1 or len(prompt_hashes) != 1 or len(token_budgets) != 1:
        raise ValueError("caption shards mix model, prompt, or token-budget contracts")
    found = set(by_hash)
    missing = sorted((expected_hashes or set()) - found)
    extra = sorted(found - expected_hashes) if expected_hashes is not None else []
    if missing or extra:
        raise ValueError(f"caption hash coverage mismatch: missing={len(missing)} extra={len(extra)}")
    rows = [by_hash[digest] for digest in sorted(by_hash)]
    summary = {
        "schema_version": "lwl.caption-store-merge.v1",
        "n_images": len(rows),
        "caption_model_path": next(iter(model_paths)),
        "caption_prompt_sha256": next(iter(prompt_hashes)),
        "max_new_tokens": next(iter(token_budgets)),
        "decoding": CAPTION_DECODING,
        "expected_hashes": len(expected_hashes) if expected_hashes is not None else None,
        "coverage_complete": expected_hashes is None or found == expected_hashes,
    }
    return rows, summary

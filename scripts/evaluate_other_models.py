#!/usr/bin/env python3
"""Evaluate an InternVL3 or Gemma-3 checkpoint on the audit sample under one image
condition, writing one scored JSONL row per item.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.evaluation.conditioned_inputs import (
    build_conditioned_messages,
    load_caption_map,
    load_geometry_rows,
)
from lwl.evaluation.other_models import (
    NONQWEN_BACKENDS,
    create_nonqwen_adapter,
    nonqwen_runtime_metadata_valid,
)
from lwl.evaluation.prompt_contract import prompt_contract_metadata, response_satisfies_contract
from lwl.rewards.answer import PARSER_VERSION, answers_match, extract_answer_span


CONDITIONS = ("real", "none", "caption")
SCHEMA_VERSION = "lwl.nonqwen-blind-sample.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_shard(
    rows: list[dict[str, Any]], num_shards: int, shard_index: int, limit: int | None
) -> list[dict[str, Any]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard selection")
    selected = [row for ordinal, row in enumerate(rows) if ordinal % num_shards == shard_index]
    return selected[:limit] if limit is not None else selected


def score_prediction(prediction: str, ground_truth: str) -> dict[str, Any]:
    extracted = extract_answer_span(prediction)
    acc_final = bool(answers_match(extracted.span, ground_truth))
    contract_valid = bool(response_satisfies_contract(prediction))
    return {
        "extracted_answer": extracted.span,
        "extraction_level": extracted.extraction_level,
        "extraction_fallback_used": extracted.extraction_fallback_used,
        "extractor_valid": extracted.extractor_valid,
        "contract_valid": contract_valid,
        "acc_final": acc_final,
        "acc_strict": acc_final and contract_valid,
        "parser_version": PARSER_VERSION,
    }


def validate_resume_prefix(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    backend: str,
    condition: str,
    source_hash: str,
    caption_hash: str | None,
    max_new_tokens: int,
) -> list[str]:
    """Rows of an earlier partial run that this run may keep, checked item by item."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines) or len(lines) > len(rows):
        raise ValueError("resume source must be non-empty JSONL no longer than the selection")
    expected_decoding = {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "max_new_tokens": max_new_tokens,
    }
    for index, (line, source) in enumerate(zip(lines, rows), start=1):
        try:
            resumed = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid resume JSON at line {index}: {error}") from error
        expected = {
            "schema_version": SCHEMA_VERSION,
            "row_index": source["row_index"],
            "qid": source.get("qid"),
            "backend": backend,
            "condition": condition,
            "source_manifest_sha256": source_hash,
            "caption_store_sha256": caption_hash,
            "decoding": expected_decoding,
            "parser_version": PARSER_VERSION,
        }
        for key, value in expected.items():
            if resumed.get(key) != value:
                raise ValueError(
                    f"resume contract mismatch at line {index} for {key}: "
                    f"expected {value!r}, found {resumed.get(key)!r}"
                )
        if not nonqwen_runtime_metadata_valid(resumed.get("runtime"), backend):
            raise ValueError(f"resume contract mismatch at line {index} for runtime")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=NONQWEN_BACKENDS, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--format-prompt", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--caption-shards", type=Path, nargs="*")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.max_new_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("token and item limits must be positive")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    partial = Path(f"{args.output}.partial")
    if args.condition == "caption" and not args.caption_shards:
        raise ValueError("caption condition requires fixed caption shards")
    if args.condition != "caption" and args.caption_shards:
        raise ValueError("caption shards are only valid for caption condition")

    rows = load_geometry_rows(args.manifest, ("audit",))
    rows = select_shard(rows, args.num_shards, args.shard_index, args.limit)
    if not rows:
        raise ValueError("selected shard is empty")
    source_hash = sha256(args.manifest)
    format_hash = sha256(args.format_prompt)
    caption_files = sorted(args.caption_shards or [])
    caption_hash = (
        hashlib.sha256(
            "".join(f"{sha256(path)}  {path}\n" for path in caption_files).encode()
        ).hexdigest()
        if caption_files
        else None
    )
    captions = load_caption_map(caption_files) if caption_files else None
    if captions is not None:
        required = {
            str(image["sha256"])
            for row in rows
            for image in row.get("images", [])
        }
        missing = sorted(required - captions.keys())
        if missing:
            raise ValueError(f"fixed caption store misses {len(missing)} selected images")

    resumed = validate_resume_prefix(
        partial,
        rows,
        backend=args.backend,
        condition=args.condition,
        source_hash=source_hash,
        caption_hash=caption_hash,
        max_new_tokens=args.max_new_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if not partial.exists():
        partial.touch(exist_ok=False)
    if len(resumed) == len(rows):
        os.replace(partial, args.output)
        return

    adapter = create_nonqwen_adapter(
        args.backend,
        args.model_path,
        max_new_tokens=args.max_new_tokens,
    )
    format_prompt = args.format_prompt.read_text(encoding="utf-8")
    contract = prompt_contract_metadata()
    decoding = {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "max_new_tokens": args.max_new_tokens,
    }
    with partial.open("a", encoding="utf-8") as handle:
        for position, source in enumerate(rows[len(resumed) :], start=len(resumed) + 1):
            messages, image_paths = build_conditioned_messages(
                source,
                format_prompt,
                args.condition,
                args.cache_dir,
                captions=captions,
                noise_seed=0,
            )
            if args.condition in {"none", "caption"} and image_paths:
                raise AssertionError("an image-free condition materialized an image")
            prediction = adapter.generate(messages[0]["content"])
            output = {
                "schema_version": SCHEMA_VERSION,
                "split": source["split"],
                "row_index": source["row_index"],
                "qid": source.get("qid"),
                "problem": source["problem"],
                "ground_truth": source["answer"],
                "image_sha256": [
                    image["sha256"] for image in source.get("images", [])
                ],
                "source_metadata": source.get("metadata"),
                "backend": args.backend,
                "condition": args.condition,
                "prediction": prediction,
                "runtime": adapter.runtime_metadata(),
                "image_payload_count": len(image_paths),
                "source_manifest_sha256": source_hash,
                "format_prompt_sha256": format_hash,
                "caption_store_sha256": caption_hash,
                "decoding": decoding,
                **score_prediction(prediction, str(source["answer"])),
                **contract,
            }
            handle.write(json.dumps(output, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(
                json.dumps(
                    {
                        "backend": args.backend,
                        "condition": args.condition,
                        "processed": position,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
    os.replace(partial, args.output)


if __name__ == "__main__":
    main()

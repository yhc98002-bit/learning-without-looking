#!/usr/bin/env python3
"""Evaluate one checkpoint on the Geometry3K test split under a single image condition:
greedy vLLM decoding, scored with the training reward and the canonical scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_audit import _vllm_request
from lwl.audit.sampling import (
    CONDITIONS,
    build_conditioned_messages,
    load_caption_map,
    load_geometry_rows,
    score_greedy_item_pilot,
    vllm_multimodal_limits,
)
from lwl.evaluation.prompt_contract import (
    DEFAULT_PROMPT_CONTRACT,
    load_prompt_contract_from_run_manifest,
    prompt_contract_metadata,
)


ROW_SCHEMA_VERSION = "lwl.pilot-geo3k-step100-eval.v1"
FOLLOWUP_ROW_SCHEMA_VERSION = "lwl.pilot-followup-geo3k-checkpoint-eval.v1"
CHECKPOINT_ROW_SCHEMA_VERSION = "lwl.geo3k-long-horizon-checkpoint-eval.v1"
CHECKPOINT_EVAL_STEPS = frozenset({150, 200, 300, 400})
EVAL_MAX_TOKENS = 2048
EVAL_SEED = 20260710
EVAL_DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "n": 1,
    "max_tokens": EVAL_MAX_TOKENS,
    "seed": EVAL_SEED,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validated_resume_prefix(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    arm: str,
    condition: str,
    model_revision: str,
    checkpoint_index_sha256: str,
    source_manifest_sha256: str,
    source_training_manifest_sha256: str,
    global_step: int = 100,
    row_schema_version: str = ROW_SCHEMA_VERSION,
) -> list[str]:
    """Rows of an earlier partial run that this run may keep, checked item by item."""
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if not raw_lines or any(not line.strip() for line in raw_lines):
        raise ValueError("resume source must be non-empty JSONL without blank rows")
    if len(raw_lines) > len(rows):
        raise ValueError("resume source is longer than the test split")

    expected_static = {
        "schema_version": row_schema_version,
        "arm": arm,
        "global_step": global_step,
        "condition": condition,
        "model_revision": model_revision,
        "checkpoint_index_sha256": checkpoint_index_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_training_manifest_sha256": source_training_manifest_sha256,
        "prompt_contract_sha256": DEFAULT_PROMPT_CONTRACT.sha256,
        "decoding": EVAL_DECODING,
    }
    seen: set[tuple[str, int]] = set()
    for line_number, (raw_line, source) in enumerate(zip(raw_lines, rows), start=1):
        try:
            resumed = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid resume JSON at line {line_number}: {error}") from error
        identity = (str(resumed.get("split")), int(resumed.get("row_index", -1)))
        expected_identity = (str(source["split"]), int(source["row_index"]))
        if identity != expected_identity:
            raise ValueError(
                f"resume identity mismatch at line {line_number}: "
                f"expected {expected_identity}, found {identity}"
            )
        if identity in seen:
            raise ValueError(f"duplicate resume identity: {identity}")
        seen.add(identity)
        expected = {
            **expected_static,
            "problem": source["problem"],
            "ground_truth": source["answer"],
            "image_sha256": [image["sha256"] for image in source.get("images", [])],
        }
        for key, value in expected.items():
            if resumed.get(key) != value:
                raise ValueError(
                    f"resume contract mismatch at line {line_number} for {key}: "
                    f"expected {value!r}, found {resumed.get(key)!r}"
                )
        required = {
            "greedy_response",
            "training_reward",
            "acc_final",
            "acc_strict",
            "extractor_valid",
            "contract_valid",
            "canonical_eval_reward",
            "native_r1v_shadow_reward",
            "reward_disagreement_reason",
        }
        missing = sorted(required - resumed.keys())
        if missing:
            raise ValueError(f"resume row {line_number} lacks score fields: {missing}")
    return raw_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--format-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--source-training-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-index-sha256", required=True)
    parser.add_argument("--caption-shards", type=Path, nargs="*")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=EVAL_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--global-step", type=int, default=100)
    parser.add_argument("--row-schema-version")
    args = parser.parse_args()

    if args.max_tokens != EVAL_MAX_TOKENS or args.seed != EVAL_SEED:
        raise ValueError("decoding differs from the pinned token budget and seed")
    if args.batch_size <= 0 or args.max_model_len <= 0:
        raise ValueError("batch size and max model length must be positive")
    if args.row_schema_version == FOLLOWUP_ROW_SCHEMA_VERSION:
        if args.global_step not in {60, 100}:
            raise ValueError("the follow-up endpoint is global step 60 or 100")
        expected_schema_version = FOLLOWUP_ROW_SCHEMA_VERSION
    elif args.global_step == 100:
        expected_schema_version = ROW_SCHEMA_VERSION
    elif args.global_step in CHECKPOINT_EVAL_STEPS:
        expected_schema_version = CHECKPOINT_ROW_SCHEMA_VERSION
    else:
        raise ValueError("global step is not one of the evaluation endpoints")
    row_schema_version = args.row_schema_version or expected_schema_version
    if row_schema_version != expected_schema_version:
        raise ValueError("row schema version does not match the evaluation endpoint")
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    if args.condition == "caption" and not args.caption_shards:
        raise ValueError("the caption condition requires the question-blind caption store")
    if args.condition != "caption" and args.caption_shards:
        raise ValueError("caption shards are only valid for the caption condition")

    prompt_contract = load_prompt_contract_from_run_manifest(args.run_manifest)
    if prompt_contract.sha256 != DEFAULT_PROMPT_CONTRACT.sha256:
        raise ValueError("run manifest does not use the default prompt contract")
    rows = load_geometry_rows(args.manifest, splits=("test",), train_filter_ids=None)
    if len(rows) != 601:
        raise ValueError(f"the Geometry3K test split must contain 601 rows, found {len(rows)}")

    source_manifest_sha256 = _sha256(args.manifest)
    source_training_manifest_sha256 = _sha256(args.source_training_manifest)
    model_revision = str(args.model_path)
    resume_lines = (
        load_validated_resume_prefix(
            args.resume_from,
            rows,
            arm=args.arm,
            condition=args.condition,
            model_revision=model_revision,
            checkpoint_index_sha256=args.checkpoint_index_sha256,
            source_manifest_sha256=source_manifest_sha256,
            source_training_manifest_sha256=source_training_manifest_sha256,
            global_step=args.global_step,
            row_schema_version=row_schema_version,
        )
        if args.resume_from
        else []
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for line in resume_lines:
            handle.write(line + "\n")
        handle.flush()
    if len(resume_lines) == len(rows):
        print(json.dumps({"processed": len(rows), "total": len(rows), "resumed": len(rows)}))
        return

    captions = load_caption_map(args.caption_shards or []) if args.condition == "caption" else None
    format_prompt = args.format_prompt.read_text(encoding="utf-8")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    max_images = max((len(row.get("images", [])) for row in rows), default=0)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    llm = LLM(
        model=str(args.model_path),
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt=vllm_multimodal_limits(args.condition, max_images=max_images),
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        n=1,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    contract_metadata = prompt_contract_metadata(prompt_contract)
    format_prompt_sha256 = _sha256(args.format_prompt)

    with args.output.open("a", encoding="utf-8") as handle:
        for start in range(len(resume_lines), len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            requests = []
            opened_images: list[Image.Image] = []
            for row in batch:
                messages, image_paths = build_conditioned_messages(
                    row,
                    format_prompt,
                    args.condition,
                    args.cache_dir,
                    captions=captions,
                    noise_seed=args.seed,
                )
                request, opened = _vllm_request(processor, messages, image_paths)
                requests.append(request)
                opened_images.extend(opened)
            try:
                generated = llm.generate(requests, sampling, use_tqdm=False)
            finally:
                for image in opened_images:
                    image.close()

            for row, generation in zip(batch, generated):
                response = generation.outputs[0].text.strip()
                scored = score_greedy_item_pilot(
                    str(row["answer"]),
                    response,
                    prompt_contract,
                )
                output = {
                    "schema_version": row_schema_version,
                    "arm": args.arm,
                    "global_step": args.global_step,
                    "split": row["split"],
                    "row_index": row["row_index"],
                    "qid": row.get("qid"),
                    "problem": row["problem"],
                    "ground_truth": row["answer"],
                    "image_sha256": [image["sha256"] for image in row.get("images", [])],
                    "condition": args.condition,
                    "source_metadata": row.get("metadata"),
                    "source_manifest_sha256": source_manifest_sha256,
                    "source_training_manifest_sha256": source_training_manifest_sha256,
                    "format_prompt_sha256": format_prompt_sha256,
                    "model_revision": model_revision,
                    "checkpoint_index_sha256": args.checkpoint_index_sha256,
                    "greedy_response": response,
                    "decoding": EVAL_DECODING,
                    **scored,
                    **contract_metadata,
                }
                handle.write(json.dumps(output, sort_keys=True, ensure_ascii=True) + "\n")
                handle.flush()
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "processed": min(start + len(batch), len(rows)),
                        "resumed": len(resume_lines),
                        "total": len(rows),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

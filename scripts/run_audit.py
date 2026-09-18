#!/usr/bin/env python3
"""Run the visual-necessity audit for one image condition: one greedy answer plus 16
sampled rollouts at temperature 1 per item, scored and written as JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from lwl.audit.sampling import (
    CONDITIONS,
    PILOT_ROW_SCHEMA_VERSION,
    PILOT_SCORING_MODE,
    build_conditioned_messages,
    load_caption_map,
    load_geometry_rows,
    load_train_filter_ids,
    score_item_pilot,
    vllm_multimodal_limits,
)
from lwl.evaluation.prompt_contract import (
    DEFAULT_PROMPT_CONTRACT,
    load_prompt_contract_from_run_manifest,
    prompt_contract_metadata,
)
from lwl.rewards.answer import PARSER_VERSION
from lwl.rewards.standard import (
    DEFAULT_SYMBOLIC_GRADER_TIMEOUT_SECONDS,
    PILOT_REWARD_VERSION,
)


AUDIT_MAX_TOKENS = 2048
AUDIT_SAMPLE_COUNT = 16
AUDIT_SAMPLE_TEMPERATURE = 1.0
AUDIT_GROUP_SIZE = 5
AUDIT_FORMAT_WEIGHT = 0.5
AUDIT_SYMBOLIC_GRADER_TIMEOUT_SECONDS = DEFAULT_SYMBOLIC_GRADER_TIMEOUT_SECONDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_decoding(seed: int) -> dict[str, Any]:
    return {
        "greedy": {"temperature": 0.0, "top_p": 1.0, "n": 1},
        "sampled": {
            "temperature": AUDIT_SAMPLE_TEMPERATURE,
            "top_p": 1.0,
            "n": AUDIT_SAMPLE_COUNT,
        },
        "max_tokens": AUDIT_MAX_TOKENS,
        "seed": seed,
    }


def _vllm_request(
    processor: Any, messages: list[dict[str, Any]], image_paths: list[str]
) -> tuple[dict[str, Any], list[Image.Image]]:
    """One generation request plus the images it holds open; the caller closes them."""
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    opened = []
    for path in image_paths:
        with Image.open(path) as source:
            opened.append(source.convert("RGB"))
    request: dict[str, Any] = {"prompt": prompt}
    if opened:
        request["multi_modal_data"] = {"image": opened[0] if len(opened) == 1 else opened}
    return request, opened


def load_validated_resume_prefix(
    resume_from: Path,
    rows: list[dict[str, Any]],
    *,
    condition: str,
    batch_size: int,
    seed: int,
    source_manifest_sha256: str,
    train_filter_sha256: str | None,
) -> list[str]:
    """Rows of an earlier partial run that this run may keep, checked item by item."""
    raw_lines = resume_from.read_text(encoding="utf-8").splitlines()
    if not raw_lines or any(not line.strip() for line in raw_lines):
        raise ValueError("resume source must be non-empty JSONL without blank rows")
    if len(raw_lines) > len(rows):
        raise ValueError("resume source is longer than the selected corpus")
    if len(raw_lines) != len(rows) and len(raw_lines) % batch_size:
        raise ValueError("resume prefix is not aligned to the batch size")

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
            "schema_version": PILOT_ROW_SCHEMA_VERSION,
            "condition": condition,
            "problem": source["problem"],
            "ground_truth": source["answer"],
            "image_sha256": [image["sha256"] for image in source.get("images", [])],
            "scoring_mode": PILOT_SCORING_MODE,
            "pilot_reward_version": PILOT_REWARD_VERSION,
            "parser_version": PARSER_VERSION,
            "prompt_contract_sha256": DEFAULT_PROMPT_CONTRACT.sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "train_filter_sha256": train_filter_sha256,
            "decoding": _expected_decoding(seed),
        }
        for key, value in expected.items():
            if resumed.get(key) != value:
                raise ValueError(
                    f"resume contract mismatch at line {line_number} for {key}: "
                    f"expected {value!r}, found {resumed.get(key)!r}"
                )
        sampled = resumed.get("sampled_responses")
        if not isinstance(sampled, list) or len(sampled) != AUDIT_SAMPLE_COUNT:
            raise ValueError(f"resume sample count mismatch at line {line_number}")
        required = {
            "p_i_jeffreys",
            "q_i",
            "greedy_training_reward",
            "mean_sampled_training_reward",
            "canonical_p_sample",
            "sampled_reward_disagreement_reasons",
        }
        missing = sorted(required - resumed.keys())
        if missing:
            raise ValueError(f"resume row {line_number} lacks scoring fields: {missing}")
    return raw_lines


def _validate_decoding_contract(args: argparse.Namespace) -> None:
    """The audit's rows are comparable only at the pinned decoding and reward settings."""
    expected = {
        "max_tokens": AUDIT_MAX_TOKENS,
        "sample_count": AUDIT_SAMPLE_COUNT,
        "sample_temperature": AUDIT_SAMPLE_TEMPERATURE,
        "group_size": AUDIT_GROUP_SIZE,
        "format_weight": AUDIT_FORMAT_WEIGHT,
        "symbolic_grader_timeout_seconds": AUDIT_SYMBOLIC_GRADER_TIMEOUT_SECONDS,
    }
    actual = {key: getattr(args, key) for key in expected}
    if actual != expected:
        raise ValueError(f"decoding/reward settings differ: expected {expected}, found {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-filter-ids", type=Path)
    parser.add_argument("--format-prompt", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--caption-shards", type=Path, nargs="*")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=AUDIT_MAX_TOKENS)
    parser.add_argument("--sample-count", type=int, default=AUDIT_SAMPLE_COUNT)
    parser.add_argument(
        "--sample-temperature", type=float, default=AUDIT_SAMPLE_TEMPERATURE
    )
    parser.add_argument("--group-size", type=int, default=AUDIT_GROUP_SIZE)
    parser.add_argument("--format-weight", type=float, default=AUDIT_FORMAT_WEIGHT)
    parser.add_argument(
        "--symbolic-grader-timeout-seconds",
        type=float,
        default=AUDIT_SYMBOLIC_GRADER_TIMEOUT_SECONDS,
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    args = parser.parse_args()

    _validate_decoding_contract(args)
    if args.output.exists():
        raise FileExistsError(f"audit output already exists: {args.output}")
    if args.condition == "caption" and not args.caption_shards:
        raise ValueError("caption condition requires the caption store shards")
    if args.condition != "caption" and args.caption_shards:
        raise ValueError("caption shards are only valid for the caption condition")

    prompt_contract = load_prompt_contract_from_run_manifest(args.run_manifest)
    if prompt_contract.sha256 != DEFAULT_PROMPT_CONTRACT.sha256:
        raise ValueError("run manifest does not use the default prompt contract")
    train_filter_ids = load_train_filter_ids(args.train_filter_ids) if args.train_filter_ids else None
    train_filter_sha256 = _sha256(args.train_filter_ids) if args.train_filter_ids else None
    source_manifest_sha256 = _sha256(args.manifest)
    rows = load_geometry_rows(
        args.manifest,
        args.splits,
        train_filter_ids=train_filter_ids,
    )
    if not rows:
        raise ValueError("corpus selection is empty")
    captions = load_caption_map(args.caption_shards or []) if args.condition == "caption" else None
    format_prompt = args.format_prompt.read_text(encoding="utf-8")

    resume_lines = (
        load_validated_resume_prefix(
            args.resume_from,
            rows,
            condition=args.condition,
            batch_size=args.batch_size,
            seed=args.seed,
            source_manifest_sha256=source_manifest_sha256,
            train_filter_sha256=train_filter_sha256,
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
        print(json.dumps({"condition": args.condition, "processed": len(rows), "total": len(rows)}))
        return

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    max_images = max((len(row.get("images", [])) for row in rows), default=0)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt=vllm_multimodal_limits(args.condition, max_images=max_images),
    )
    greedy_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        n=1,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    sample_params = SamplingParams(
        temperature=args.sample_temperature,
        top_p=1.0,
        n=args.sample_count,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    decoding = _expected_decoding(args.seed)
    contract_metadata = prompt_contract_metadata(prompt_contract)

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
                greedy_outputs = llm.generate(requests, greedy_params, use_tqdm=False)
                sampled_outputs = llm.generate(requests, sample_params, use_tqdm=False)
            finally:
                for image in opened_images:
                    image.close()

            for row, greedy_output, sampled_output in zip(batch, greedy_outputs, sampled_outputs):
                greedy_response = greedy_output.outputs[0].text.strip()
                sampled_responses = [output.text.strip() for output in sampled_output.outputs]
                scored = score_item_pilot(
                    str(row["answer"]),
                    greedy_response,
                    sampled_responses,
                    args.group_size,
                    prompt_contract,
                    format_weight=args.format_weight,
                    symbolic_grader_timeout_seconds=args.symbolic_grader_timeout_seconds,
                )
                output = {
                    "schema_version": PILOT_ROW_SCHEMA_VERSION,
                    "split": row["split"],
                    "row_index": row["row_index"],
                    "qid": row.get("qid"),
                    "problem": row["problem"],
                    "ground_truth": row["answer"],
                    "image_sha256": [image["sha256"] for image in row.get("images", [])],
                    "condition": args.condition,
                    "source_metadata": row.get("metadata"),
                    "source_manifest_sha256": source_manifest_sha256,
                    "train_filter_sha256": train_filter_sha256,
                    "format_prompt_sha256": _sha256(args.format_prompt),
                    "greedy_response": greedy_response,
                    "sampled_responses": sampled_responses,
                    **scored,
                    **contract_metadata,
                    "decoding": decoding,
                }
                handle.write(json.dumps(output, sort_keys=True, ensure_ascii=True) + "\n")
                handle.flush()
            print(
                json.dumps(
                    {
                        "condition": args.condition,
                        "processed": min(start + len(batch), len(rows)),
                        "resumed": len(resume_lines),
                        "total": len(rows),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

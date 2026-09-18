#!/usr/bin/env python3
"""Evaluate one Qwen2.5-VL checkpoint on a pair manifest: greedy decoding under the
answer-tag contract for both members of each pair, scored and written as JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.evaluation.metrics import aggregate_pair_metrics, pair_score
from lwl.evaluation.image_conditions import IMAGE_MODES, materialize_image
from lwl.evaluation.prompt_contract import format_question
from lwl.paths import resolve_data_reference


def generate(model, processor, image_path: str, question: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": format_question(question)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    out = out[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def generate_text_only(model, processor, question: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": format_question(question)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    out = out[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--image-mode",
        choices=tuple(IMAGE_MODES) + ("no_image", "mismatched_real", "twin_counterfactual"),
        default="real",
    )
    parser.add_argument("--image-override-map", default=None)
    parser.add_argument("--image-cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    set_seed(args.seed)
    image_override = None
    image_override_sha256 = None
    if args.image_mode == "mismatched_real":
        if not args.image_override_map:
            raise ValueError("mismatched_real requires --image-override-map")
        override_path = Path(args.image_override_map)
        image_override_sha256 = hashlib.sha256(override_path.read_bytes()).hexdigest()
        image_override = json.loads(override_path.read_text(encoding="utf-8"))
    elif args.image_override_map:
        raise ValueError("--image-override-map is only valid for mismatched_real")

    out_path = Path(args.output)
    partial_out = Path(f"{out_path}.partial")
    if out_path.exists() or partial_out.exists():
        raise FileExistsError(f"predictions already exist: {out_path}")
    metrics_path = Path(args.metrics_output) if args.metrics_output else None
    partial_metrics = Path(f"{metrics_path}.partial") if metrics_path else None
    if metrics_path and (metrics_path.exists() or partial_metrics.exists()):
        raise FileExistsError(f"metrics already exist: {metrics_path}")

    rows = []
    manifest_dir = Path(args.manifest).parent
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx % args.num_shards != args.shard_index:
                continue
            rows.append(json.loads(line))
            if args.limit is not None and len(rows) >= args.limit:
                break

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.image_cache_dir) if args.image_cache_dir else out_path.parent.parent / f"{args.image_mode}_image_cache"
    scored = []
    with partial_out.open("x", encoding="utf-8") as out:
        for row in rows:
            row = dict(row)
            row["eval_image_mode"] = args.image_mode
            source_a = row["image_a_path"]
            source_b = row["image_b_path"]
            materialize_mode = args.image_mode
            if args.image_mode == "twin_counterfactual":
                source_a, source_b = row["image_b_path"], row["image_a_path"]
                materialize_mode = "real"
            elif args.image_mode == "mismatched_real":
                entry = image_override["per_pair"][str(row["pair_id"])]
                if str(entry["a"]) == str(row["image_a_path"]) or str(entry["b"]) == str(row["image_b_path"]):
                    raise ValueError(f"override equals own image for pair {row['pair_id']}")
                source_a, source_b = entry["a"], entry["b"]
                row["mismatched_source_pair_id"] = entry["source_pair_id"]
                row["image_override_map_sha256"] = image_override_sha256
                materialize_mode = "real"
            if args.image_mode == "no_image":
                row["eval_image_a_path"] = None
                row["eval_image_b_path"] = None
                row["noise_pair_shared"] = False
                row["prediction_a"] = generate_text_only(model, processor, row["question"], args.max_new_tokens)
                row["prediction_b"] = generate_text_only(model, processor, row["question"], args.max_new_tokens)
            else:
                condition_key = str(row["pair_id"]) if materialize_mode == "noise" else None
                image_a = materialize_image(
                    str(resolve_data_reference(source_a, manifest_dir)),
                    materialize_mode, cache_dir, args.noise_seed, condition_key=condition_key
                )
                image_b = materialize_image(
                    str(resolve_data_reference(source_b, manifest_dir)),
                    materialize_mode, cache_dir, args.noise_seed, condition_key=condition_key
                )
                row["eval_image_a_path"] = image_a
                row["eval_image_b_path"] = image_b
                row["noise_pair_shared"] = materialize_mode == "noise"
                row["prediction_a"] = generate(model, processor, image_a, row["question"], args.max_new_tokens)
                row["prediction_b"] = generate(model, processor, image_b, row["question"], args.max_new_tokens)
            row.update(pair_score(row))
            scored.append(row)
            out.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            out.flush()
    os.replace(partial_out, out_path)
    metrics = aggregate_pair_metrics(scored)
    metrics.update(
        {
            "image_mode": args.image_mode,
            "num_shards": float(args.num_shards),
            "shard_index": float(args.shard_index),
            "seed": args.seed,
            "noise_seed": args.noise_seed,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        assert partial_metrics is not None
        partial_metrics.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial_metrics, metrics_path)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()

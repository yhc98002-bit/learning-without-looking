#!/usr/bin/env python3
"""Run one benchmark with its images removed: greedy vLLM decoding on the text-only
prompts, scored into per-item rows and a metrics file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.evaluation.benchmark_audit import build_text_prompt, load_rows, score_predictions, write_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.metrics_output.exists():
        raise FileExistsError("benchmark audit outputs already exist")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = load_rows(config["input_tsv"], config["dataset_type"])

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(config["model_path"], trust_remote_code=True, local_files_only=True)
    prompts = []
    for row in rows:
        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": [{"type": "text", "text": build_text_prompt(row, config["dataset_type"])}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if "<|vision_start|>" in prompt or "<|image_pad|>" in prompt:
            raise ValueError(f"processor inserted a vision token for text-only row {row['index']}")
        prompts.append(prompt)

    model = LLM(
        model=config["model_path"],
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=config.get("max_model_len", 32768),
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.85),
        seed=config["seed"],
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        n=1,
        max_tokens=config["max_new_tokens"],
        seed=config["seed"],
    )
    generations = model.generate(prompts, sampling, use_tqdm=True)
    predictions = [generation.outputs[0].text.strip() for generation in generations]
    scored_rows, metrics = score_predictions(rows, predictions, config["dataset_type"])
    metrics.update(
        {
            "model_path": config["model_path"],
            "system_prompt": config["system_prompt"],
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
                "max_new_tokens": config["max_new_tokens"],
                "seed": config["seed"],
            },
            "input_tsv": config["input_tsv"],
        }
    )
    write_results(scored_rows, metrics, args.output, args.metrics_output)
    print(json.dumps(metrics["overall"], sort_keys=True))


if __name__ == "__main__":
    main()

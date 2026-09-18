#!/usr/bin/env python3
"""Evaluate an InternVL3 or Gemma-3 checkpoint on the grounding suite or its twin under one
image condition, and write the scored pairs with their metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.evaluation.metrics import (
    aggregate_pair_metrics,
    aggregate_pair_metrics_by_template,
    pair_accuracy_ci,
    pair_score,
)
from lwl.evaluation.other_models import (
    GROUNDING_CONDITIONS,
    NONQWEN_BACKENDS,
    create_nonqwen_adapter,
    grounding_content,
)
from lwl.evaluation.prompt_contract import prompt_contract_metadata
from lwl.rewards.answer import PARSER_VERSION


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_caption_pairs(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if not pair_id or pair_id in result:
            raise ValueError(f"missing or duplicate caption pair_id: {pair_id!r}")
        if not str(row.get("caption_a", "")).strip() or not str(
            row.get("caption_b", "")
        ).strip():
            raise ValueError(f"caption pair lacks a nonempty side: {pair_id}")
        result[pair_id] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=NONQWEN_BACKENDS, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--dataset-id",
        choices=("suite", "twin"),
        required=True,
        help="which instrument the manifest holds: the grounding suite or its twin",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--caption-input", type=Path)
    parser.add_argument("--condition", choices=GROUNDING_CONDITIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.condition == "caption" and args.caption_input is None:
        raise ValueError("caption condition requires --caption-input")
    if args.condition != "caption" and args.caption_input is not None:
        raise ValueError("--caption-input is only valid for caption condition")
    for output in (args.output, args.metrics_output):
        if output.exists() or Path(f"{output}.partial").exists():
            raise FileExistsError(f"output already exists: {output}")

    rows = load_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("selected manifest is empty")
    captions = load_caption_pairs(args.caption_input)
    row_ids = {str(row["pair_id"]) for row in rows}
    if args.condition == "caption":
        missing = sorted(row_ids - captions.keys())
        if missing:
            raise ValueError(f"caption input misses {len(missing)} selected pairs")

    adapter = create_nonqwen_adapter(
        args.backend,
        args.model_path,
        max_new_tokens=args.max_new_tokens,
    )
    contract = prompt_contract_metadata()
    partial_output = Path(f"{args.output}.partial")
    partial_metrics = Path(f"{args.metrics_output}.partial")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    scored = []
    with partial_output.open("x", encoding="utf-8") as handle:
        for source in rows:
            row = dict(source)
            caption_row = captions.get(str(row["pair_id"]))
            row["prediction_a"] = adapter.generate(
                grounding_content(row, "a", args.condition, caption_row)
            )
            row["prediction_b"] = adapter.generate(
                grounding_content(row, "b", args.condition, caption_row)
            )
            row["runtime"] = adapter.runtime_metadata()
            row.update(pair_score(row))
            row.update(contract)
            row.update(
                {
                    "eval_backend": args.backend,
                    "eval_dataset": args.dataset_id,
                    "eval_condition": args.condition,
                    "parser_version": PARSER_VERSION,
                    "decoding": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "n": 1,
                        "max_new_tokens": args.max_new_tokens,
                    },
                }
            )
            scored.append(row)
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
    metrics = aggregate_pair_metrics(scored)
    metrics["per_template"] = aggregate_pair_metrics_by_template(scored)
    ci_low, ci_high = pair_accuracy_ci(scored, n_boot=2000, seed=0)
    metrics["pair_accuracy_ci95_low"] = ci_low
    metrics["pair_accuracy_ci95_high"] = ci_high
    metrics.update(contract)
    metrics.update(
        {
            "backend": args.backend,
            "dataset_id": args.dataset_id,
            "condition": args.condition,
            "parser_version": PARSER_VERSION,
            "row_count": len(scored),
            "runtime": adapter.runtime_metadata(),
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
                "max_new_tokens": args.max_new_tokens,
            },
        }
    )
    partial_metrics.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial_output, args.output)
    os.replace(partial_metrics, args.metrics_output)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()

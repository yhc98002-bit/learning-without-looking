#!/usr/bin/env python3
"""Score a VLMEvalKit prediction workbook: one row per item with its extracted answer and
tier, plus overall, per-category and paired summaries.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.evaluation.benchmark_audit import (
    choice_payload,
    not_missing,
    score_mcq_prediction,
    score_open_prediction,
)
from lwl.evaluation.prompt_contract import (
    PromptContractLike,
    load_prompt_contract_from_legacy_config_run_manifest,
    load_prompt_contract_from_run_manifest,
    prompt_contract_metadata,
)
from lwl.rewards.answer import PARSER_VERSION


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "n": 0.0,
            "Acc_strict": math.nan,
            "Acc_final": math.nan,
            "Format_valid": math.nan,
            "Extractor_valid": math.nan,
            "Contract_valid": math.nan,
            "Ambiguous_rate": math.nan,
            "Extraction_fallback_rate": math.nan,
            "Scoring_error_rate": math.nan,
        }
    n = len(rows)
    return {
        "n": float(n),
        "Acc_strict": sum(row["acc_strict"] for row in rows) / n,
        "Acc_final": sum(row["acc_final"] for row in rows) / n,
        "Format_valid": sum(row["format_valid"] for row in rows) / n,
        "Extractor_valid": sum(row["extractor_valid"] for row in rows) / n,
        "Contract_valid": sum(row["contract_valid"] for row in rows) / n,
        "Ambiguous_rate": sum(row["ambiguous"] for row in rows) / n,
        "Extraction_fallback_rate": sum(row["extraction_fallback_used"] for row in rows) / n,
        "Scoring_error_rate": sum(row.get("scoring_error") is not None for row in rows) / n,
    }


def _aggregate_pairs(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows or not all(row.get("pair_id") is not None for row in rows):
        return None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    malformed = sorted(pair_id for pair_id, members in grouped.items() if len(members) != 2)
    if malformed:
        raise ValueError(f"paired benchmark contains non-binary groups: {malformed[:5]}")
    return {
        "n_pairs": float(len(grouped)),
        "Pair_accuracy": sum(all(member["acc_final"] for member in members) for members in grouped.values())
        / len(grouped),
        "Strict_pair_accuracy": sum(
            all(member["acc_strict"] for member in members) for members in grouped.values()
        )
        / len(grouped),
    }


def postprocess(
    input_path: Path,
    rows_output: Path,
    metrics_output: Path,
    prompt_contract: PromptContractLike = None,
    prompt_contract_resolution: str = "embedded-run-manifest",
) -> dict[str, Any]:
    frame = pd.read_excel(input_path)
    required = {"index", "answer", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"VLMEval workbook missing columns: {sorted(missing)}")
    scored_rows: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in frame.to_dict(orient="records"):
        choices = choice_payload(raw)
        if choices is None:
            labels: list[str] = []
            scored = score_open_prediction(raw["prediction"], raw["answer"], prompt_contract)
            scoring_contract = "open_final_span"
            gold = str(raw["answer"])
        else:
            labels, options, gold = choices
            scored = score_mcq_prediction(
                raw["prediction"], gold, labels, options, prompt_contract
            )
            scoring_contract = "multiple_choice_final_span"
        category = raw.get("category")
        if not not_missing(category):
            category = raw.get("task", "unknown")
        record = {
            "index": str(raw["index"]),
            "gold": gold,
            "gold_value": str(raw["answer"]),
            "prediction": str(raw["prediction"]),
            "category": str(category),
            "l2_category": str(raw.get("l2_category", "unknown")),
            "question_type": str(raw.get("question_type", "unknown")),
            "answer_type": str(raw.get("answer_type", "unknown")),
            "scoring_contract": scoring_contract,
            "option_labels": labels,
            "pair_id": str(raw["pair_id"]) if not_missing(raw.get("pair_id")) else None,
            "pair_member": str(raw["pair_member"]) if not_missing(raw.get("pair_member")) else None,
            "visual_input": str(raw["visual_input"]) if not_missing(raw.get("visual_input")) else None,
            "set_id": str(raw["set_id"]) if not_missing(raw.get("set_id")) else None,
            "figure_id": str(raw["figure_id"]) if not_missing(raw.get("figure_id")) else None,
            "question_id": str(raw["question_id"]) if not_missing(raw.get("question_id")) else None,
            **scored,
        }
        scored_rows.append(record)
        by_category[record["category"]].append(record)
    payload = {
        "schema_version": "lwl.vlmeval-unified-scores.v2",
        "source_workbook": str(input_path),
        "parser_version": PARSER_VERSION,
        "prompt_contract_resolution": prompt_contract_resolution,
        **prompt_contract_metadata(prompt_contract),
        "overall": _aggregate(scored_rows),
        "paired": _aggregate_pairs(scored_rows),
        "per_category": {key: _aggregate(value) for key, value in sorted(by_category.items())},
    }
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    with rows_output.open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--allow-legacy-config-contract", action="store_true")
    args = parser.parse_args()
    if args.run_manifest is None:
        if args.allow_legacy_config_contract:
            raise ValueError("legacy config contract resolution requires --run-manifest")
        contract = None
        resolution = "default-contract"
    elif args.allow_legacy_config_contract:
        contract = load_prompt_contract_from_legacy_config_run_manifest(
            args.run_manifest, Path(__file__).resolve().parents[1]
        )
        resolution = "legacy-hash-pinned-run-config"
    else:
        contract = load_prompt_contract_from_run_manifest(args.run_manifest)
        resolution = "embedded-run-manifest"
    payload = postprocess(
        args.input,
        args.rows_output,
        args.metrics_output,
        contract,
        prompt_contract_resolution=resolution,
    )
    print(json.dumps(payload["overall"], sort_keys=True))


if __name__ == "__main__":
    main()

"""Benchmark audit: the public benchmarks re-asked with the image removed.

Holds the text-only prompt builders, which mirror the VLMEvalKit prompts minus the image,
and the multiple-choice and open-answer scorers shared with the with-image scoring.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lwl.evaluation.metrics import match_tier
from lwl.evaluation.prompt_contract import (
    PromptContractLike,
    prompt_contract_metadata,
    response_satisfies_contract,
)
from lwl.rewards.answer import PARSER_VERSION, extract_answer_span


PROTOCOL_VERSION = "lwl.layer1-image-removed.v1"


def mmstar_text_prompt(row: dict[str, Any]) -> str:
    options = {
        label: row[label]
        for label in string.ascii_uppercase
        if label in row and not pd.isna(row[label])
    }
    prompt = f"Question: {row['question']}\n"
    if options:
        prompt += "Options:\n"
        prompt += "".join(f"{label}. {value}\n" for label, value in options.items())
        prompt += "Please select the correct answer from the options above. \n"
    return prompt


def mathvista_text_prompt(row: dict[str, Any]) -> str:
    return str(row["question"])


# Each builder mirrors the VLMEvalKit class that produced the paired with-image column, so
# the text-only prompt is that prompt minus the image messages and nothing else:
#   ImageMCQDataset.build_prompt  -> "Question:" + "Options:" block + select instruction,
#     used for MMStar, BLINK and MMVP.
#   ImageBaseDataset.build_prompt -> the question text verbatim, used for HallusionBench
#     and MathVerse, which carry any options inside the question text.
#   MMMUDataset.build_prompt      -> ImageMCQDataset.build_prompt followed by split_MMMU,
#     which consumes the "<image N>" markers as it interleaves the images, so the text-only
#     mirror deletes them. Plain ImageMCQDataset leaves such a marker in its text, which is
#     why the other builders keep it.
_MCQ_OPTION_BLOCK_TYPES = frozenset({"mmstar", "blink", "mmvp"})
_QUESTION_VERBATIM_TYPES = frozenset({"mathvista", "hallusionbench", "mathverse"})
_MMMU_IMAGE_MARKER = re.compile(r"<image\s+\d+>")


def mmmu_text_prompt(row: dict[str, Any]) -> str:
    return _MMMU_IMAGE_MARKER.sub("", mmstar_text_prompt(row))


def build_text_prompt(row: dict[str, Any], dataset_type: str) -> str:
    if dataset_type in _MCQ_OPTION_BLOCK_TYPES:
        return mmstar_text_prompt(row)
    if dataset_type in _QUESTION_VERBATIM_TYPES:
        return mathvista_text_prompt(row)
    if dataset_type == "mmmu":
        return mmmu_text_prompt(row)
    raise ValueError(f"unsupported dataset type: {dataset_type}")


def load_rows(path: str | Path, dataset_type: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t")
    required = {"index", "question", "answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset_type} input missing columns: {sorted(missing)}")
    rows = frame.to_dict(orient="records")
    for row in rows:
        prompt = build_text_prompt(row, dataset_type)
        if "<image>" in prompt or "<|vision_" in prompt:
            raise ValueError(f"row {row['index']} retains an image token in the text-only prompt")
    return rows


def _explicit_choice_labels(span: str, labels: list[str]) -> list[str]:
    stripped = span.strip()
    upper = stripped.upper()
    if upper in labels:
        return [upper]
    label_class = "".join(re.escape(label) for label in labels)
    prefixed = re.match(
        rf"^(?:option|answer)\s*[\[(]?([{label_class}])[\])]?\s*(?:[.:,-]|$)", stripped, re.IGNORECASE
    )
    marked = re.match(rf"^[\[(]?([{label_class}])[\]).:,-]", stripped)
    if prefixed or marked:
        return [(prefixed or marked).group(1).upper()]
    mentioned = sorted(set(re.findall(rf"(?<![A-Za-z0-9])([{label_class}])(?![A-Za-z0-9])", stripped)))
    return [label.upper() for label in mentioned]


def score_mcq_prediction(
    prediction: Any,
    gold: Any,
    labels: list[str],
    options: dict[str, Any] | None = None,
    prompt_contract: PromptContractLike = None,
) -> dict[str, Any]:
    """Score one multiple-choice answer: an explicit label wins, otherwise the option text
    that matches at the highest tier."""
    extracted = extract_answer_span(prediction)
    explicit = _explicit_choice_labels(extracted.span, labels)
    if explicit:
        tiers = {label: (2 if label in explicit else 0) for label in labels}
        winners = explicit
    else:
        options = options or {label: label for label in labels}
        tiers = {
            label: match_tier(extracted.span, options[label])
            if len(str(options[label]).strip()) > 1
            else 0
            for label in labels
        }
        highest = max(tiers.values(), default=0)
        winners = sorted(label for label, tier in tiers.items() if tier == highest and tier > 0)
    if isinstance(gold, (list, tuple, set)):
        normalized_gold = sorted({str(label).strip().upper() for label in gold})
    else:
        normalized_gold = [str(gold).strip().upper()]
    invalid_gold = not normalized_gold or any(
        label not in labels for label in normalized_gold
    )
    ambiguous = len(winners) > 1 and not (explicit and sorted(winners) == normalized_gold)
    acc_final = not invalid_gold and sorted(winners) == normalized_gold
    contract_valid = response_satisfies_contract(prediction, prompt_contract)
    return {
        "extracted_answer": extracted.span,
        "extraction_level": extracted.extraction_level,
        "extraction_fallback_used": extracted.extraction_fallback_used,
        "extractor_valid": extracted.extractor_valid,
        "contract_valid": contract_valid,
        "format_valid": contract_valid,
        "parser_version": PARSER_VERSION,
        **prompt_contract_metadata(prompt_contract),
        "ambiguous": ambiguous,
        "winning_labels": winners,
        "gold_labels": normalized_gold,
        "scoring_error": "gold_label_missing_from_options" if invalid_gold else None,
        "match_tiers": tiers,
        "acc_final": acc_final,
        "acc_strict": contract_valid and acc_final,
    }


def score_open_prediction(
    prediction: Any, gold: Any, prompt_contract: PromptContractLike = None
) -> dict[str, Any]:
    """Score one open answer against its gold at the tiered matcher."""
    extracted = extract_answer_span(prediction)
    tier = match_tier(extracted.span, gold)
    acc_final = tier > 0
    contract_valid = response_satisfies_contract(prediction, prompt_contract)
    return {
        "extracted_answer": extracted.span,
        "extraction_level": extracted.extraction_level,
        "extraction_fallback_used": extracted.extraction_fallback_used,
        "extractor_valid": extracted.extractor_valid,
        "contract_valid": contract_valid,
        "format_valid": contract_valid,
        "parser_version": PARSER_VERSION,
        **prompt_contract_metadata(prompt_contract),
        "ambiguous": False,
        "scoring_error": None,
        "winning_labels": ["gold"] if acc_final else [],
        "match_tiers": {"gold": tier},
        "acc_final": acc_final,
        "acc_strict": contract_valid and acc_final,
    }


def not_missing(value: Any) -> bool:
    """False for None and for the missing values pandas produces."""
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    return bool(not missing) if isinstance(missing, (bool, np.bool_)) else True


def choice_payload(raw: dict[str, Any]) -> tuple[list[str], dict[str, Any], str | list[str]] | None:
    """Option labels, option texts and the gold label of a row, or None if it is open-answer."""
    labels = [label for label in string.ascii_uppercase if label in raw and not_missing(raw[label])]
    options = {label: raw[label] for label in labels}
    if not labels:
        serialized = raw.get("choices")
        if isinstance(serialized, str) and serialized.strip():
            try:
                parsed = ast.literal_eval(serialized)
            except (SyntaxError, ValueError):
                parsed = []
        elif isinstance(serialized, (list, tuple)):
            parsed = list(serialized)
        else:
            parsed = []
        labels = list(string.ascii_uppercase[: len(parsed)])
        options = dict(zip(labels, parsed))
    if not labels:
        return None

    answer_options = raw.get("answer_options")
    parsed_answer_options: list[str] = []
    if not_missing(answer_options) and str(answer_options).strip():
        if isinstance(answer_options, str):
            try:
                parsed = ast.literal_eval(answer_options)
            except (SyntaxError, ValueError):
                parsed = []
        else:
            parsed = answer_options
        if isinstance(parsed, (list, tuple, set)):
            parsed_answer_options = [str(label).strip().upper() for label in parsed]
    answer_option = raw.get("answer_option")
    if parsed_answer_options:
        unknown = sorted(set(parsed_answer_options) - set(labels))
        if unknown:
            raise ValueError(f"answer_options contains unknown labels: {unknown}")
        gold: str | list[str] = sorted(set(parsed_answer_options))
    elif not_missing(answer_option):
        gold = str(answer_option).strip().upper()
    else:
        raw_gold = str(raw["answer"]).strip()
        raw_gold_label = raw_gold.upper()
        if raw_gold_label in labels:
            gold = raw_gold_label
        else:
            matching = [
                label for label, option in options.items() if str(option).strip() == raw_gold
            ]
            gold = matching[0] if len(matching) == 1 else raw_gold_label
    return labels, options, gold


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
        }
    count = len(rows)
    return {
        "n": float(count),
        "Acc_strict": sum(row["acc_strict"] for row in rows) / count,
        "Acc_final": sum(row["acc_final"] for row in rows) / count,
        "Format_valid": sum(row["format_valid"] for row in rows) / count,
        "Extractor_valid": sum(row["extractor_valid"] for row in rows) / count,
        "Contract_valid": sum(row["contract_valid"] for row in rows) / count,
        "Ambiguous_rate": sum(row["ambiguous"] for row in rows) / count,
        "Extraction_fallback_rate": sum(row["extraction_fallback_used"] for row in rows) / count,
    }


def score_predictions(
    rows: list[dict[str, Any]],
    predictions: Iterable[str],
    dataset_type: str,
    prompt_contract: PromptContractLike = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = list(predictions)
    if len(rows) != len(predictions):
        raise ValueError("prediction count does not match the input rows")
    scored_rows: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        if dataset_type == "mmstar":
            labels = [
                label
                for label in string.ascii_uppercase
                if label in row and not pd.isna(row[label])
            ]
            options = {label: row[label] for label in labels}
            gold = str(row["answer"]).strip().upper()
            scored = score_mcq_prediction(
                prediction, gold, labels, options, prompt_contract
            )
            category = str(row.get("category", "unknown"))
            grading_contract = "multiple_choice_final_span"
        else:
            payload = choice_payload(row)
            if payload is None:
                labels = []
                gold = str(row["answer"])
                scored = score_open_prediction(prediction, gold, prompt_contract)
                grading_contract = "open_final_span"
            else:
                labels, options, gold = payload
                scored = score_mcq_prediction(
                    prediction, gold, labels, options, prompt_contract
                )
                grading_contract = "multiple_choice_final_span"
            category = row.get("category")
            if not not_missing(category):
                category = row.get("task", "unknown")
            category = str(category)
        prompt = build_text_prompt(row, dataset_type)
        record = {
            "index": str(row["index"]),
            "dataset_type": dataset_type,
            "image_removed": True,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "gold": gold,
            "gold_value": str(row["answer"]),
            "prediction": prediction,
            "category": category,
            "scoring_contract": grading_contract,
            "option_labels": labels,
            **scored,
        }
        scored_rows.append(record)
        by_category[category].append(record)
    metrics = {
        "schema_version": PROTOCOL_VERSION,
        "dataset_type": dataset_type,
        "image_removed": True,
        "parser_version": PARSER_VERSION,
        **prompt_contract_metadata(prompt_contract),
        "overall": _aggregate(scored_rows),
        "per_category": {key: _aggregate(value) for key, value in sorted(by_category.items())},
    }
    return scored_rows, metrics


def write_results(rows: list[dict[str, Any]], metrics: dict[str, Any], output: Path, metrics_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

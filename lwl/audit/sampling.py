"""Scoring of the visual-necessity audit: one greedy answer plus sampled rollouts per item
and per condition, turned into the per-item success rates the audit records.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from lwl.evaluation.conditioned_inputs import (
    CONDITIONS,
    build_conditioned_messages,
    load_caption_map,
    load_geometry_rows,
)
from lwl.evaluation.prompt_contract import (
    DEFAULT_PROMPT_CONTRACT,
    PromptContractLike,
    prompt_contract_metadata,
    response_satisfies_contract,
)
from lwl.rewards.answer import PARSER_VERSION, answer_reward, extract_answer_span
from lwl.rewards.standard import (
    DEFAULT_SYMBOLIC_GRADER_TIMEOUT_SECONDS,
    PILOT_REWARD_VERSION,
    REASON_CODES,
    SYMBOLIC_GRADER_GUARD_VERSION,
    compute_score as pilot_compute_score,
)


PILOT_SCORING_MODE = "pilot-reward-v1+canonical-v2"
PILOT_ROW_SCHEMA_VERSION = "lwl.blind-solvability-pilot.v1"


def vllm_multimodal_limits(condition: str, max_images: int = 1) -> dict[str, int]:
    """Per-request image budget: the conditions that remove the image allow none."""
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported audit condition: {condition}")
    if max_images < 0:
        raise ValueError("max_images cannot be negative")
    return (
        {"image": max_images, "video": 0}
        if condition in {"real", "gray", "noise"}
        else {"image": 0, "video": 0}
    )


def load_train_filter_ids(path: str | Path) -> set[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("train filter must be a non-empty JSON list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in payload):
        raise ValueError("train filter IDs must be non-negative integers")
    values = set(payload)
    if len(values) != len(payload):
        raise ValueError("train filter contains duplicate IDs")
    return values


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from c correct answers among n samples."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("pass@k requires 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - index) / (n - index) for index in range(k))


def jeffreys_smoothed_probability(n: int, c: int) -> float:
    if n <= 0 or not 0 <= c <= n:
        raise ValueError("Jeffreys smoothing requires n > 0 and 0 <= c <= n")
    return (c + 0.5) / (n + 1.0)


def mixed_group_probability(probability: float, group_size: int) -> float:
    """Probability that a group of `group_size` samples is neither all correct nor all wrong."""
    if not 0.0 <= probability <= 1.0 or group_size <= 0:
        raise ValueError("mixed-group probability requires p in [0, 1] and group_size > 0")
    return 1.0 - probability**group_size - (1.0 - probability) ** group_size


def score_greedy_item_pilot(
    gold: str,
    response: str,
    prompt_contract: PromptContractLike = None,
    *,
    format_weight: float = 0.5,
    symbolic_grader_timeout_seconds: float = DEFAULT_SYMBOLIC_GRADER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Score one greedy response with the training reward and the evaluation grader."""

    contract_metadata = prompt_contract_metadata(prompt_contract)
    if contract_metadata["prompt_contract_sha256"] != DEFAULT_PROMPT_CONTRACT.sha256:
        raise ValueError("this scoring path requires the default prompt contract")

    inherited_shadow_path = os.environ.pop("LWL_REWARD_SHADOW_LOG", None)
    try:
        pilot_score = pilot_compute_score(
            {"response": response, "ground_truth": gold},
            format_weight=format_weight,
            require_shadow_log=False,
            symbolic_grader_timeout_seconds=symbolic_grader_timeout_seconds,
        )
    finally:
        if inherited_shadow_path is not None:
            os.environ["LWL_REWARD_SHADOW_LOG"] = inherited_shadow_path

    inverse_reasons = {value: key for key, value in REASON_CODES.items()}
    reason_code = float(pilot_score["reward_disagreement_reason_code"])
    if reason_code not in inverse_reasons:
        raise ValueError(f"unknown pilot reward disagreement reason code: {reason_code}")
    extracted = extract_answer_span(response)
    contract_valid = response_satisfies_contract(response, prompt_contract)
    acc_final = bool(pilot_score["accuracy"])
    canonical_correct = bool(answer_reward(response, gold))
    return {
        "scoring_mode": PILOT_SCORING_MODE,
        "pilot_reward_version": PILOT_REWARD_VERSION,
        "symbolic_grader_guard_version": SYMBOLIC_GRADER_GUARD_VERSION,
        "symbolic_grader_timeout_seconds": symbolic_grader_timeout_seconds,
        "format_weight": format_weight,
        "training_reward": float(pilot_score["training_reward"]),
        "pilot_accuracy_reward": float(pilot_score["accuracy"]),
        "format_reward": float(pilot_score["format"]),
        "native_r1v_shadow_reward": float(pilot_score["native_r1v_shadow_reward"]),
        "native_r1v_shadow_valid": bool(pilot_score["native_r1v_shadow_valid"]),
        "canonical_eval_reward": float(pilot_score["canonical_eval_reward"]),
        "canonical_correct": canonical_correct,
        "reward_disagreement_reason": inverse_reasons[reason_code],
        "extracted_answer": extracted.span,
        "extractor_valid": extracted.extractor_valid,
        "contract_valid": contract_valid,
        "acc_final": acc_final,
        "acc_strict": contract_valid and acc_final,
        "parser_version": PARSER_VERSION,
        **contract_metadata,
    }


def score_item_pilot(
    gold: str,
    greedy_response: str,
    sampled_responses: list[str],
    group_size: int,
    prompt_contract: PromptContractLike = None,
    *,
    format_weight: float = 0.5,
    symbolic_grader_timeout_seconds: float = DEFAULT_SYMBOLIC_GRADER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Score one item's greedy and sampled responses with the training reward and the
    evaluation grader, and derive the item's success rates."""

    if not sampled_responses:
        raise ValueError("sampled responses cannot be empty")
    contract_metadata = prompt_contract_metadata(prompt_contract)
    if contract_metadata["prompt_contract_sha256"] != DEFAULT_PROMPT_CONTRACT.sha256:
        raise ValueError("this scoring path requires the default prompt contract")

    responses = [greedy_response, *sampled_responses]
    inherited_shadow_path = os.environ.pop("LWL_REWARD_SHADOW_LOG", None)
    try:
        pilot_scores = [
            pilot_compute_score(
                {"response": response, "ground_truth": gold},
                format_weight=format_weight,
                require_shadow_log=False,
                symbolic_grader_timeout_seconds=symbolic_grader_timeout_seconds,
            )
            for response in responses
        ]
    finally:
        if inherited_shadow_path is not None:
            os.environ["LWL_REWARD_SHADOW_LOG"] = inherited_shadow_path
    inverse_reasons = {value: key for key, value in REASON_CODES.items()}
    reason_names = [
        inverse_reasons[float(score["reward_disagreement_reason_code"])] for score in pilot_scores
    ]
    canonical_correct = [bool(answer_reward(response, gold)) for response in responses]
    extracted = [extract_answer_span(response) for response in responses]
    contract_valid = [
        response_satisfies_contract(response, prompt_contract) for response in responses
    ]

    greedy_score = pilot_scores[0]
    sampled_scores = pilot_scores[1:]
    sampled_correct = [bool(score["accuracy"]) for score in sampled_scores]
    n = len(sampled_correct)
    c = sum(sampled_correct)
    p_raw = c / n
    p_i = jeffreys_smoothed_probability(n, c)
    sampled_training_rewards = [float(score["training_reward"]) for score in sampled_scores]
    sampled_format_rewards = [float(score["format"]) for score in sampled_scores]
    sampled_native_rewards = [
        float(score["native_r1v_shadow_reward"]) for score in sampled_scores
    ]
    sampled_native_valid = [
        bool(score["native_r1v_shadow_valid"]) for score in sampled_scores
    ]
    sampled_canonical_rewards = [
        float(score["canonical_eval_reward"]) for score in sampled_scores
    ]
    canonical_sample_count = sum(canonical_correct[1:])

    return {
        "scoring_mode": PILOT_SCORING_MODE,
        "pilot_reward_version": PILOT_REWARD_VERSION,
        "symbolic_grader_guard_version": SYMBOLIC_GRADER_GUARD_VERSION,
        "symbolic_grader_timeout_seconds": symbolic_grader_timeout_seconds,
        "format_weight": format_weight,
        "p_greedy": float(greedy_score["accuracy"]),
        "greedy_correct": bool(greedy_score["accuracy"]),
        "greedy_training_reward": float(greedy_score["training_reward"]),
        "greedy_format_reward": float(greedy_score["format"]),
        "greedy_native_r1v_shadow_reward": float(greedy_score["native_r1v_shadow_reward"]),
        "greedy_native_r1v_shadow_valid": bool(greedy_score["native_r1v_shadow_valid"]),
        "greedy_canonical_correct": canonical_correct[0],
        "greedy_reward_disagreement_reason": reason_names[0],
        "greedy_extracted_answer": extracted[0].span,
        "greedy_extractor_valid": extracted[0].extractor_valid,
        "greedy_contract_valid": contract_valid[0],
        "greedy_format_valid": contract_valid[0],
        "greedy_acc_strict": contract_valid[0] and bool(greedy_score["accuracy"]),
        "sampled_extractor_valid": [value.extractor_valid for value in extracted[1:]],
        "sampled_contract_valid": contract_valid[1:],
        "sampled_training_rewards": sampled_training_rewards,
        "sampled_format_rewards": sampled_format_rewards,
        "sampled_native_r1v_shadow_rewards": sampled_native_rewards,
        "sampled_native_r1v_shadow_valid": sampled_native_valid,
        "sampled_canonical_rewards": sampled_canonical_rewards,
        "sampled_reward_disagreement_reasons": reason_names[1:],
        "parser_version": PARSER_VERSION,
        **contract_metadata,
        "sample_count": n,
        "sample_correct_count": c,
        "sample_correct": sampled_correct,
        "p_sample": p_raw,
        "p_i_jeffreys": p_i,
        "q_i": mixed_group_probability(p_i, group_size),
        "pass_at_g": pass_at_k(n, c, group_size),
        "pass_at_k16": pass_at_k(n, c, n),
        "variance_proxy": p_i * (1.0 - p_i),
        "mean_sampled_training_reward": sum(sampled_training_rewards) / n,
        "mean_sampled_format_reward": sum(sampled_format_rewards) / n,
        "canonical_sample_correct_count": canonical_sample_count,
        "canonical_p_sample": canonical_sample_count / n,
        "sampled_canonical_correct": canonical_correct[1:],
    }

#!/usr/bin/env python3
"""Compare two scored runs on the same pairs: paired accuracy deltas with bootstrap
intervals and an exact McNemar test, overall and per template.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.audit.summary import bootstrap_mean_ci
from lwl.evaluation.metrics import aggregate_pair_metrics, pair_score


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in indexed:
            raise ValueError(f"{label} contains duplicate pair_id: {pair_id}")
        indexed[pair_id] = row
    if not indexed:
        raise ValueError(f"{label} contains no rows")
    return indexed


def _paired_exact(left: list[bool], right: list[bool]) -> dict[str, float]:
    if len(left) != len(right):
        raise ValueError("paired exact test requires equal-length inputs")
    b01 = sum((not a) and b for a, b in zip(left, right))
    b10 = sum(a and (not b) for a, b in zip(left, right))
    discordant = b01 + b10
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(b01, b10)
        p_value = min(
            1.0,
            2 * sum(math.comb(discordant, value) for value in range(tail + 1))
            / (2**discordant),
        )
    return {
        "n_common": float(len(left)),
        "b01": float(b01),
        "b10": float(b10),
        "p_value": p_value,
    }


def _paired_summary(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    left_scores = [pair_score(row) for row in left_rows]
    right_scores = [pair_score(row) for row in right_rows]
    left_final = [bool(score["pair_correct"]) for score in left_scores]
    right_final = [bool(score["pair_correct"]) for score in right_scores]
    left_strict = [bool(score["strict_pair_correct"]) for score in left_scores]
    right_strict = [bool(score["strict_pair_correct"]) for score in right_scores]
    final_ci = bootstrap_mean_ci(
        (float(right) - float(left) for left, right in zip(left_final, right_final)),
        seed=seed,
        draws=bootstrap_draws,
    )
    strict_ci = bootstrap_mean_ci(
        (float(right) - float(left) for left, right in zip(left_strict, right_strict)),
        seed=seed + 1,
        draws=bootstrap_draws,
    )
    return {
        "left_pair_accuracy": sum(left_final) / len(left_final),
        "right_pair_accuracy": sum(right_final) / len(right_final),
        "pair_accuracy_delta": final_ci["mean"],
        "pair_accuracy_delta_ci95_low": final_ci["ci_low"],
        "pair_accuracy_delta_ci95_high": final_ci["ci_high"],
        "mcnemar": _paired_exact(left_final, right_final),
        "left_strict_pair_accuracy": sum(left_strict) / len(left_strict),
        "right_strict_pair_accuracy": sum(right_strict) / len(right_strict),
        "strict_pair_accuracy_delta": strict_ci["mean"],
        "strict_pair_accuracy_delta_ci95_low": strict_ci["ci_low"],
        "strict_pair_accuracy_delta_ci95_high": strict_ci["ci_high"],
        "strict_mcnemar": _paired_exact(left_strict, right_strict),
    }


def compare_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_label: str,
    right_label: str,
    *,
    seed: int = 20260712,
    bootstrap_draws: int = 2000,
) -> dict[str, Any]:
    left = _index(left_rows, left_label)
    right = _index(right_rows, right_label)
    if set(left) != set(right):
        raise ValueError(
            f"pair coverage mismatch: {left_label}={len(left)} {right_label}={len(right)} "
            f"left_only={len(set(left) - set(right))} right_only={len(set(right) - set(left))}"
        )
    pair_ids = sorted(left)
    for pair_id in pair_ids:
        if str(left[pair_id].get("template_id")) != str(right[pair_id].get("template_id")):
            raise ValueError(f"template mismatch for pair_id: {pair_id}")

    left_ordered = [left[pair_id] for pair_id in pair_ids]
    right_ordered = [right[pair_id] for pair_id in pair_ids]
    left_metrics = aggregate_pair_metrics(left_ordered)
    right_metrics = aggregate_pair_metrics(right_ordered)
    paired = _paired_summary(
        left_ordered,
        right_ordered,
        seed=seed,
        bootstrap_draws=bootstrap_draws,
    )
    result: dict[str, Any] = {
        "schema_version": "lwl.paired-run-comparison.v2",
        "left_label": left_label,
        "right_label": right_label,
        "n_pairs": len(pair_ids),
        **paired,
        "bootstrap": {
            "unit": "paired item",
            "draws": bootstrap_draws,
            "seed": seed,
            "interval": 0.95,
        },
        "per_template": {},
    }
    if (
        result["left_pair_accuracy"] != left_metrics["pair_accuracy"]
        or result["right_pair_accuracy"] != right_metrics["pair_accuracy"]
        or result["left_strict_pair_accuracy"] != left_metrics["strict_pair_accuracy"]
        or result["right_strict_pair_accuracy"] != right_metrics["strict_pair_accuracy"]
    ):
        raise AssertionError("paired and aggregate scores disagree")
    templates = sorted({str(row.get("template_id")) for row in left_ordered})
    for template in templates:
        template_ids = [
            pair_id for pair_id in pair_ids if str(left[pair_id].get("template_id")) == template
        ]
        left_template = [left[pair_id] for pair_id in template_ids]
        right_template = [right[pair_id] for pair_id in template_ids]
        template_summary = _paired_summary(
            left_template,
            right_template,
            seed=seed + 100 + len(result["per_template"]) * 10,
            bootstrap_draws=bootstrap_draws,
        )
        result["per_template"][template] = {
            "n_pairs": len(template_ids),
            **template_summary,
        }
    return result


def _load(patterns: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    for pattern in patterns:
        # glob.glob, unlike Path.glob, also takes absolute patterns.
        for path in sorted(Path(match) for match in glob.glob(pattern, recursive=True)):
            paths.append(str(path))
            with path.open(encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    return rows, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", nargs="+", required=True)
    parser.add_argument("--right", nargs="+", required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    left_rows, left_paths = _load(args.left)
    right_rows, right_paths = _load(args.right)
    result = compare_rows(
        left_rows,
        right_rows,
        args.left_label,
        args.right_label,
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    result["left_inputs"] = left_paths
    result["right_inputs"] = right_paths

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{args.output}.partial")
    if args.output.exists() or partial.exists():
        raise FileExistsError(f"paired comparison already exists: {args.output}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

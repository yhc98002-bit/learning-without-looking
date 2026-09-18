"""Summaries over the per-item rows of the visual-necessity audit: bootstrap means
per condition and the with-image / image-removed agreement counts.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np


def bootstrap_mean_ci(
    values: Iterable[float],
    seed: int = 20260710,
    draws: int = 2000,
) -> dict[str, float]:
    """Sample mean with a 95% percentile bootstrap interval over items."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        count = min(200, draws - start)
        indices = rng.integers(0, array.size, size=(count, array.size))
        means[start : start + count] = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def summarize_condition(rows: list[dict[str, Any]], seed: int = 20260710) -> dict[str, Any]:
    """Per-condition metric means and the distribution of per-item success rates."""
    if not rows:
        raise ValueError("condition summary requires rows")
    fields = ("p_greedy", "p_sample", "pass_at_g", "pass_at_k16", "variance_proxy")
    distribution_tests = {
        "zero": lambda value: value == 0.0,
        "low_0_0p2": lambda value: 0.0 < value < 0.2,
        "mid_0p2_0p8": lambda value: 0.2 <= value <= 0.8,
        "high_0p8_1": lambda value: 0.8 < value < 1.0,
        "one": lambda value: value == 1.0,
    }
    distribution = {
        name: bootstrap_mean_ci(
            (float(predicate(float(row["p_sample"]))) for row in rows),
            seed=seed + 20 + offset,
        )
        for offset, (name, predicate) in enumerate(distribution_tests.items())
    }
    return {
        "n": len(rows),
        "metrics": {
            field: bootstrap_mean_ci((float(row[field]) for row in rows), seed=seed + offset)
            for offset, field in enumerate(fields)
        },
        "p_sample_midband_0p2_0p8": bootstrap_mean_ci(
            (float(0.2 <= float(row["p_sample"]) <= 0.8) for row in rows), seed=seed + 10
        ),
        "p_sample_distribution": distribution,
    }


def real_blind_quadrants(real_rows: list[dict[str, Any]], blind_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Item counts by whether the with-image and image-removed greedy answers are correct."""
    real = {(row["split"], int(row["row_index"])): bool(row["greedy_correct"]) for row in real_rows}
    blind = {(row["split"], int(row["row_index"])): bool(row["greedy_correct"]) for row in blind_rows}
    if real.keys() != blind.keys():
        raise ValueError("real and blind conditions do not cover identical items")
    counts: Counter[str] = Counter()
    for key in real:
        label = (
            "both_correct"
            if real[key] and blind[key]
            else "real_only"
            if real[key]
            else "blind_only"
            if blind[key]
            else "neither_correct"
        )
        counts[label] += 1
    return {key: counts.get(key, 0) for key in ("both_correct", "real_only", "blind_only", "neither_correct")}

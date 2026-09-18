"""Resampling statistics for the released analyses: paired bootstrap intervals,
rank correlation and the floor/tail summaries used on per-item gains.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata


def deterministic_seed(base_seed: int, label: str) -> int:
    """Seed for one labelled stream, stable across runs and independent of call order."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**32)


def percentile_interval(values: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Two-sided percentile interval at confidence 1 - alpha."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("interval values must be nonempty and finite")
    return [
        float(np.quantile(array, alpha / 2, method="linear")),
        float(np.quantile(array, 1 - alpha / 2, method="linear")),
    ]


def mean_with_paired_bootstrap(
    contributions: Sequence[float], *, draws: int, seed: int
) -> dict[str, Any]:
    """Mean of per-item contributions with a bootstrap interval resampled over items."""
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired contributions must be a nonempty finite vector")
    if draws < 100:
        raise ValueError("bootstrap requires at least 100 draws")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(draws, start + 500)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95": percentile_interval(means),
        "paired_se": float(values.std(ddof=1) / math.sqrt(values.size))
        if values.size > 1
        else 0.0,
        "n": int(values.size),
        "bootstrap_draws": draws,
    }


def paired_difference(
    before: Sequence[bool], after: Sequence[bool], *, draws: int, seed: int
) -> dict[str, Any]:
    """Mean of after - before over paired outcomes, with the paired bootstrap."""
    if len(before) != len(after):
        raise ValueError("paired vectors differ in length")
    contributions = [float(post) - float(pre) for pre, post in zip(before, after)]
    return mean_with_paired_bootstrap(contributions, draws=draws, seed=seed)


def paired_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Ratio of two paired means; bootstrap draws with a vanishing denominator are dropped."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    if num.shape != den.shape or num.ndim != 1 or num.size == 0:
        raise ValueError("ratio contributions must be aligned nonempty vectors")
    den_mean = float(den.mean())
    estimate = float(num.mean() / den_mean) if abs(den_mean) > 1e-12 else None
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    for _ in range(draws):
        indices = rng.integers(0, num.size, size=num.size)
        sampled_denominator = float(den[indices].mean())
        if abs(sampled_denominator) > 1e-12:
            ratios.append(float(num[indices].mean() / sampled_denominator))
    return {
        "estimate": estimate,
        "ci95": percentile_interval(ratios) if ratios else None,
        "bootstrap_draws": draws,
        "retained_bootstrap_draws": len(ratios),
        "denominator_estimate": den_mean,
    }


def tied_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Spearman correlation on average ranks; None when either vector is constant."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("rank vectors must be aligned and contain at least two rows")
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    if np.all(x_rank == x_rank[0]) or np.all(y_rank == y_rank[0]):
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def hurdle_summary(
    correct_counts: Sequence[int],
    gains: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Mean gain above the zero-correct floor minus the floor's own mean gain."""
    counts = np.asarray(correct_counts, dtype=np.int64)
    values = np.asarray(gains, dtype=np.float64)
    if counts.shape != values.shape or counts.ndim != 1 or counts.size == 0:
        raise ValueError("hurdle rows must be aligned and nonempty")
    floor = values[counts == 0]
    above = values[counts > 0]
    if floor.size == 0 or above.size == 0:
        raise ValueError("hurdle contrast requires both floor and above-floor rows")
    rng = np.random.default_rng(seed)
    contrasts = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        floor_sample = floor[rng.integers(0, floor.size, size=floor.size)]
        above_sample = above[rng.integers(0, above.size, size=above.size)]
        contrasts[index] = above_sample.mean() - floor_sample.mean()
    return {
        "estimate": float(above.mean() - floor.mean()),
        "ci95": percentile_interval(contrasts),
        "floor_n": int(floor.size),
        "floor_mean_gain": float(floor.mean()),
        "above_floor_n": int(above.size),
        "above_floor_mean_gain": float(above.mean()),
        "bootstrap_draws": draws,
    }


def floor_and_tail_deciles(
    rows: Sequence[dict[str, Any]],
    *,
    count_key: str = "sample_correct_count",
    q_key: str = "q_i",
    gain_key: str = "gain",
) -> list[dict[str, Any]]:
    """Mean gain for the zero-correct floor and for each decile of the rest, ordered by q."""
    floor = [row for row in rows if int(row[count_key]) == 0]
    above = sorted(
        (row for row in rows if int(row[count_key]) > 0),
        key=lambda row: (float(row[q_key]), int(row["row_index"])),
    )
    if not floor or len(above) < 10:
        raise ValueError("floor/decile table requires a floor and at least ten tail rows")

    def summarize(label: str, group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        q_values = [float(row[q_key]) for row in group]
        gains = [float(row[gain_key]) for row in group]
        return {
            "group": label,
            "n": len(group),
            "q_min": min(q_values),
            "q_max": max(q_values),
            "mean_gain": float(np.mean(gains)),
        }

    result = [summarize("floor_c0", floor)]
    for decile in range(10):
        start = (decile * len(above)) // 10
        stop = ((decile + 1) * len(above)) // 10
        result.append(summarize(f"above_d{decile + 1}", above[start:stop]))
    return result

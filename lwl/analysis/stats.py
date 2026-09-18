"""Statistics over the released records: accuracies, seed means, paired differences with
bootstrap intervals, binomial intervals, exact and permutation tests, failure-set overlap
and chance-corrected retention.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest, norm

from lwl.analysis.resample import (
    deterministic_seed,
    mean_with_paired_bootstrap,
    paired_difference,
    percentile_interval,
)
from lwl.evaluation.metrics import (
    bootstrap_ci,
    mcnemar_exact,
    pair_accuracy_ci,
    permutation_null_pair_accuracy,
)

Row = dict[str, Any]

__all__ = [
    "bootstrap_ci",
    "pair_accuracy_ci",
    "permutation_null_pair_accuracy",
    "mcnemar_exact",
    "mean_with_paired_bootstrap",
    "paired_difference",
    "percentile_interval",
    "deterministic_seed",
    "flags",
    "pair_accuracy",
    "item_accuracy",
    "accuracy",
    "accuracy_with_ci",
    "seed_means",
    "mean_over_seeds",
    "align",
    "difference",
    "difference_with_ci",
    "wilson_interval",
    "clopper_pearson_interval",
    "mcnemar_flags",
    "mcnemar_rows",
    "to_scorable",
    "correct_ids",
    "failed_ids",
    "jaccard_overlap",
    "jaccard_permutation_null",
    "chance_level",
    "chance_corrected_retention",
    "retention_summary",
]


def _flag(row: Row, field: str) -> float:
    """One outcome as 0.0 or 1.0; an unscored row is an error, not a wrong answer."""
    value = row[field]
    if value is None:
        raise ValueError(f"row {row.get('pair_id') or row.get('item_id')!r} has no {field}")
    return float(bool(value))


def flags(rows: Iterable[Row], field: str) -> list[float]:
    """The named boolean field of each row as 0.0 or 1.0."""
    return [_flag(row, field) for row in rows]


def accuracy(values: Sequence[float]) -> float:
    if len(values) == 0:
        raise ValueError("accuracy needs at least one value")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def pair_accuracy(rows: Iterable[Row], field: str = "pair_correct") -> float:
    """Share of pairs with both members correct."""
    return accuracy(flags(rows, field))


def item_accuracy(rows: Iterable[Row], field: str = "correct") -> float:
    """Share of single-image items answered correctly."""
    return accuracy(flags(rows, field))


def accuracy_with_ci(
    rows: Iterable[Row], field: str, *, draws: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Accuracy with a percentile bootstrap interval over items; exact interval when constant."""
    values = flags(rows, field)
    n = len(values)
    successes = int(round(sum(values)))
    if len(set(values)) <= 1:
        return {
            "estimate": accuracy(values),
            "ci95": clopper_pearson_interval(successes, n),
            "ci_method": "clopper_pearson",
            "n": n,
            "successes": successes,
        }
    summary = mean_with_paired_bootstrap(values, draws=draws, seed=seed)
    return {
        "estimate": summary["estimate"],
        "ci95": summary["ci95"],
        "ci_method": "percentile_bootstrap",
        "n": n,
        "successes": successes,
        "bootstrap_draws": draws,
    }


def seed_means(
    rows: Iterable[Row], field: str = "pair_correct", *, seed_field: str = "seed"
) -> dict[Any, float]:
    """Accuracy within each seed, keyed by seed."""
    grouped: dict[Any, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.get(seed_field), []).append(_flag(row, field))
    return {seed: accuracy(values) for seed, values in sorted(grouped.items(), key=lambda item: str(item[0]))}


def mean_over_seeds(
    rows: Iterable[Row], field: str = "pair_correct", *, seed_field: str = "seed"
) -> dict[str, Any]:
    """Mean of the per-seed accuracies, with the spread across seeds."""
    per_seed = seed_means(rows, field, seed_field=seed_field)
    if not per_seed:
        raise ValueError("no rows to average over seeds")
    values = np.asarray(list(per_seed.values()), dtype=np.float64)
    return {
        "estimate": float(values.mean()),
        "per_seed": per_seed,
        "n_seeds": int(values.size),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def align(
    left: Iterable[Row], right: Iterable[Row], field: str, *, id_field: str = "pair_id"
) -> tuple[list[float], list[float], list[Any]]:
    """The named field of both row sets over the ids they share, in a common order."""
    left_by_id = {row[id_field]: _flag(row, field) for row in left}
    right_by_id = {row[id_field]: _flag(row, field) for row in right}
    shared = sorted(set(left_by_id) & set(right_by_id), key=str)
    if not shared:
        raise ValueError("the two row sets share no ids")
    return [left_by_id[i] for i in shared], [right_by_id[i] for i in shared], shared


def difference(
    left: Iterable[Row], right: Iterable[Row], field: str = "pair_correct"
) -> float:
    """Accuracy of `right` minus accuracy of `left`, unpaired."""
    return accuracy(flags(right, field)) - accuracy(flags(left, field))


def difference_with_ci(
    left: Iterable[Row],
    right: Iterable[Row],
    field: str = "pair_correct",
    *,
    id_field: str = "pair_id",
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired difference right - left over shared ids, with the paired bootstrap interval."""
    before, after, shared = align(left, right, field, id_field=id_field)
    summary = paired_difference(before, after, draws=draws, seed=seed)
    summary["n_shared"] = len(shared)
    return summary


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> list[float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = float(norm.isf(alpha / 2))
    denominator = n + z**2
    centre = (successes + z**2 / 2) / denominator
    half = z / denominator * np.sqrt(successes * (n - successes) / n + z**2 / 4)
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def clopper_pearson_interval(successes: int, n: int, alpha: float = 0.05) -> list[float]:
    """Exact (Clopper-Pearson) interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    high = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return [low, high]


def mcnemar_flags(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    """Exact McNemar test on two aligned outcome vectors; matches `mcnemar_exact` on pair rows."""
    if len(left) != len(right):
        raise ValueError("paired vectors differ in length")
    b01 = sum((not bool(a)) and bool(b) for a, b in zip(left, right))
    b10 = sum(bool(a) and (not bool(b)) for a, b in zip(left, right))
    discordant = b01 + b10
    p_value = 1.0 if discordant == 0 else float(binomtest(min(b01, b10), discordant, 0.5).pvalue)
    return {
        "n_common": float(len(left)),
        "b01": float(b01),
        "b10": float(b10),
        "n_discordant": float(discordant),
        "p_value": p_value,
    }


def mcnemar_rows(
    left: Iterable[Row],
    right: Iterable[Row],
    field: str = "pair_correct",
    *,
    id_field: str = "pair_id",
) -> dict[str, float]:
    """Exact McNemar test on the recorded flags of two row sets, over the ids they share."""
    before, after, _ = align(left, right, field, id_field=id_field)
    return mcnemar_flags(before, after)


def to_scorable(rows: Iterable[Row]) -> list[Row]:
    """Released pair records in the shape the pair scorer expects, with the golds as answers."""
    scorable = []
    for row in rows:
        scorable.append(
            {
                **row,
                "answer_a": row["gold_a"],
                "answer_b": row["gold_b"],
                "prediction_a": row.get("prediction_a", ""),
                "prediction_b": row.get("prediction_b", ""),
            }
        )
    return scorable


def correct_ids(
    rows: Iterable[Row], field: str = "pair_correct", *, id_field: str = "pair_id"
) -> set[Any]:
    """Ids of the rows the run got right."""
    return {row[id_field] for row in rows if _flag(row, field)}


def failed_ids(
    rows: Iterable[Row], field: str = "pair_correct", *, id_field: str = "pair_id"
) -> set[Any]:
    """Ids of the rows the run got wrong."""
    return {row[id_field] for row in rows if not _flag(row, field)}


def jaccard_overlap(*sets: Iterable[Any]) -> float:
    """Size of the intersection over the size of the union, for two or more sets."""
    if len(sets) < 2:
        raise ValueError("overlap needs at least two sets")
    members = [set(one) for one in sets]
    union = set().union(*members)
    if not union:
        return 0.0
    intersection = set(members[0]).intersection(*members[1:])
    return len(intersection) / len(union)


def jaccard_permutation_null(
    sets: Sequence[Iterable[Any]],
    universe: Iterable[Any],
    *,
    draws: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Observed overlap against sets of the same sizes drawn at random from `universe`."""
    members = [set(one) for one in sets]
    if len(members) < 2:
        raise ValueError("overlap needs at least two sets")
    pool = sorted(set(universe), key=str)
    sizes = [len(one) for one in members]
    if any(size > len(pool) for size in sizes):
        raise ValueError("a set is larger than the universe it is drawn from")
    observed = jaccard_overlap(*members)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = [
            set(rng.choice(len(pool), size=size, replace=False).tolist()) for size in sizes
        ]
        null[index] = jaccard_overlap(*sampled)
    p_value = float((np.sum(null >= observed) + 1) / (draws + 1))
    return {
        "jaccard": observed,
        "null_mean": float(null.mean()),
        "p_value": p_value,
        "permutations": draws,
        "set_sizes": sizes,
        "universe_size": len(pool),
    }


def chance_level(options_per_item: Iterable[Any]) -> float:
    """Mean of the item-specific 1/k, counting a free-form item (no options) as zero."""
    values = []
    for count in options_per_item:
        if count is None or int(count) <= 0:
            values.append(0.0)
        else:
            values.append(1.0 / int(count))
    if not values:
        raise ValueError("chance level needs at least one item")
    return float(np.mean(values))


def chance_corrected_retention(
    image_removed_accuracy: float, with_image_accuracy: float, chance: float
) -> float:
    """(accuracy - chance) / (with-image accuracy - chance); chance 0 gives the naive ratio."""
    denominator = with_image_accuracy - chance
    if abs(denominator) < 1e-12:
        raise ValueError("with-image accuracy equals the chance level")
    return (image_removed_accuracy - chance) / denominator


def retention_summary(
    with_image: Sequence[float],
    image_removed: Sequence[float],
    chance: float,
    *,
    draws: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Naive and chance-corrected retention with item-paired bootstrap intervals.

    The two vectors are per-item outcomes for the same items in the same order.
    """
    present = np.asarray(with_image, dtype=np.float64)
    removed = np.asarray(image_removed, dtype=np.float64)
    if present.shape != removed.shape or present.ndim != 1 or present.size == 0:
        raise ValueError("retention needs aligned nonempty per-item vectors")
    naive = []
    corrected = []
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        indices = rng.integers(0, present.size, size=present.size)
        present_mean = float(present[indices].mean())
        removed_mean = float(removed[indices].mean())
        if abs(present_mean) > 1e-12:
            naive.append(removed_mean / present_mean)
        if abs(present_mean - chance) > 1e-12:
            corrected.append((removed_mean - chance) / (present_mean - chance))
    present_mean = float(present.mean())
    removed_mean = float(removed.mean())
    return {
        "n": int(present.size),
        "with_image": present_mean,
        "image_removed": removed_mean,
        "chance": float(chance),
        "naive_retention": removed_mean / present_mean if abs(present_mean) > 1e-12 else None,
        "naive_ci95": percentile_interval(naive) if naive else None,
        "corrected_retention": chance_corrected_retention(removed_mean, present_mean, chance),
        "corrected_ci95": percentile_interval(corrected) if corrected else None,
        "bootstrap_draws": draws,
        "retained_bootstrap_draws": len(corrected),
    }

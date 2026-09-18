"""Rebuilds of the long-horizon trajectory table (E.1), the shared-error table (E.2), the two
corrosion panels of Figure 2 and the corrosion figures quoted in Section 3 and Appendix E.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from lwl.analysis.load import iter_records, select
from lwl.evaluation.metrics import bootstrap_ci

Row = dict[str, Any]

LONG_HORIZON = "long_horizon_geometry3k_3b"
ACCESS_MATRIX = "access_matrix_geometry3k_3b"
BASELINES = "instrument_baselines"

GROUNDING_TASK = "coordinate_register_twenty_point_x_v02"
GROUNDING_SET = "grounding-suite"
GROUNDING_PAIRS = 600
BENCHMARK_SET = "geometry3k-test"
BENCHMARK_ITEMS = 601

# Every evaluation below is read under the real test condition and is chosen by run and
# measurement; each must hold exactly one record per pair or item.
# The untrained 3B model on the grounding task is the instrument-release evaluation. The later
# remeasure evaluation gives identical answers in another record order, and the printed
# interval resamples the release evaluation's order.
BASE_GROUNDING = {"run": "base-3b", "measurement": "instrument-release"}
# The untrained model on the benchmark: greedy answers recorded in its 16-sample audit.
BASE_BENCHMARK = {"run": "base-3b", "measurement": "primary"}
# The first run continues an earlier 100-step run (the stem). Its step-100 grounding pairs are
# the stem's long-horizon-stem evaluation, and its step-100 benchmark answers are the greedy
# answers recorded in the stem's audit.
STEM_GROUNDING = {"run": "long-horizon-stem-3b", "measurement": "long-horizon-stem"}
STEM_BENCHMARK = {"run": "long-horizon-stem-3b", "measurement": "primary"}
LONG_HORIZON_RUNS = {1: "long-horizon-run1-3b", 2: "long-horizon-run2-3b"}
GRAY_RUNS = {seed: f"geometry3k-gray-seed{seed}-3b" for seed in (1, 2, 3)}
GRAY_STEP = 100

# Canonical outcome of a greedy answer, in audit and in item records.
AUDIT_FIELD = "greedy_correct"
ITEM_FIELD = "correct"

# Interval settings of the published numbers: item bootstrap for the trajectory levels and
# changes, permutation null for the degraded sets.
LEVEL_DRAWS, LEVEL_SEED = 2000, 20260728
PERMUTATIONS, PERMUTATION_SEED = 10000, 20260828
Z95 = 1.959963984540054

GRAY_SEEDS = (1, 2, 3)
# The order the pairwise nulls were drawn in, which their shared generator makes part of them.
SEED_PAIRS = ((3, 1), (3, 2), (1, 2))

# Agreement margins: half of the last printed digit.
THREE_DECIMALS = 5e-4
FOUR_DECIMALS = 5e-5
ONE_DECIMAL = 0.05
EXACT = 0

TRAJECTORY_STEPS = (0, 100, 150, 200, 300, 400)

# Printed cells of the trajectory table, keyed by (seed, step). Seed 1's grounding intervals
# exist only under the earlier matcher and are not printed.
TRAJECTORY_PRINTED: dict[tuple[int, int], dict[str, Any]] = {
    (1, 0): {"benchmark": (0.175, [0.145, 0.206]), "grounding": (0.455, None)},
    (1, 100): {"benchmark": (0.431, [0.393, 0.471]), "grounding": (0.477, None),
               "grounding_change": (0.022, None)},
    (1, 150): {"benchmark": (0.463, [0.424, 0.504]), "grounding": (0.465, None),
               "grounding_change": (0.010, None)},
    (1, 200): {"benchmark": (0.483, [0.444, 0.524]), "grounding": (0.450, None),
               "grounding_change": (-0.005, None)},
    (1, 300): {"benchmark": (0.464, [0.424, 0.506]), "grounding": (0.440, None),
               "grounding_change": (-0.015, None)},
    (1, 400): {"benchmark": (0.436, [0.398, 0.474]), "grounding": (0.407, None),
               "grounding_change": (-0.048, None)},
    (2, 0): {"benchmark": (0.175, [0.145, 0.206]), "grounding": (0.455, [0.415, 0.497])},
    (2, 100): {"benchmark": (0.428, [0.389, 0.468]), "grounding": (0.465, [0.425, 0.505]),
               "grounding_change": (0.010, [-0.020, 0.042])},
    (2, 150): {"benchmark": (0.446, [0.404, 0.488]), "grounding": (0.468, [0.428, 0.508]),
               "grounding_change": (0.013, [-0.022, 0.048])},
    (2, 200): {"benchmark": (0.496, [0.454, 0.537]), "grounding": (0.447, [0.407, 0.487]),
               "grounding_change": (-0.008, [-0.042, 0.027])},
    (2, 300): {"benchmark": (0.449, [0.409, 0.488]), "grounding": (0.400, [0.362, 0.440]),
               "grounding_change": (-0.055, [-0.093, -0.017])},
    (2, 400): {"benchmark": (0.453, [0.411, 0.493]), "grounding": (0.413, [0.373, 0.453]),
               "grounding_change": (-0.042, [-0.078, -0.002])},
}

# Printed cells of the overlap table.
OVERLAP_PER_SEED_PRINTED: dict[int, dict[str, Any]] = {
    1: {"degraded": 51, "gained": 24, "wrong_slots": 52},
    2: {"degraded": 49, "gained": 22, "wrong_slots": 53},
    3: {"degraded": 45, "gained": 23, "wrong_slots": 46},
}

OVERLAP_PRINTED: dict[str, dict[str, Any]] = {
    "seeds 1 vs 2": {"intersection": 42, "union": 58, "jaccard": 0.724,
                     "permutation_null_mean": 0.097, "permutation_null_p95": 0.149,
                     "identical_wrong_count": 41,
                     "identical_wrong_rate": (0.976, [0.877, 0.996])},
    "seeds 1 vs 3": {"intersection": 43, "union": 53, "jaccard": 0.811,
                     "permutation_null_mean": 0.093, "permutation_null_p95": 0.143,
                     "identical_wrong_count": 44,
                     "identical_wrong_rate": (1.000, [0.920, 1.000])},
    "seeds 2 vs 3": {"intersection": 40, "union": 54, "jaccard": 0.741,
                     "permutation_null_mean": 0.091, "permutation_null_p95": 0.146,
                     "identical_wrong_count": 40,
                     "identical_wrong_rate": (0.976, [0.874, 0.996])},
    "all three": {"intersection": 39, "union": 59, "jaccard": 0.661,
                  "permutation_null_mean": 0.011, "permutation_null_p95": 0.026,
                  "identical_wrong_count": 39,
                  "identical_wrong_rate": (0.975, [0.871, 0.996])},
}

FOOTNOTE_PRINTED: dict[str, Any] = {
    "base_pairs": 600,
    "base_solved_pairs": 283,
    "seed 1 only": 5,
    "seed 2 only": 6,
    "seed 3 only": 1,
    "seeds 1 and 2 only": 3,
    "seeds 1 and 3 only": 4,
    "seeds 2 and 3 only": 1,
    "all three": 39,
    "union": 59,
}

# Printed as p = 10^-4 for every comparison.
CAPTION_PERMUTATION_P = 1e-4

# Appendix I: under the current matcher the untrained 3B model solves 273 of the 600 pairs.
PRINTED_SOLVED_CURRENT_MATCHER = 273

# The three-way null mean was stored to four decimals (0.0115) and printed from that; the
# printing formats the stored binary value, which lies just below 0.0115, so it reads 0.011
# where the exact 0.01155 rounds to 0.012.
NULL_MEAN_NOTE = (
    "printed from the four-decimal value 0.0115, formatted to three decimals from its "
    "binary value, just below 0.0115"
)

# Values plotted in the two corrosion panels, to four decimals.
PANEL_B_PLOTTED: dict[tuple[str, str, int], float] = {
    ("benchmark_accuracy", "base", 0): 0.1747,
    ("benchmark_accuracy", "seed 1", 0): 0.1747,
    ("benchmark_accuracy", "seed 1", 100): 0.4309,
    ("benchmark_accuracy", "seed 1", 150): 0.4626,
    ("benchmark_accuracy", "seed 1", 200): 0.4825,
    ("benchmark_accuracy", "seed 1", 300): 0.4642,
    ("benchmark_accuracy", "seed 1", 400): 0.4359,
    ("benchmark_accuracy", "seed 2", 0): 0.1747,
    ("benchmark_accuracy", "seed 2", 100): 0.4276,
    ("benchmark_accuracy", "seed 2", 150): 0.4459,
    ("benchmark_accuracy", "seed 2", 200): 0.4958,
    ("benchmark_accuracy", "seed 2", 300): 0.4493,
    ("benchmark_accuracy", "seed 2", 400): 0.4526,
    ("grounding_pair_accuracy", "base", 0): 0.4550,
    ("grounding_pair_accuracy", "seed 1", 0): 0.4550,
    ("grounding_pair_accuracy", "seed 1", 100): 0.4767,
    ("grounding_pair_accuracy", "seed 1", 150): 0.4650,
    ("grounding_pair_accuracy", "seed 1", 200): 0.4500,
    ("grounding_pair_accuracy", "seed 1", 300): 0.4400,
    ("grounding_pair_accuracy", "seed 1", 400): 0.4067,
    ("grounding_pair_accuracy", "seed 2", 0): 0.4550,
    ("grounding_pair_accuracy", "seed 2", 100): 0.4650,
    ("grounding_pair_accuracy", "seed 2", 150): 0.4683,
    ("grounding_pair_accuracy", "seed 2", 200): 0.4467,
    ("grounding_pair_accuracy", "seed 2", 300): 0.4000,
    ("grounding_pair_accuracy", "seed 2", 400): 0.4133,
}

PANEL_C_PLOTTED: dict[str, Any] = {
    "seed 1 only": 5,
    "seed 2 only": 6,
    "seed 3 only": 1,
    "seeds 1 and 2 only": 3,
    "seeds 1 and 3 only": 4,
    "seeds 2 and 3 only": 1,
    "all three": 39,
    "seed 1 total": 51,
    "seed 2 total": 49,
    "seed 3 total": 45,
    "three-way Jaccard": 0.661,
    "permutation null": 0.011,
    "identical wrong answers": 39,
    "shared wrong slots": 40,
}

# Corrosion figures quoted in Appendix E; the peak accuracies are the cells Table E.1 marks.
# Peaks and churn use the canonical matcher; grounding uses the current pair scorer.
APPENDIX_PRINTED: dict[str, Any] = {
    "peak": {1: 0.483, 2: 0.496},
    "peak_step": {1: 200, 2: 200},
    "grounding_drop_pp": {1: 7.0, 2: 5.2},
    "below_base_pp": {1: 4.8, 2: 4.2},
    "churn_items": 137,
    "churn_total": 601,
    "net_gain_items": 3,
}
CHURN_SEED, CHURN_FROM, CHURN_TO = 1, 100, 400
# Section 3: grounding is below the untrained model from step 200 onward in both runs.
BELOW_BASE_STEPS = (200, 300, 400)


def _wilson(successes: int, n: int) -> list[float] | None:
    """Wilson score interval for a proportion."""
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _jaccard(*sets: set[str]) -> float:
    union = set().union(*sets)
    if not union:
        return 0.0
    return len(set(sets[0]).intersection(*sets[1:])) / len(union)


def _describe(selection: dict[str, Any], kind: str) -> str:
    step = f" step {selection['step']}" if "step" in selection else ""
    return f"{selection['run']} {kind} [{selection['measurement']}]{step}"


@lru_cache(maxsize=None)
def _records(family: str, kind: str) -> tuple[Row, ...]:
    return tuple(iter_records(family, kind))


def _cell(family: str, kind: str, key: str, size: int, **selection: Any) -> dict[str, Row]:
    """One evaluation, keyed by pair or item id in the order it was recorded."""
    cell: dict[str, Row] = {}
    for row in select(_records(family, kind), test_condition="real", **selection):
        if row[key] in cell:
            raise ValueError(f"{family}/{kind} {selection}: {row[key]} recorded twice")
        cell[row[key]] = row
    if len(cell) != size:
        raise ValueError(f"{family}/{kind} {selection}: {len(cell)} records, expected {size}")
    return cell


def _grounding_selection(seed: int, step: int) -> tuple[str, dict[str, Any]]:
    if step == 0:
        return BASELINES, BASE_GROUNDING
    if seed == 1 and step == 100:
        return LONG_HORIZON, STEM_GROUNDING
    return LONG_HORIZON, {"run": LONG_HORIZON_RUNS[seed], "measurement": "primary", "step": step}


@lru_cache(maxsize=None)
def _grounding(seed: int, step: int) -> tuple[dict[str, Row], str]:
    """Grounding pairs of one checkpoint (step 0 is the untrained model) and their source."""
    family, selection = _grounding_selection(seed, step)
    cell = _cell(family, "pairs", "pair_id", GROUNDING_PAIRS,
                 eval_set=GROUNDING_SET, task=GROUNDING_TASK, **selection)
    return cell, _describe(selection, "pairs")


def _base_grounding() -> dict[str, Row]:
    return _grounding(1, 0)[0]


@lru_cache(maxsize=None)
def _gray_grounding(seed: int) -> dict[str, Row]:
    """Grounding pairs of one gray-trained checkpoint at the end of its schedule."""
    return _cell(ACCESS_MATRIX, "pairs", "pair_id", GROUNDING_PAIRS, eval_set=GROUNDING_SET,
                 task=GROUNDING_TASK, run=GRAY_RUNS[seed], measurement="primary", step=GRAY_STEP)


@lru_cache(maxsize=None)
def _benchmark(seed: int, step: int) -> dict[str, Any]:
    """Geometry3K test outcomes of one checkpoint under the canonical matcher."""
    if step == 0:
        family, kind, selection, field = ACCESS_MATRIX, "audits", BASE_BENCHMARK, AUDIT_FIELD
    elif seed == 1 and step == 100:
        family, kind, selection, field = LONG_HORIZON, "audits", STEM_BENCHMARK, AUDIT_FIELD
    else:
        family, kind, field = LONG_HORIZON, "items", ITEM_FIELD
        selection = {"run": LONG_HORIZON_RUNS[seed], "measurement": "primary", "step": step}
    cell = _cell(family, kind, "item_id", BENCHMARK_ITEMS, eval_set=BENCHMARK_SET, **selection)
    return {
        "canonical": {item: bool(row[field]) for item, row in cell.items()},
        "source": _describe(selection, kind),
    }


def _values(outcomes: dict[str, bool]) -> list[float]:
    return [1.0 if correct else 0.0 for correct in outcomes.values()]


def _pair_outcomes(cell: dict[str, Row], field: str = "pair_correct") -> dict[str, bool]:
    return {pair_id: bool(row[field]) for pair_id, row in cell.items()}


def _level(outcomes: dict[str, bool]) -> tuple[float, list[float]]:
    """Accuracy and its item bootstrap, resampled in the order the evaluation recorded."""
    values = _values(outcomes)
    low, high = bootstrap_ci(values, n_boot=LEVEL_DRAWS, seed=LEVEL_SEED)
    return sum(values) / len(values), [low, high]


def _accuracy(outcomes: dict[str, bool]) -> float:
    return sum(outcomes.values()) / len(outcomes)


COLUMNS = (
    "scope", "step", "quantity", "n", "value", "ci_low", "ci_high", "value_current_matcher",
    "paper_value", "paper_ci_low", "paper_ci_high", "tolerance", "relation", "paper_rounding",
    "status", "source", "note",
)

# Columns a table keeps only when at least one of its cells fills them.
OPTIONAL_COLUMNS = frozenset(COLUMNS) - {"scope", "quantity", "n", "value", "paper_value",
                                         "tolerance", "status"}


def _row(
    scope: str,
    quantity: str,
    n: int,
    value: Any,
    printed: Any,
    *,
    step: int | None = None,
    interval: list[float] | None = None,
    printed_interval: list[float] | None = None,
    tolerance: float | None = THREE_DECIMALS,
    relation: str | None = None,
    paper_rounding: float | None = None,
    current: Any = None,
    source: str | None = None,
    note: str | None = None,
) -> Row:
    """One printed cell: the rebuilt number beside the printed one."""
    return {
        "scope": scope, "step": step, "quantity": quantity, "n": n, "value": value,
        "ci_low": interval[0] if interval else None,
        "ci_high": interval[1] if interval else None,
        "value_current_matcher": current,
        "paper_value": printed,
        "paper_ci_low": printed_interval[0] if printed_interval else None,
        "paper_ci_high": printed_interval[1] if printed_interval else None,
        "tolerance": tolerance, "relation": relation, "paper_rounding": paper_rounding,
        "status": "rebuilt", "source": source, "note": note,
    }


def _four_then_three(value: float) -> float:
    """A value stored to four decimals, then formatted to three from that stored binary value."""
    return float(f"{float(f'{value:.4f}'):.3f}")


def _finish(rows: list[Row]) -> list[Row]:
    """Drop the optional columns this table leaves empty."""
    columns = [
        key
        for key in COLUMNS
        if key not in OPTIONAL_COLUMNS or any(row.get(key) is not None for row in rows)
    ]
    return [{key: row.get(key) for key in columns} for row in rows]


def _change_vs_base(cell: dict[str, Row]) -> list[float]:
    """Per-pair changes from the untrained model, in pair-id order."""
    base = _base_grounding()
    keys = sorted(base)
    if set(cell) != set(keys):
        raise ValueError("a checkpoint and the untrained model cover different pairs")
    return [
        (1.0 if cell[key]["pair_correct"] else 0.0) - (1.0 if base[key]["pair_correct"] else 0.0)
        for key in keys
    ]


def trajectories() -> list[Row]:
    """Table E.1: benchmark and grounding of both long-horizon runs at steps 0 to 400."""
    rows: list[Row] = []
    for seed in (1, 2):
        for step in TRAJECTORY_STEPS:
            printed = TRAJECTORY_PRINTED[(seed, step)]
            scope = f"seed {seed}"

            benchmark = _benchmark(seed, step)
            value, interval = _level(benchmark["canonical"])
            paper, paper_interval = printed["benchmark"]
            rows.append(_row(
                scope, "benchmark_accuracy", BENCHMARK_ITEMS, value, paper, step=step,
                interval=interval, printed_interval=paper_interval, source=benchmark["source"],
            ))

            cell, source = _grounding(seed, step)
            paper, paper_interval = printed["grounding"]
            value, interval = _level(_pair_outcomes(cell))
            rows.append(_row(
                scope, "grounding_pair_accuracy", GROUNDING_PAIRS, value, paper, step=step,
                interval=interval if paper_interval else None, printed_interval=paper_interval,
                source=source,
            ))

            if "grounding_change" not in printed:
                continue
            differences = _change_vs_base(cell)
            paper, paper_interval = printed["grounding_change"]
            interval = None
            if paper_interval:
                interval = list(bootstrap_ci(differences, n_boot=LEVEL_DRAWS, seed=LEVEL_SEED))
            rows.append(_row(
                scope, "grounding_change_vs_base", GROUNDING_PAIRS,
                sum(differences) / len(differences), paper, step=step, interval=interval,
                printed_interval=paper_interval, source=source,
            ))
    return _finish(rows)


def _degradation(as_logged: bool) -> dict[str, Any]:
    """Degraded sets, wrong slots and their overlap under one scoring of the pairs.

    The overlap table was scored with the earlier matcher revision that defined the sets,
    so the published numbers come from the `_as_logged` fields; the same analysis under the
    repository's own scorer is reported beside them.
    """
    suffix = "_as_logged" if as_logged else ""
    pair_field = f"pair_correct{suffix}"
    member_field = {side: f"correct_{side}{suffix}" for side in ("a", "b")}

    base = _base_grounding()
    solved = {pair_id for pair_id, row in base.items() if row[pair_field]}
    universe = sorted(solved)

    degraded: dict[int, set[str]] = {}
    wrong: dict[int, dict[str, str | None]] = {}
    per_seed: dict[int, dict[str, Any]] = {}
    ids = sorted(base)
    for seed in GRAY_SEEDS:
        arm = _gray_grounding(seed)
        if set(arm) != set(ids):
            raise ValueError(f"gray seed {seed} and the untrained model cover different pairs")
        degraded[seed] = {p for p in solved if not arm[p][pair_field]}
        slots: dict[str, str | None] = {}
        for pair_id in degraded[seed]:
            for side in ("a", "b"):
                if not arm[pair_id][member_field[side]]:
                    slots[f"{pair_id}|{side}"] = arm[pair_id][f"answer_{side}"]
        wrong[seed] = slots
        per_seed[seed] = {
            "n": len(arm),
            "degraded": len(degraded[seed]),
            "gained": sum(1 for p, row in arm.items() if row[pair_field] and p not in solved),
            "wrong_slots": len(slots),
        }

    # The four nulls share one generator, so the order they are drawn in is part of the
    # published numbers.
    rng_perm = np.random.default_rng(PERMUTATION_SEED)
    overlap: dict[str, Any] = {}
    comparisons = [((left, right), _label(left, right)) for left, right in SEED_PAIRS]
    comparisons.append((GRAY_SEEDS, "all three"))
    for seeds, label in comparisons:
        sets = tuple(degraded[seed] for seed in seeds)
        null = np.empty(PERMUTATIONS, dtype=np.float64)
        for index in range(PERMUTATIONS):
            drawn = [
                set(rng_perm.choice(universe, size=len(one), replace=False).tolist())
                for one in sets
            ]
            null[index] = _jaccard(*drawn)
        overlap[label] = _overlap_entry(sets, _jaccard(*sets), null, wrong, seeds)

    sets = tuple(degraded[seed] for seed in GRAY_SEEDS)
    regions = {}
    for seed in GRAY_SEEDS:
        others = [degraded[other] for other in GRAY_SEEDS if other != seed]
        regions[f"seed {seed} only"] = len(degraded[seed] - others[0] - others[1])
    for left, right in ((1, 2), (1, 3), (2, 3)):
        third = next(seed for seed in GRAY_SEEDS if seed not in (left, right))
        regions[f"seeds {left} and {right} only"] = len(
            (degraded[left] & degraded[right]) - degraded[third]
        )
    regions["all three"] = len(sets[0] & sets[1] & sets[2])
    regions["union"] = len(sets[0] | sets[1] | sets[2])

    return {
        "base_solved_pairs": len(solved),
        "base_n": len(base),
        "per_seed": per_seed,
        "overlap": overlap,
        "regions": regions,
    }


def _label(left: int, right: int) -> str:
    low, high = sorted((left, right))
    return f"seeds {low} vs {high}"


def _overlap_entry(
    sets: tuple[set[str], ...],
    observed: float,
    null: np.ndarray,
    wrong: dict[int, dict[str, str | None]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    shared = set(wrong[seeds[0]])
    for seed in seeds[1:]:
        shared &= set(wrong[seed])
    identical = 0
    for key in shared:
        answers = [wrong[seed][key] for seed in seeds]
        if answers[0] is not None and all(answer == answers[0] for answer in answers):
            identical += 1
    return {
        "intersection": len(set(sets[0]).intersection(*sets[1:])),
        "union": len(set().union(*sets)),
        "jaccard": observed,
        "permutation_null_mean": float(null.mean()),
        "permutation_null_p95": float(np.percentile(null, 95)),
        "permutation_p": float((int(np.sum(null >= observed)) + 1) / (PERMUTATIONS + 1)),
        "shared_wrong_slots": len(shared),
        "identical_wrong_count": identical,
        "identical_wrong_rate": identical / len(shared) if shared else None,
        "identical_wrong_ci": _wilson(identical, len(shared)),
    }


@lru_cache(maxsize=1)
def _overlap_analysis() -> tuple[dict[str, Any], dict[str, Any]]:
    """The degraded-set analysis under the earlier matcher and under the current scorer."""
    return _degradation(True), _degradation(False)


OVERLAP_SOURCE = (f"{BASE_GROUNDING['run']} pairs [{BASE_GROUNDING['measurement']}]; "
                  f"gray seeds 1-3 pairs [primary], step {GRAY_STEP}")


def _seed_source(seed: int) -> str:
    return (f"{BASE_GROUNDING['run']} pairs [{BASE_GROUNDING['measurement']}]; "
            f"{GRAY_RUNS[seed]} pairs [primary], step {GRAY_STEP}")


def degraded_set_overlap() -> list[Row]:
    """Table E.2: per-run degraded sets, their overlap and the answers that replace the correct
    ones, with the count of pairs the untrained model solves that Appendix I quotes."""
    earlier, current = _overlap_analysis()
    rows: list[Row] = []

    for seed in GRAY_SEEDS:
        printed = OVERLAP_PER_SEED_PRINTED[seed]
        got = earlier["per_seed"][seed]
        other = current["per_seed"][seed]
        scope, source = f"seed {seed}", _seed_source(seed)
        for quantity in ("degraded", "gained", "wrong_slots"):
            rows.append(_row(
                scope, quantity, got["n"], got[quantity], printed[quantity], tolerance=EXACT,
                current=other[quantity], source=source,
            ))

    for scope, printed in OVERLAP_PRINTED.items():
        got = earlier["overlap"][scope]
        other = current["overlap"][scope]
        for quantity in ("intersection", "union"):
            rows.append(_row(
                scope, quantity, earlier["base_solved_pairs"], got[quantity], printed[quantity],
                tolerance=EXACT, current=other[quantity], source=OVERLAP_SOURCE,
            ))
        rows.append(_row(
            scope, "jaccard", earlier["base_solved_pairs"], got["jaccard"], printed["jaccard"],
            current=other["jaccard"], source=OVERLAP_SOURCE,
        ))
        rows.append(_null_mean_row(scope, got, other, printed["permutation_null_mean"]))
        rows.append(_row(
            scope, "permutation_null_p95", PERMUTATIONS, got["permutation_null_p95"],
            printed["permutation_null_p95"], current=other["permutation_null_p95"],
            source=OVERLAP_SOURCE,
        ))
        rows.append(_row(
            scope, "identical_wrong_count", got["shared_wrong_slots"],
            got["identical_wrong_count"], printed["identical_wrong_count"], tolerance=EXACT,
            current=other["identical_wrong_count"], source=OVERLAP_SOURCE,
        ))
        paper, paper_interval = printed["identical_wrong_rate"]
        rows.append(_row(
            scope, "identical_wrong_rate", got["shared_wrong_slots"],
            got["identical_wrong_rate"], paper, interval=got["identical_wrong_ci"],
            printed_interval=paper_interval, current=other["identical_wrong_rate"],
            source=OVERLAP_SOURCE,
        ))

    rows.append(_row(
        "caption", "permutation_p", PERMUTATIONS,
        max(entry["permutation_p"] for entry in earlier["overlap"].values()),
        CAPTION_PERMUTATION_P, tolerance=FOUR_DECIMALS,
        current=max(entry["permutation_p"] for entry in current["overlap"].values()),
        source=OVERLAP_SOURCE, note="largest of the four comparisons",
    ))
    rows.append(_row(
        "footnote", "base_pairs", earlier["base_n"], earlier["base_n"],
        FOOTNOTE_PRINTED["base_pairs"], tolerance=EXACT, current=current["base_n"],
        source=OVERLAP_SOURCE,
    ))
    rows.append(_row(
        "footnote", "base_solved_pairs", earlier["base_n"], earlier["base_solved_pairs"],
        FOOTNOTE_PRINTED["base_solved_pairs"], tolerance=EXACT,
        current=current["base_solved_pairs"], source=OVERLAP_SOURCE,
    ))
    for region in (
        "seed 1 only", "seed 2 only", "seed 3 only",
        "seeds 1 and 2 only", "seeds 1 and 3 only", "seeds 2 and 3 only", "all three", "union",
    ):
        rows.append(_row(
            "footnote", region, earlier["base_solved_pairs"], earlier["regions"][region],
            FOOTNOTE_PRINTED[region], tolerance=EXACT, current=current["regions"][region],
            source=OVERLAP_SOURCE,
        ))
    rows.append(_row(
        "Appendix I", "base_solved_pairs_current_matcher", current["base_n"],
        current["base_solved_pairs"], PRINTED_SOLVED_CURRENT_MATCHER, tolerance=EXACT,
        source=OVERLAP_SOURCE, note="the same count under the current matcher revision",
    ))
    return _finish(rows)


def _null_mean_row(scope: str, got: dict[str, Any], other: dict[str, Any], printed: float) -> Row:
    """The permutation-null mean; the three-way mean was printed from its four-decimal value."""
    three_way = scope == "all three"
    return _row(
        scope, "permutation_null_mean", PERMUTATIONS, got["permutation_null_mean"], printed,
        paper_rounding=_four_then_three(got["permutation_null_mean"]) if three_way else None,
        current=other["permutation_null_mean"], source=OVERLAP_SOURCE,
        note=NULL_MEAN_NOTE if three_way else None,
    )


def panel_trajectories() -> list[Row]:
    """Panel (b): the plotted benchmark and grounding trajectories of both runs."""
    rows: list[Row] = []
    for (metric, series, step), plotted in PANEL_B_PLOTTED.items():
        seed = 1 if series == "base" else int(series.split()[-1])
        if metric == "grounding_pair_accuracy":
            cell, source = _grounding(seed, step)
            value, n = _accuracy(_pair_outcomes(cell)), len(cell)
        else:
            benchmark = _benchmark(seed, step)
            value, n = _accuracy(benchmark["canonical"]), len(benchmark["canonical"])
            source = benchmark["source"]
        rows.append(_row(
            series, metric, n, value, plotted, step=step, tolerance=FOUR_DECIMALS, source=source,
        ))
    return _finish(rows)


def panel_overlap() -> list[Row]:
    """Panel (c): the degraded-set regions, their overlap and the answer agreement."""
    earlier, current = _overlap_analysis()
    rows: list[Row] = []
    for region in (
        "seed 1 only", "seed 2 only", "seed 3 only",
        "seeds 1 and 2 only", "seeds 1 and 3 only", "seeds 2 and 3 only", "all three",
    ):
        rows.append(_row(
            region, "pairs", earlier["base_solved_pairs"], earlier["regions"][region],
            PANEL_C_PLOTTED[region], tolerance=EXACT, current=current["regions"][region],
            source=OVERLAP_SOURCE,
        ))
    for seed in GRAY_SEEDS:
        rows.append(_row(
            f"seed {seed}", "degraded", earlier["base_solved_pairs"],
            earlier["per_seed"][seed]["degraded"], PANEL_C_PLOTTED[f"seed {seed} total"],
            tolerance=EXACT, current=current["per_seed"][seed]["degraded"],
            source=_seed_source(seed),
        ))
    triple, other = earlier["overlap"]["all three"], current["overlap"]["all three"]
    rows.append(_row(
        "all three", "three_way_jaccard", earlier["base_solved_pairs"], triple["jaccard"],
        PANEL_C_PLOTTED["three-way Jaccard"], current=other["jaccard"], source=OVERLAP_SOURCE,
    ))
    rows.append(_null_mean_row("all three", triple, other, PANEL_C_PLOTTED["permutation null"]))
    rows.append(_row(
        "all three", "identical_wrong_answers", triple["shared_wrong_slots"],
        triple["identical_wrong_count"], PANEL_C_PLOTTED["identical wrong answers"],
        tolerance=EXACT, current=other["identical_wrong_count"], source=OVERLAP_SOURCE,
    ))
    rows.append(_row(
        "all three", "shared_wrong_slots", triple["shared_wrong_slots"],
        triple["shared_wrong_slots"], PANEL_C_PLOTTED["shared wrong slots"], tolerance=EXACT,
        current=other["shared_wrong_slots"], source=OVERLAP_SOURCE,
    ))
    return _finish(rows)


def appendix_text() -> list[Row]:
    """The corrosion figures quoted in Section 3 and Appendix E: benchmark peaks, grounding
    below the untrained model from step 200, the grounding decline in percentage points and
    the benchmark churn of the first run."""
    rows: list[Row] = []
    base = _accuracy(_pair_outcomes(_base_grounding()))
    for seed in (1, 2):
        scope = f"seed {seed}"
        levels = {step: _benchmark(seed, step) for step in TRAJECTORY_STEPS}
        peak = max(TRAJECTORY_STEPS, key=lambda step: _accuracy(levels[step]["canonical"]))
        rows.append(_row(
            scope, "benchmark_peak", BENCHMARK_ITEMS, _accuracy(levels[peak]["canonical"]),
            APPENDIX_PRINTED["peak"][seed], step=peak, source=levels[peak]["source"],
            note="the cell Table E.1 marks as the peak",
        ))
        rows.append(_row(
            scope, "benchmark_peak_step", BENCHMARK_ITEMS, peak,
            APPENDIX_PRINTED["peak_step"][seed], tolerance=EXACT, source=levels[peak]["source"],
        ))

        for step in BELOW_BASE_STEPS:
            cell, source = _grounding(seed, step)
            rows.append(_row(
                scope, "grounding_change_vs_base", GROUNDING_PAIRS,
                _accuracy(_pair_outcomes(cell)) - base, 0.0, step=step, relation="lt",
                source=f"{_grounding(seed, 0)[1]}; {source}",
                note="Section 3: below the untrained model from step 200 onward",
            ))

        start, start_source = _grounding(seed, 100)
        end, end_source = _grounding(seed, 400)
        start_level, end_level = _accuracy(_pair_outcomes(start)), _accuracy(_pair_outcomes(end))
        rows.append(_row(
            scope, "grounding_drop_step100_to_400_pp", GROUNDING_PAIRS,
            100 * (start_level - end_level), APPENDIX_PRINTED["grounding_drop_pp"][seed],
            tolerance=ONE_DECIMAL, source=f"{start_source}; {end_source}",
        ))
        rows.append(_row(
            scope, "grounding_below_base_step400_pp", GROUNDING_PAIRS, 100 * (base - end_level),
            APPENDIX_PRINTED["below_base_pp"][seed], tolerance=ONE_DECIMAL,
            source=f"{_grounding(seed, 0)[1]}; {end_source}",
        ))

    before = _benchmark(CHURN_SEED, CHURN_FROM)
    after = _benchmark(CHURN_SEED, CHURN_TO)
    if set(before["canonical"]) != set(after["canonical"]):
        raise ValueError("the two benchmark evaluations cover different items")
    changed = sum(
        1 for item, correct in before["canonical"].items() if correct != after["canonical"][item]
    )
    net = sum(after["canonical"].values()) - sum(before["canonical"].values())
    scope, source = f"seed {CHURN_SEED}", f"{before['source']}; {after['source']}"
    rows.append(_row(
        scope, "benchmark_items_changed_step100_to_400", BENCHMARK_ITEMS, changed,
        APPENDIX_PRINTED["churn_items"], tolerance=EXACT, source=source,
    ))
    rows.append(_row(
        scope, "benchmark_items", BENCHMARK_ITEMS, len(before["canonical"]),
        APPENDIX_PRINTED["churn_total"], tolerance=EXACT, source=source,
    ))
    rows.append(_row(
        scope, "benchmark_net_gain_items_step100_to_400", BENCHMARK_ITEMS, net,
        APPENDIX_PRINTED["net_gain_items"], tolerance=EXACT, source=source,
    ))
    return _finish(rows)


TARGETS = {
    "tableE1": trajectories,
    "tableE2": degraded_set_overlap,
    "figure2b": panel_trajectories,
    "figure2c": panel_overlap,
    "appendixE": appendix_text,
}

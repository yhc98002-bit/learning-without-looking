"""Rebuilds of the access matrix (Table 1), its per-seed table (D.2) and the two
visual-necessity audits (Tables A.1 and A.2) from the released per-item outputs.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from lwl.analysis.load import load_audits, load_items, load_pairs
from lwl.analysis.resample import (
    mean_with_paired_bootstrap,
    paired_ratio,
    percentile_interval,
)
from lwl.analysis.paper import INPUT_MISSING
from lwl.paths import data_path

Row = dict[str, Any]

ACCESS_FAMILY = "access_matrix_geometry3k_3b"
PAIR_FAMILY = "access_pair_geometry3k_7b"
BENCHMARK_FAMILY = "necessity_audit_benchmarks"
CROSS_FAMILY = "necessity_audit_cross_family"

# Every table here reports the primary evaluation of each cell; repeat evaluations of the same
# checkpoint are not used.
PRIMARY = "primary"
TEST_SPLIT = "geometry3k-test"
ARM_STEP = 100
UNTRAINED_3B = "base-3b"
PAIR_RUNS = ("base-7b", "geometry3k-real-7b", "geometry3k-gray-7b")

TEST_ITEMS = 601
SEEDS = (1, 2, 3)
ARMS = ("real", "caption", "none", "gray")
CONDITIONS = ("real", "gray", "none", "caption")
IMAGE_REMOVED = ("gray", "none")

GAIN_DRAWS = 5000
SHARE_DRAWS = 20000
GAIN_SEED = 20260716
SHARE_SEED = 20260717

# The 7B interval in the table footnote came from a labelled random stream; this is that
# stream's seed, so the interval below reproduces the printed one exactly.
PAIR_DRAWS = 5000
PAIR_SHARE_SEED = 3883718640

RETENTION_DRAWS = 10000
RETENTION_CHUNK = 1000
# BLINK, MathVerse, MMVP, MMMU and HallusionBench were resampled with a fresh stream per row.
RETENTION_SEED = 20260729
# MMStar, MathVista and the cross-family sample were resampled together from one stream, two
# draws per subset (the second scored a stricter matcher the tables do not use), in the order
# `_shared_stream_plan` lists. Replaying that order reproduces the printed intervals exactly.
SHARED_STREAM_SEED = 20260728
SHARED_STREAM_BENCHMARKS = ("mmstar", "mathvista")
# mean(with image) - mean(null) can land on ~1e-16 rather than exactly 0; dividing by that
# gives a meaningless ratio, so the per-row streams treat anything that small as undefined.
# The shared stream treated only an exact zero as undefined.
RETENTION_TOLERANCE = 1e-12

# Five free-form MathVerse items whose gold differs between the two evaluations; the row
# excludes them.
MATHVERSE_UNPAIRED = frozenset(f"mathverse_{index}" for index in range(2956, 2961))

CROSS_MODELS = ("gemma3-12b", "internvl3-9b")
CROSS_SPLIT = "virl39k-audit"
CROSS_SAMPLE = ("audit", "virl39k_rows.jsonl")
FREE_FORM_TYPES = ("numeric", "text_or_expression")

TWO_DECIMALS = 5e-3

TABLE1_BASE = {"real": 0.175, "gray": 0.090, "none": 0.068}
TABLE1_GAINS = {
    "real": {"real": 0.239, "gray": 0.019, "none": 0.036},
    "caption": {"real": 0.171, "gray": 0.024, "none": 0.037},
    "none": {"real": 0.137, "gray": 0.015, "none": 0.044},
    "gray": {"real": 0.125, "gray": 0.014, "none": 0.032},
}
TABLE1_SHARES = {"real": 1.0, "caption": 0.72, "none": 0.57, "gray": 0.52}
TABLE1_FOOTNOTE_SHARE = (0.78, 0.64, 0.92)
# The caption: with the test image removed, no condition gains more than this.
TABLE1_REMOVED_GAIN_BOUND = 0.05

# Per seed: the four accuracies, the four gains, then the recovery share.
TABLE_D2 = {
    ("real", 1): (0.423, 0.106, 0.108, 0.311, 0.248, 0.017, 0.040, 0.101, None),
    ("real", 2): (0.419, 0.112, 0.098, 0.301, 0.245, 0.022, 0.030, 0.091, None),
    ("real", 3): (0.398, 0.110, 0.106, 0.326, 0.223, 0.020, 0.038, 0.116, None),
    ("caption", 1): (0.361, 0.118, 0.112, 0.318, 0.186, 0.028, 0.043, 0.108, 0.752),
    ("caption", 2): (0.346, 0.108, 0.106, 0.291, 0.171, 0.018, 0.038, 0.082, 0.701),
    ("caption", 3): (0.331, 0.117, 0.097, 0.326, 0.156, 0.027, 0.028, 0.116, 0.701),
    ("none", 1): (0.325, 0.095, 0.097, 0.286, 0.150, 0.005, 0.028, 0.076, 0.604),
    ("none", 2): (0.308, 0.117, 0.121, 0.294, 0.133, 0.027, 0.053, 0.085, 0.544),
    ("none", 3): (0.303, 0.103, 0.118, 0.261, 0.128, 0.013, 0.050, 0.051, 0.574),
    ("gray", 1): (0.311, 0.110, 0.108, 0.286, 0.136, 0.020, 0.040, 0.076, 0.550),
    ("gray", 2): (0.306, 0.097, 0.088, 0.270, 0.132, 0.007, 0.020, 0.060, 0.538),
    ("gray", 3): (0.281, 0.106, 0.103, 0.245, 0.106, 0.017, 0.035, 0.035, 0.478),
}
TABLE_D2_BASE = (0.175, 0.090, 0.068, 0.210)

_BINARY = "formatted to three decimals from its binary value, just below {v}"

# Printed cells the manuscript derived from accuracies stored to four decimals: gains and
# shares were formed from those and printed to three decimals. Printing formats the stored
# binary value, so a four-decimal value ending in 5 can round down. These cells differ from the
# exact value rounded once; each keeps the exact value and carries what that pipeline gives,
# which reproduces the printed cell. Key: (table, arm, seed, evaluation, quantity).
PAPER_ROUNDING_CELLS: dict[tuple[str, str, int | None, str, str], str] = {
    ("1", "real", None, "real", "gain over base"):
        "mean of the per-seed gains from four-decimal accuracies (0.2479, 0.2446, 0.2230)",
    ("D2", "real", 2, "gray", "accuracy"): "67/601 rounded to 0.1115, then to three decimals",
    ("D2", "real", 2, "caption", "gain over base"):
        "four-decimal accuracies, 0.3012 - 0.2097 = 0.0915, " + _BINARY.format(v="0.0915"),
    ("D2", "caption", 1, "none", "accuracy"): "67/601 rounded to 0.1115, then to three decimals",
    ("D2", "caption", 3, "gray", "accuracy"): "70/601 rounded to 0.1165, then to three decimals",
    ("D2", "none", 1, "real", "accuracy"): "195/601 rounded to 0.3245, then to three decimals",
    ("D2", "none", 1, "caption", "gain over base"):
        "four-decimal accuracies, 0.2862 - 0.2097 = 0.0765, " + _BINARY.format(v="0.0765"),
    ("D2", "none", 2, "gray", "accuracy"): "70/601 rounded to 0.1165, then to three decimals",
    ("D2", "none", 2, "caption", "accuracy"):
        "177/601 rounded to 0.2945, " + _BINARY.format(v="0.2945"),
    ("D2", "none", 3, "caption", "gain over base"):
        "four-decimal accuracies, 0.2612 - 0.2097 = 0.0515, " + _BINARY.format(v="0.0515"),
    ("D2", "none", 3, "real", "share of real's gain"): "four-decimal gains, 0.1281 / 0.2230",
    ("D2", "gray", 1, "caption", "gain over base"):
        "four-decimal accuracies, 0.2862 - 0.2097 = 0.0765, " + _BINARY.format(v="0.0765"),
    ("D2", "gray", 2, "real", "gain over base"): "four-decimal accuracies, 0.3062 - 0.1747",
    ("D2", "gray", 2, "real", "share of real's gain"): "four-decimal gains, 0.1315 / 0.2446",
}


def _four(value: float) -> float:
    return float(f"{value:.4f}")


def _three(value: float) -> float:
    return float(f"{value:.3f}")


# label -> benchmark, answer format, chance-level override, then the printed n, with-image,
# blind, chance level, naive retention and corrected retention.
AUDIT_ROWS = (
    (
        "BLINK (MC, pooled)",
        "blink",
        "multiple_choice",
        None,
        (1901, 0.493, 0.409, 0.377, (0.829, 0.781, 0.879), (0.271, 0.089, 0.454)),
    ),
    (
        "MathVerse (MC, pooled)",
        "mathverse",
        "multiple_choice",
        None,
        (2180, 0.465, 0.394, 0.260, (0.848, 0.807, 0.889), (0.655, 0.568, 0.743)),
    ),
    (
        "MathVerse (free-form)",
        "mathverse",
        "free_form",
        0.0,
        (1755, 0.055, 0.019, 0.000, (0.340, 0.240, 0.458), (0.340, 0.240, 0.458)),
    ),
    (
        "MMVP (MC, k=2)",
        "mmvp",
        "multiple_choice",
        None,
        (300, 0.660, 0.500, 0.500, (0.758, 0.668, 0.854), (0.000, -0.417, 0.342)),
    ),
    (
        "MMMU dev+val (MC, pooled)",
        "mmmu",
        "multiple_choice",
        None,
        (988, 0.506, 0.413, 0.263, (0.816, 0.764, 0.873), (0.617, 0.512, 0.729)),
    ),
    (
        "MMMU dev+val (free-form)",
        "mmmu",
        "free_form",
        0.0,
        (62, 0.097, 0.048, 0.000, (0.500, 0.000, 1.000), (0.500, 0.000, 1.000)),
    ),
    (
        "HallusionBench (free-form, chance 0)",
        "hallusion",
        "free_form",
        0.0,
        (1129, 0.598, 0.475, 0.000, (0.794, 0.748, 0.843), (0.794, 0.748, 0.843)),
    ),
    (
        "HallusionBench (Yes/No, chance 0.5)",
        "hallusion",
        "free_form",
        0.5,
        (1129, 0.598, 0.475, 0.500, (0.794, 0.748, 0.843), (-0.258, -0.673, 0.039)),
    ),
    (
        "MMStar (MC, pooled)",
        "mmstar",
        "multiple_choice",
        None,
        (1500, 0.554, 0.261, 0.269, (0.471, 0.430, 0.513), (-0.029, -0.108, 0.049)),
    ),
    (
        "MathVista testmini (MC, pooled)",
        "mathvista",
        "multiple_choice",
        None,
        (539, 0.725, 0.512, 0.332, (0.706, 0.646, 0.765), (0.458, 0.351, 0.564)),
    ),
    (
        "MathVista testmini (free-form)",
        "mathvista",
        "free_form",
        None,
        (460, 0.504, 0.115, 0.000, (0.228, 0.174, 0.287), (0.228, 0.174, 0.287)),
    ),
)

# model, pool, answer format, then the printed n, with-image, blind, chance level, naive
# retention and corrected retention. The image is removed rather than replaced. Gemma-3 scores
# below chance on the multiple-choice pool, where the table prints the ratio as undefined.
UNDEFINED = "undefined"

CROSS_ROWS = (
    (
        "gemma3-12b",
        "ViRL39K free-form",
        "free_form",
        (2789, 0.430, 0.312, 0.000, (0.727, 0.690, 0.765), (0.727, 0.690, 0.765)),
    ),
    (
        "gemma3-12b",
        "ViRL39K multiple choice",
        "multiple_choice",
        (1215, 0.135, 0.086, 0.268, (0.634, 0.533, 0.744), UNDEFINED),
    ),
    (
        "internvl3-9b",
        "ViRL39K free-form",
        "free_form",
        (2789, 0.269, 0.130, 0.000, (0.485, 0.439, 0.533), (0.485, 0.439, 0.533)),
    ),
    (
        "internvl3-9b",
        "ViRL39K multiple choice",
        "multiple_choice",
        (1215, 0.294, 0.205, 0.268, (0.697, 0.619, 0.782), (-2.439, -17.961, 6.505)),
    ),
)
CROSS_SAMPLE_SIZE = 4096
# The caption and note of Table A.2: InternVL3-9B's with-image margin over chance on the
# multiple-choice pool, the items the two pools hold, and the multiple-choice items left out
# because their number of options is unknown.
PRINTED_CHANCE_MARGIN = ("internvl3-9b", 0.026)
PRINTED_POOLED_ITEMS = 4004
PRINTED_UNKNOWN_OPTIONS = 92

CROSS_PAIR_CLAIMS = (
    ("gemma3-12b", "grounding-suite"),
    ("gemma3-12b", "grounding-twin"),
    ("internvl3-9b", "grounding-suite"),
    ("internvl3-9b", "grounding-twin"),
)

ADDED_INTERVAL = "interval added here; the table prints none"

NOT_RELEASED = "no released records for this row"

UNREPLAYED = (
    "not rebuilt: replaying the shared resampling stream needs the MMStar, MathVista and "
    "cross-family records, and for the cross-family rows data/audit/virl39k_rows.jsonl"
)

_CACHE: dict[str, Any] = {}


def _item_order(item_id: str) -> Any:
    tail = item_id.rsplit("-", 1)[-1]
    return (0, int(tail)) if tail.isdigit() else (1, item_id)


def _outcomes(rows: Sequence[Row], field: str) -> np.ndarray:
    return np.array([1.0 if row[field] else 0.0 for row in rows], dtype=np.float64)


def _test_cell(rows: Iterable[Row], field: str, cell: Any) -> np.ndarray | None:
    """Outcomes of one evaluation on the test split in item order; None unless complete."""
    by_item: dict[str, Row] = {}
    for row in rows:
        if row["item_id"] in by_item:
            raise ValueError(f"item {row['item_id']} appears twice in {cell}")
        by_item[row["item_id"]] = row
    ordered = [by_item[item] for item in sorted(by_item, key=_item_order)]
    if [_item_order(row["item_id"]) for row in ordered] != [
        (0, index) for index in range(TEST_ITEMS)
    ]:
        return None
    if any(row[field] is None for row in ordered):
        return None
    return _outcomes(ordered, field)


def _trained_cells() -> dict[tuple[str, int, str], np.ndarray]:
    """Per-item outcomes of each (arm, seed, evaluation) cell: the primary evaluation of the
    step-100 checkpoint on the test split, canonical matcher."""
    grouped: dict[tuple[str, int, str], list[Row]] = {}
    rows = load_items(
        ACCESS_FAMILY,
        measurement=PRIMARY,
        eval_set=TEST_SPLIT,
        step=ARM_STEP,
        train_condition=ARMS,
        seed=SEEDS,
    )
    for row in rows:
        key = (row["train_condition"], row["seed"], row["test_condition"])
        grouped.setdefault(key, []).append(row)
    cells = {}
    for key, cell_rows in grouped.items():
        outcomes = _test_cell(cell_rows, "correct", key)
        if outcomes is not None:
            cells[key] = outcomes
    return cells


def _untrained_cells() -> dict[str, np.ndarray]:
    """Per-item outcomes of the untrained 3B model on the test split, by evaluation.

    These are the greedy answers recorded inside its 16-sample audit, scored with the same
    canonical matcher as the trained runs.
    """
    grouped: dict[str, list[Row]] = {}
    rows = load_audits(ACCESS_FAMILY, run=UNTRAINED_3B, measurement=PRIMARY, eval_set=TEST_SPLIT)
    for row in rows:
        grouped.setdefault(row["test_condition"], []).append(row)
    cells = {}
    for condition, cell_rows in grouped.items():
        outcomes = _test_cell(cell_rows, "greedy_correct", condition)
        if outcomes is not None:
            cells[condition] = outcomes
    return cells


def _gain(
    trained: dict[tuple[str, int, str], np.ndarray],
    untrained: dict[str, np.ndarray],
    arm: str,
    condition: str,
    seed: int | None = None,
) -> np.ndarray | None:
    """Per-item gain over the untrained model, averaged over seeds when no seed is given."""
    base = untrained.get(condition)
    if base is None:
        return None
    seeds = SEEDS if seed is None else (seed,)
    cells = [trained.get((arm, one, condition)) for one in seeds]
    if any(cell is None for cell in cells):
        return None
    return np.mean(cells, axis=0) - base


def _bound(interval: Sequence[float] | None, index: int) -> float | None:
    return None if interval is None else float(interval[index])


def _access_row(
    table: str,
    trained_with: str,
    seed: Any,
    evaluation: str,
    quantity: str,
    paper_value: Any,
    *,
    value: Any = None,
    interval: Sequence[float] | None = None,
    paper_interval: Sequence[float] | None = None,
    added_interval: Sequence[float] | None = None,
    tolerance: float | None = None,
    interval_tolerance: float | None = None,
    relation: str | None = None,
    paper_rounding: float | None = None,
    note: str = "",
) -> Row:
    rounding = PAPER_ROUNDING_CELLS.get((table, trained_with, seed, evaluation, quantity))
    if rounding is None or value is None:
        paper_rounding = rounding = None
    else:
        rounding = f"printed from {rounding}"
    return {
        "table": table,
        "trained_with": trained_with,
        "seed": seed,
        "evaluation": evaluation,
        "quantity": quantity,
        "items": TEST_ITEMS,
        "value": value,
        "ci_low": _bound(interval, 0),
        "ci_high": _bound(interval, 1),
        "paper_value": paper_value,
        "paper_ci_low": _bound(paper_interval, 0),
        "paper_ci_high": _bound(paper_interval, 1),
        "ci_low_rebuilt": _bound(added_interval, 0),
        "ci_high_rebuilt": _bound(added_interval, 1),
        "tolerance": tolerance,
        "interval_tolerance": interval_tolerance,
        "relation": relation,
        "paper_rounding": paper_rounding,
        "status": "rebuilt" if value is not None else "not in the released records",
        "note": "; ".join(part for part in (note, rounding) if part),
    }


def _manuscript_gain(
    trained: dict[tuple[str, int, str], np.ndarray],
    untrained: dict[str, np.ndarray],
    arm: str,
    condition: str,
    seeds: Sequence[int],
) -> float | None:
    """Mean gain over `seeds` formed, as the manuscript did, from four-decimal accuracies."""
    base = untrained.get(condition)
    cells = [trained.get((arm, seed, condition)) for seed in seeds]
    if base is None or any(cell is None for cell in cells):
        return None
    return float(np.mean([_four(float(cell.mean())) - _four(float(base.mean())) for cell in cells]))


def _missing_note(arm_present: bool, base_present: bool) -> str:
    missing = []
    if not arm_present:
        missing.append("this arm under this evaluation")
    if not base_present:
        missing.append("the untrained model under this evaluation")
    return "no released records for " + " or ".join(missing)


def _base_row(untrained: dict[str, np.ndarray], table: str, condition: str, printed: float) -> Row:
    cell = untrained.get(condition)
    return _access_row(
        table,
        "base",
        None,
        condition,
        "accuracy",
        printed,
        value=None if cell is None else float(cell.mean()),
        note="" if cell is not None else _missing_note(True, False),
    )


def _pooled_gain_row(
    trained: dict[tuple[str, int, str], np.ndarray],
    untrained: dict[str, np.ndarray],
    arm: str,
    condition: str,
) -> Row:
    contributions = _gain(trained, untrained, arm, condition)
    if contributions is None:
        return _access_row(
            "1",
            arm,
            None,
            condition,
            "gain over base",
            TABLE1_GAINS[arm][condition],
            note=_missing_note(
                all((arm, seed, condition) in trained for seed in SEEDS),
                condition in untrained,
            ),
        )
    summary = mean_with_paired_bootstrap(contributions, draws=GAIN_DRAWS, seed=GAIN_SEED)
    return _access_row(
        "1",
        arm,
        None,
        condition,
        "gain over base",
        TABLE1_GAINS[arm][condition],
        value=summary["estimate"],
        added_interval=summary["ci95"],
        paper_rounding=_three(_manuscript_gain(trained, untrained, arm, condition, SEEDS)),
        note=ADDED_INTERVAL,
    )


def _pooled_share_row(
    trained: dict[tuple[str, int, str], np.ndarray],
    untrained: dict[str, np.ndarray],
    arm: str,
    reference: np.ndarray | None,
) -> Row:
    printed = TABLE1_SHARES[arm]
    if arm == "real":
        return _access_row(
            "1",
            arm,
            None,
            "real",
            "share of real's gain",
            printed,
            value=1.0,
            tolerance=TWO_DECIMALS,
            note="reference arm",
        )
    contributions = _gain(trained, untrained, arm, "real")
    if contributions is None or reference is None:
        return _access_row(
            "1",
            arm,
            None,
            "real",
            "share of real's gain",
            printed,
            tolerance=TWO_DECIMALS,
            note=_missing_note(
                all((arm, seed, "real") in trained for seed in SEEDS), "real" in untrained
            ),
        )
    summary = paired_ratio(contributions, reference, draws=SHARE_DRAWS, seed=SHARE_SEED)
    return _access_row(
        "1",
        arm,
        None,
        "real",
        "share of real's gain",
        printed,
        value=summary["estimate"],
        added_interval=summary["ci95"],
        tolerance=TWO_DECIMALS,
        note=ADDED_INTERVAL,
    )


def _removed_gain_bound_row(gain_rows: Sequence[Row]) -> Row:
    """The caption's bound: with the test image removed, no condition gains more than 0.05."""
    removed = [row for row in gain_rows if row["evaluation"] in IMAGE_REMOVED]
    value = None
    note = "needs every gain with the test image removed"
    if removed and all(row["value"] is not None for row in removed):
        largest = max(removed, key=lambda row: row["value"])
        value = largest["value"]
        note = f"largest such gain: {largest['trained_with']}, {largest['evaluation']} evaluation"
    return _access_row(
        "1",
        "any",
        None,
        "gray or none",
        "largest gain with the test image removed",
        TABLE1_REMOVED_GAIN_BOUND,
        value=value,
        relation="le",
        note=note,
    )


def table1() -> list[Row]:
    """The access matrix: three-seed gains over the untrained 3B model, shares, 7B footnote."""
    trained = _trained_cells()
    untrained = _untrained_cells()
    rows = [
        _base_row(untrained, "1", condition, printed)
        for condition, printed in TABLE1_BASE.items()
    ]
    reference = _gain(trained, untrained, "real", "real")
    gains: list[Row] = []
    for arm in ARMS:
        arm_gains = [
            _pooled_gain_row(trained, untrained, arm, condition)
            for condition in ("real", "gray", "none")
        ]
        gains.extend(arm_gains)
        rows.append(arm_gains[0])
        rows.append(_pooled_share_row(trained, untrained, arm, reference))
        rows.extend(arm_gains[1:])
    rows.append(_removed_gain_bound_row(gains))
    rows.append(_footnote_row())
    return rows


def _footnote_row() -> Row:
    """The 7B recovery share quoted under the table, with its published interval."""
    printed, low, high = TABLE1_FOOTNOTE_SHARE
    cells = _pair_scale_cells()
    share = None
    interval = None
    if cells is not None:
        base, real_arm, gray_arm = cells
        summary = paired_ratio(
            gray_arm - base,
            real_arm - base,
            draws=PAIR_DRAWS,
            seed=PAIR_SHARE_SEED,
        )
        share = summary["estimate"]
        interval = summary["ci95"]
    return _access_row(
        "1",
        "gray at 7B (one run)",
        None,
        "real",
        "share of real's gain",
        printed,
        value=share,
        interval=interval,
        paper_interval=(low, high),
        tolerance=TWO_DECIMALS,
        interval_tolerance=TWO_DECIMALS,
        note="footnote value",
    )


def _pair_scale_cells() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Untrained, real-trained and gray-trained 7B outcomes on the test split, real image.

    All three are the greedy answers recorded inside each model's 16-sample audit.
    """
    vectors = []
    for run in PAIR_RUNS:
        rows = load_audits(
            PAIR_FAMILY,
            run=run,
            measurement=PRIMARY,
            eval_set=TEST_SPLIT,
            test_condition="real",
        )
        outcomes = _test_cell(rows, "greedy_correct", run)
        if outcomes is None:
            return None
        vectors.append(outcomes)
    return vectors[0], vectors[1], vectors[2]


def tableD2() -> list[Row]:
    """The access matrix per seed: accuracies, gains and recovery shares of each run."""
    trained = _trained_cells()
    untrained = _untrained_cells()
    rows: list[Row] = []

    for index, condition in enumerate(CONDITIONS):
        rows.append(_base_row(untrained, "D2", condition, TABLE_D2_BASE[index]))

    for arm in ARMS:
        for seed in SEEDS:
            printed = TABLE_D2[(arm, seed)]
            for index, condition in enumerate(CONDITIONS):
                cell = trained.get((arm, seed, condition))
                accuracy = None if cell is None else float(cell.mean())
                rows.append(
                    _access_row(
                        "D2",
                        arm,
                        seed,
                        condition,
                        "accuracy",
                        printed[index],
                        value=accuracy,
                        paper_rounding=None if accuracy is None else _three(_four(accuracy)),
                        note="" if cell is not None else _missing_note(False, True),
                    )
                )
            for index, condition in enumerate(CONDITIONS):
                contributions = _gain(trained, untrained, arm, condition, seed)
                manuscript = _manuscript_gain(trained, untrained, arm, condition, (seed,))
                rows.append(
                    _access_row(
                        "D2",
                        arm,
                        seed,
                        condition,
                        "gain over base",
                        printed[4 + index],
                        value=None
                        if contributions is None
                        else float(contributions.mean()),
                        paper_rounding=None if manuscript is None else _three(_four(manuscript)),
                        note=""
                        if contributions is not None
                        else _missing_note(
                            (arm, seed, condition) in trained, condition in untrained
                        ),
                    )
                )
            if printed[8] is None:
                continue
            contributions = _gain(trained, untrained, arm, "real", seed)
            reference = _gain(trained, untrained, "real", "real", seed)
            share = manuscript_share = None
            if contributions is not None and reference is not None:
                share = float(contributions.mean() / reference.mean())
                manuscript_share = _three(_four(
                    _manuscript_gain(trained, untrained, arm, "real", (seed,))
                    / _manuscript_gain(trained, untrained, "real", "real", (seed,))
                ))
            rows.append(
                _access_row(
                    "D2",
                    arm,
                    seed,
                    "real",
                    "share of real's gain",
                    printed[8],
                    value=share,
                    paper_rounding=manuscript_share,
                    note=""
                    if share is not None
                    else _missing_note(
                        (arm, seed, "real") in trained, "real" in untrained
                    ),
                )
            )
    return rows


def _gold_shown(row: Row) -> bool:
    labels = row.get("option_labels") or []
    gold = row.get("gold_labels") or []
    return bool(labels) and bool(gold) and all(label in labels for label in gold)


def _item_null(row: Row) -> float:
    """1/k over the option labels shown; 0 for free-form items and for a gold label not shown."""
    return 1.0 / len(row["option_labels"]) if _gold_shown(row) else 0.0


def _audit_pairs(
    rows: Iterable[Row], group: Callable[[Row], Any]
) -> dict[Any, list[tuple[Row, Row]]]:
    """(with image, image removed) records per group, in the order of the with-image records.

    The released files keep each benchmark's own item order, which the bootstrap replicates
    index into. A group whose two evaluations cover different items is left out.
    """
    cells: dict[str, dict[Any, dict[str, Row]]] = {"real": {}, "none": {}}
    for row in rows:
        if row["test_condition"] not in cells:
            continue
        cell = cells[row["test_condition"]].setdefault(group(row), {})
        if row["item_id"] in cell:
            raise ValueError(f"item {row['item_id']} appears twice in {group(row)}")
        cell[row["item_id"]] = row
    paired = {}
    for key, with_image in cells["real"].items():
        removed = cells["none"].get(key, {})
        if set(removed) == set(with_image):
            paired[key] = [(with_image[item], removed[item]) for item in with_image]
    return paired


def _vectors(
    pairs: Sequence[tuple[Row, Row]], null: Sequence[float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """With-image outcomes, image-removed outcomes and per-item nulls, canonical matcher."""
    if null is None:
        null = [_item_null(removed) for _, removed in pairs]
    return (
        _outcomes([image for image, _ in pairs], "correct"),
        _outcomes([removed for _, removed in pairs], "correct"),
        np.asarray(null, dtype=np.float64),
    )


def _retention(
    with_image: np.ndarray,
    blind: np.ndarray,
    null: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    tolerance: float = RETENTION_TOLERANCE,
) -> dict[str, Any]:
    """Naive and chance-corrected retention with the item-paired percentile bootstrap.

    The chance level is a per-item vector and is re-averaged inside every replicate, which is
    why this does not go through `stats.retention_summary` (that one holds it fixed). The
    corrected ratio is defined only when with-image accuracy exceeds the chance level. Without
    `rng`, the row draws from its own stream.
    """
    rng = np.random.default_rng(RETENTION_SEED) if rng is None else rng
    naive: list[np.ndarray] = []
    corrected: list[np.ndarray] = []
    done = 0
    while done < RETENTION_DRAWS:
        size = min(RETENTION_CHUNK, RETENTION_DRAWS - done)
        index = rng.integers(0, with_image.size, size=(size, with_image.size))
        blind_mean = blind[index].mean(axis=1)
        image_mean = with_image[index].mean(axis=1)
        null_mean = null[index].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            naive.append(np.where(image_mean > 0, blind_mean / image_mean, np.nan))
            denominator = image_mean - null_mean
            corrected.append(
                np.where(
                    np.abs(denominator) > tolerance,
                    (blind_mean - null_mean) / denominator,
                    np.nan,
                )
            )
        done += size

    def interval(values: list[np.ndarray]) -> list[float] | None:
        drawn = np.concatenate(values)
        finite = drawn[~np.isnan(drawn)]
        return percentile_interval(finite) if finite.size else None

    image_point = float(with_image.mean())
    blind_point = float(blind.mean())
    null_point = float(null.mean())
    denominator = image_point - null_point
    defined = denominator > tolerance
    return {
        "n": int(with_image.size),
        "with_image": image_point,
        "blind": blind_point,
        "null": null_point,
        "naive": blind_point / image_point if image_point > 0 else None,
        "naive_ci": interval(naive),
        "corrected": (blind_point - null_point) / denominator if defined else None,
        "corrected_ci": interval(corrected) if defined else None,
    }


def _advance(rng: np.random.Generator, size: int) -> None:
    """Draw and discard one subset's replicates; chunking leaves the stream unchanged."""
    done = 0
    while done < RETENTION_DRAWS:
        chunk = min(RETENTION_CHUNK, RETENTION_DRAWS - done)
        rng.integers(0, size, size=(chunk, size))
        done += chunk


def _answer_types() -> dict[str, str]:
    """Answer type of each cross-family sample item, from the released audit rows."""
    path = data_path(*CROSS_SAMPLE)
    if not path.exists():
        return {}
    types = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                types[row["qid"]] = row["metadata"]["answer_type"]
    return types


def _shared_stream_plan() -> tuple[list[tuple[Any, int | None]], dict[Any, tuple]]:
    """Subset sizes in the order the shared stream drew them, and the printed subsets' vectors.

    Keys name the subsets the tables print; the rest only advance the stream. A size of None
    means the release cannot say how large that subset was, and the replay stops there.
    """
    plan: list[tuple[Any, int | None]] = []
    vectors: dict[Any, tuple] = {}

    def add(key: Any, pairs: Sequence[tuple[Row, Row]], null: Sequence[float] | None = None):
        plan.append((key, len(pairs)))
        if key is not None:
            vectors[key] = _vectors(pairs, null)

    benchmarks = _audit_pairs(
        load_items(
            BENCHMARK_FAMILY,
            model="3b",
            measurement=PRIMARY,
            benchmark=SHARED_STREAM_BENCHMARKS,
        ),
        lambda row: row["benchmark"],
    )
    mmstar = benchmarks.get("mmstar")
    mathvista = benchmarks.get("mathvista")
    if not mmstar or not mathvista:
        return [(None, None)], vectors

    # 3B, then 7B on the same items; the 7B records are not released, so its draws are
    # replayed from the 3B subset sizes.
    for printed in (True, False):
        shown = [pair for pair in mmstar if _gold_shown(pair[1])]
        for count in sorted({len(pair[1]["option_labels"]) for pair in shown}):
            add(None, [pair for pair in shown if len(pair[1]["option_labels"]) == count])
        hidden = [pair for pair in mmstar if not _gold_shown(pair[1])]
        if hidden:
            add(None, hidden)
        add(("mmstar", "multiple_choice") if printed else None, mmstar)
        add(None, mmstar)

    for printed in (True, False):
        choice = [pair for pair in mathvista if pair[0]["answer_format"] == "multiple_choice"]
        free = [pair for pair in mathvista if pair[0]["answer_format"] == "free_form"]
        for count in sorted({len(pair[1]["option_labels"]) for pair in choice}):
            add(None, [pair for pair in choice if len(pair[1]["option_labels"]) == count])
        add(("mathvista", "multiple_choice") if printed else None, choice)
        add(("mathvista", "free_form") if printed else None, free)
        add(None, mathvista)

    cross = _audit_pairs(
        load_items(CROSS_FAMILY, measurement=PRIMARY, eval_set=CROSS_SPLIT),
        lambda row: row["model"],
    )
    types = _answer_types()
    for model in CROSS_MODELS:
        pairs = sorted(cross.get(model, []), key=lambda pair: pair[0]["item_id"])
        if not pairs:
            plan.append((None, None))
            break
        # Image removed, then the caption substitution; subsets are defined per item, so the
        # caption pass has the same sizes.
        for printed in (True, False):
            choice = [
                pair
                for pair in pairs
                if pair[0]["answer_format"] == "multiple_choice" and pair[0]["n_choices"]
            ]
            for count in sorted({pair[0]["n_choices"] for pair in choice}):
                add(None, [pair for pair in choice if pair[0]["n_choices"] == count])
            add(
                (model, "multiple_choice") if printed else None,
                choice,
                [1.0 / pair[0]["n_choices"] for pair in choice],
            )
            for kind in FREE_FORM_TYPES:
                subset = [pair for pair in pairs if types.get(pair[0]["item_id"]) == kind]
                plan.append((None, len(subset) if types else None))
            free = [pair for pair in pairs if pair[0]["answer_format"] == "free_form"]
            add((model, "free_form") if printed else None, free, [0.0] * len(free))
            add(None, pairs)
    return plan, vectors


def _shared_stream() -> dict[Any, dict[str, Any]]:
    """Retention summaries of the printed subsets, drawn from the replayed shared stream."""
    if "shared" not in _CACHE:
        plan, vectors = _shared_stream_plan()
        rng = np.random.default_rng(SHARED_STREAM_SEED)
        summaries = {}
        for key, size in plan:
            if size is None:
                break
            if key in vectors:
                summaries[key] = _retention(*vectors[key], rng=rng, tolerance=0.0)
            else:
                _advance(rng, size)
            _advance(rng, size)
        _CACHE["shared"] = summaries
    return _CACHE["shared"]


def _audit_row(
    table: str,
    label: str,
    quantity: str,
    paper_value: float | None,
    *,
    value: float | None = None,
    n: int | None = None,
    interval: Sequence[float] | None = None,
    paper_interval: Sequence[float] | None = None,
    relation: str | None = None,
    status: str | None = None,
    note: str = "",
) -> Row:
    if status is None:
        status = "rebuilt" if value is not None else "not in the released records"
    return {
        "table": table,
        "row": label,
        "quantity": quantity,
        "items": n,
        "value": value,
        "ci_low": _bound(interval, 0),
        "ci_high": _bound(interval, 1),
        "paper_value": paper_value,
        "paper_ci_low": _bound(paper_interval, 0),
        "paper_ci_high": _bound(paper_interval, 1),
        "tolerance": None,
        "relation": relation,
        "status": status,
        "note": note,
    }


def _audit_rows(
    table: str,
    label: str,
    printed: tuple[Any, ...],
    summary: dict[str, Any] | None,
    note: str,
    status: str | None = None,
) -> list[Row]:
    """One row per cell of an audit table row; a cell the table omits carries no printed value.
    A ratio the table prints as undefined is matched only by a rebuilt row that leaves it
    undefined."""
    count, with_image, blind, null, naive, corrected = printed
    fields = (
        ("n", "n", None, count),
        ("accuracy with the image", "with_image", None, with_image),
        ("accuracy with the image removed", "blind", None, blind),
        ("chance level", "null", None, null),
        ("naive retention", "naive", "naive_ci", naive),
        ("corrected retention", "corrected", "corrected_ci", corrected),
    )
    rows = []
    for quantity, key, interval_key, cell in fields:
        if cell is None and summary is None:
            continue
        paper_value, paper_interval, relation = cell, None, None
        if cell == UNDEFINED:
            relation = "undefined" if summary is not None else None
        elif isinstance(cell, tuple):
            paper_value, paper_interval = cell[0], cell[1:]
        value = None
        interval = None
        if summary is not None:
            value = summary[key]
            interval = None if interval_key is None else summary[interval_key]
        rows.append(
            _audit_row(
                table,
                label,
                quantity,
                paper_value,
                value=value,
                n=None if summary is None else summary["n"],
                interval=interval,
                paper_interval=paper_interval,
                relation=relation,
                status=status if summary is None else "rebuilt",
                note="with-image accuracy is below chance" if relation == "undefined" else note,
            )
        )
    return rows


def tableA1() -> list[Row]:
    """The visual-necessity audit of the untrained 3B model on the seven benchmarks."""
    cells = _audit_pairs(
        load_items(BENCHMARK_FAMILY, model="3b", measurement=PRIMARY),
        lambda row: (row["benchmark"], row["answer_format"]),
    )
    rows: list[Row] = []
    for label, benchmark, answer_format, null_override, printed in AUDIT_ROWS:
        pairs = cells.get((benchmark, answer_format), [])
        if benchmark == "mathverse" and answer_format == "free_form":
            pairs = [pair for pair in pairs if pair[0]["item_id"] not in MATHVERSE_UNPAIRED]
        summary = None
        note = ""
        if not pairs:
            note = NOT_RELEASED
        elif benchmark in SHARED_STREAM_BENCHMARKS:
            summary = _shared_stream().get((benchmark, answer_format))
            note = "" if summary is not None else UNREPLAYED
        else:
            null = None if null_override is None else [null_override] * len(pairs)
            summary = _retention(*_vectors(pairs, null))
        rows.extend(_audit_rows("A1", label, printed, summary, note))
    return rows


def tableA2() -> list[Row]:
    """The same audit for two other model families, plus the claims in its caption and note."""
    shared = _shared_stream()
    # Without the released audit rows the replay stops before the cross-family subsets.
    missing = None if data_path(*CROSS_SAMPLE).exists() else INPUT_MISSING
    rows: list[Row] = []
    for model, pool, answer_format, printed in CROSS_ROWS:
        summary = shared.get((model, answer_format))
        rows.extend(
            _audit_rows(
                "A2",
                f"{model} / {pool}",
                printed,
                summary,
                "" if summary is not None else UNREPLAYED,
                missing,
            )
        )

    model, margin = PRINTED_CHANCE_MARGIN
    summary = shared.get((model, "multiple_choice"))
    rows.append(
        _audit_row(
            "A2",
            f"caption: {model} / ViRL39K multiple choice",
            "with-image accuracy above chance",
            margin,
            value=None if summary is None else summary["with_image"] - summary["null"],
            n=None if summary is None else summary["n"],
            status=missing if summary is None else None,
            note="" if summary is not None else UNREPLAYED,
        )
    )

    items: dict[str, Row] = {}
    for row in load_items(CROSS_FAMILY, measurement=PRIMARY, eval_set=CROSS_SPLIT):
        items.setdefault(row["item_id"], row)
    rows.append(
        _audit_row(
            "A2",
            "caption: audited sample",
            "n",
            CROSS_SAMPLE_SIZE,
            value=len(items) or None,
            n=len(items) or None,
        )
    )
    unknown = sum(
        1 for row in items.values()
        if row["answer_format"] == "multiple_choice" and not row["n_choices"]
    )
    for label, printed, value in (
        ("note: items in the two pools", PRINTED_POOLED_ITEMS, len(items) - unknown),
        ("note: multiple-choice items with an unknown number of options", PRINTED_UNKNOWN_OPTIONS,
         unknown),
    ):
        rows.append(_audit_row("A2", label, "n", printed, value=value if items else None,
                               n=len(items) or None))

    pairs = load_pairs(CROSS_FAMILY, measurement=PRIMARY, test_condition="none")
    for model, eval_set in CROSS_PAIR_CLAIMS:
        selected = [
            row for row in pairs if row["model"] == model and row["eval_set"] == eval_set
        ]
        value = None
        if selected and all(row["pair_correct"] is not None for row in selected):
            value = float(np.mean([bool(row["pair_correct"]) for row in selected]))
        rows.append(
            _audit_row(
                "A2",
                f"caption: {model} / {eval_set}",
                "pair accuracy with the image removed",
                0.0,
                value=value,
                n=len(selected) or None,
                relation="zero",
            )
        )
    return rows


TARGETS: dict[str, Callable[[], list[Row]]] = {
    "table1": table1,
    "tableD2": tableD2,
    "tableA1": tableA1,
    "tableA2": tableA2,
}

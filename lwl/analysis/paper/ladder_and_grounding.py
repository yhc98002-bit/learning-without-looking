"""Rebuilds of the cue-ladder and grounding-instrument tables (B.2, B.3, C.1), the ladder
values quoted in Section 2.3 and the gains panel of Figure 2, from the released per-item outputs.
Every outcome is scored with the canonical matcher (correct_a, correct_b, pair_correct).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from lwl.analysis.load import load_pairs

Row = dict[str, Any]

BASELINES = "instrument_baselines"
LADDER_VIRL39K_3B = "access_matrix_virl39k_3b"
LADDER_GEOMETRY3K_7B = "access_pair_geometry3k_7b"
SUITE_GEOMETRY3K_3B = "access_matrix_geometry3k_3b"

# Fields that identify one evaluation of one checkpoint.
EVALUATION_FIELDS = (
    "run", "measurement", "model", "train_condition", "seed", "step", "test_condition", "eval_set"
)

# The evaluations each table reports. The untrained 3B model on the suite is read from the
# instrument's release evaluation, which is also the baseline of the gains; its later
# remeasure returns the same answers. The untrained levels on the coordinate scenes are the
# real-image scene baselines.
UNTRAINED_3B_SUITE = {
    "run": "base-3b", "measurement": "instrument-release",
    "test_condition": "real", "eval_set": "grounding-suite",
}
SCENE_BASELINE = {
    "measurement": "scene-baseline", "test_condition": "real", "eval_set": "scenes-development"
}
LADDER_ARM = {
    "measurement": "primary", "step": 100, "test_condition": "real",
    "eval_set": "scenes-development",
}

# Discovery and its identification probe are read on the two roles that keep the pair's
# question fixed; the target-switch role is a separate measurement.
COMPOSITION_ROLES = ("target_stable", "invariance")
DENSITIES = ("n8", "n12", "n20")
LEVELS = (("discovery", "l3"), ("identification_probe", "probe"))

# Tasks of the 1,200-pair counterfactual suite, keyed by the name the paper prints.
SUITE_TASKS = {
    "grounding": "coordinate_register_twenty_point_x_v02",
    "header-cued table": "header_cued_table_code_v02",
    "cued-readout": "starred_series_value_nine_v07",
}

GAIN_CONDITIONS = ("real", "gray", "none", "caption")
GAIN_SEEDS = (1, 2, 3)
GAIN_ARM = {
    "measurement": "primary", "step": 100, "test_condition": "real",
    "eval_set": "grounding-suite",
}

# Bootstrap settings of the published intervals. The gain intervals take all 10,000 draws
# from one stream that runs over the whole condition-by-task grid; the level intervals
# restart their stream for each cell and draw in blocks of 500. Both the block size and the
# order of the records change the stream, so the grid is always rebuilt in full and each
# evaluation keeps the record order of the released file.
GAIN_DRAWS = 10000
GAIN_SEED = 20260727
LEVEL_DRAWS = 5000
LEVEL_SEED = 20260716
LEVEL_BLOCK = 500

# Half of the last printed digit, with a guard for two-seed means that land exactly on the
# rounding boundary of the third decimal.
PRINTED_TOLERANCE = 5e-4 + 1e-9
# Figure 2(a) is compared with its plotted values, which carry four decimals.
PLOTTED_TOLERANCE = 5e-5 + 1e-9

_PAIRS: dict[str, list[Row]] = {}


def _family(name: str) -> list[Row]:
    """Pair records of one family, read once and kept in the order of the released file."""
    if name not in _PAIRS:
        _PAIRS[name] = load_pairs(name)
    return _PAIRS[name]


def _evaluation(family: str, **fields: Any) -> list[Row]:
    """Records of exactly one evaluation, selected by its fields, in released-file order."""
    rows = [
        row
        for row in _family(family)
        if all(row.get(field) == value for field, value in fields.items())
    ]
    if not rows:
        raise ValueError(f"{family}: no records for {fields}")
    cells = {tuple(row.get(field) for field in EVALUATION_FIELDS) for row in rows}
    if len(cells) != 1:
        raise ValueError(f"{family}: {fields} matches {len(cells)} evaluations")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError(f"{family}: {fields} repeats a pair")
    return rows


def _task(rows: list[Row], task_id: str) -> list[Row]:
    return [row for row in rows if row["task"] == task_id]


def _flag(row: Row, field: str) -> float:
    """One outcome as 0.0 or 1.0; an unscored row is an error, not a wrong answer."""
    value = row[field]
    if value is None:
        raise ValueError(f"pair {row['pair_id']!r} has no {field}")
    return float(bool(value))


def _flags(rows: list[Row], *fields: str) -> list[float]:
    """The named outcome fields of each row as 0.0 or 1.0."""
    return [_flag(row, field) for row in rows for field in fields]


def _member_accuracy(rows: list[Row]) -> float:
    """Share of individual pair members answered correctly."""
    return float(np.mean(_flags(rows, "correct_a", "correct_b")))


def _bootstrap_means(
    values: np.ndarray, rng: np.random.Generator, *, draws: int, block: int
) -> np.ndarray:
    """Means of `draws` bootstrap resamples of a value vector, drawn in blocks."""
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, block):
        stop = min(draws, start + block)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return means


def _interval(means: np.ndarray) -> tuple[float, float]:
    """The 2.5 and 97.5 percentiles of a set of bootstrap means."""
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _ladder_cell(rows: list[Row], layer: str, density: str) -> list[Row]:
    return [
        row
        for row in rows
        if row["layer"] == layer
        and row["density"] == density
        and row["role"] in COMPOSITION_ROLES
    ]


def _ladder_levels(rows: list[Row]) -> dict[tuple[str, str], tuple[float, int]]:
    """Member accuracy and member count for every level and density of one model."""
    levels: dict[tuple[str, str], tuple[float, int]] = {}
    for level, layer in LEVELS:
        for density in DENSITIES:
            cell = _ladder_cell(rows, layer, density)
            if not cell:
                raise ValueError(f"no {layer} pairs at {density}")
            levels[(level, density)] = (_member_accuracy(cell), 2 * len(cell))
    return levels


# Printed cells of Table B.2: (discovery at 8, 12, 20 points), (probe at 8, 12, 20 points).
# Two-seed means are printed from the exact mean of the two per-seed counts.
B2_PRINTED_3B = {
    ("base", None): ((0.330, 0.260, 0.245), (0.705, 0.680, 0.630)),
    ("real", "mean"): ((0.383, 0.338, 0.330), (0.728, 0.700, 0.610)),
    ("real", 1): ((0.380, 0.330, 0.330), (0.725, 0.710, 0.620)),
    ("real", 2): ((0.385, 0.345, 0.330), (0.730, 0.690, 0.600)),
    ("caption", "mean"): ((0.370, 0.323, 0.340), (0.728, 0.715, 0.635)),
    ("caption", 1): ((0.365, 0.325, 0.345), (0.715, 0.710, 0.610)),
    ("caption", 2): ((0.375, 0.320, 0.335), (0.740, 0.720, 0.660)),
    ("none", "mean"): ((0.368, 0.303, 0.318), (0.730, 0.700, 0.638)),
    ("none", 1): ((0.365, 0.300, 0.305), (0.740, 0.705, 0.650)),
    ("none", 2): ((0.370, 0.305, 0.330), (0.720, 0.695, 0.625)),
    ("gray", "mean"): ((0.373, 0.298, 0.315), (0.713, 0.683, 0.625)),
    ("gray", 1): ((0.370, 0.295, 0.320), (0.725, 0.685, 0.630)),
    ("gray", 2): ((0.375, 0.300, 0.310), (0.700, 0.680, 0.620)),
}

B2_PRINTED_7B = {
    "base": ((0.660, 0.575, 0.470), (0.940, 0.910, 0.840)),
    "real": ((0.765, 0.575, 0.460), (0.940, 0.905, 0.855)),
    "gray": ((0.720, 0.570, 0.460), (0.945, 0.905, 0.845)),
}

# Section 2.3: the blind-trained conditions finish within 0.04 of REAL on discovery at every
# density (two-seed means).
BLIND_CONDITIONS = ("none", "gray")
BLIND_GAP_BOUND = 0.04


def _b2_row(
    model: str,
    condition: str,
    seed: Any,
    level: str,
    density: str,
    value: float,
    members: int,
    printed: float,
) -> Row:
    return {
        "model": model,
        "train_condition": condition,
        "seed": seed,
        "level": level,
        "density": density,
        "eval_set": "scenes-development",
        "n_members": members,
        "value": value,
        "paper_value": printed,
        "tolerance": PRINTED_TOLERANCE,
    }


def _untrained_ladder(model: str) -> dict[tuple[str, str], tuple[float, int]]:
    return _ladder_levels(_evaluation(BASELINES, run=f"base-{model}", **SCENE_BASELINE))


def _virl39k_ladder(condition: str, seed: int) -> dict[tuple[str, str], tuple[float, int]]:
    """Levels of one ViRL39K-trained 3B run at step 100."""
    return _ladder_levels(
        _evaluation(LADDER_VIRL39K_3B, train_condition=condition, seed=seed, **LADDER_ARM)
    )


def _geometry3k_7b_ladder(condition: str) -> dict[tuple[str, str], tuple[float, int]]:
    """Levels of one Geometry3K-trained 7B run at step 100."""
    return _ladder_levels(
        _evaluation(LADDER_GEOMETRY3K_7B, train_condition=condition, **LADDER_ARM)
    )


def table_b2() -> list[Row]:
    """Discovery and identification-probe levels on the development scenes (Table B.2), and the
    Section 2.3 bound on the blind-trained conditions' distance from REAL."""
    untrained_3b = _untrained_ladder("3b")
    untrained_7b = _untrained_ladder("7b")

    rows: list[Row] = []
    for (condition, seed), printed in B2_PRINTED_3B.items():
        if condition == "base":
            cells = untrained_3b
        elif seed == "mean":
            per_seed = [_virl39k_ladder(condition, one) for one in (1, 2)]
            cells = {
                key: (float(np.mean([levels[key][0] for levels in per_seed])), per_seed[0][key][1])
                for key in per_seed[0]
            }
        else:
            cells = _virl39k_ladder(condition, seed)
        for index, (level, _) in enumerate(LEVELS):
            for position, density in enumerate(DENSITIES):
                value, members = cells[(level, density)]
                rows.append(
                    _b2_row(
                        "3b",
                        condition,
                        seed,
                        level,
                        density,
                        value,
                        members,
                        printed[index][position],
                    )
                )

    means = {
        (row["train_condition"], row["density"]): row["value"]
        for row in rows
        if row["seed"] == "mean" and row["level"] == "discovery"
    }
    for condition in BLIND_CONDITIONS:
        gap = max(
            abs(means[(condition, density)] - means[("real", density)]) for density in DENSITIES
        )
        row = _b2_row("3b", condition, "mean", "discovery", "largest gap to real over densities",
                      gap, rows[0]["n_members"], BLIND_GAP_BOUND)
        row["relation"] = "le"
        rows.append(row)

    for condition, printed in B2_PRINTED_7B.items():
        cells = untrained_7b if condition == "base" else _geometry3k_7b_ladder(condition)
        for index, (level, _) in enumerate(LEVELS):
            for position, density in enumerate(DENSITIES):
                value, members = cells[(level, density)]
                rows.append(
                    _b2_row(
                        "7b",
                        condition,
                        None,
                        level,
                        density,
                        value,
                        members,
                        printed[index][position],
                    )
                )
    return rows


_GRID: dict[str, dict[str, dict[str, Any]]] | None = None


def _untrained_suite() -> dict[str, dict[str, float]]:
    """Untrained 3B outcome per pair of the grounding suite, keyed by task and pair."""
    untrained: dict[str, dict[str, float]] = {}
    for row in _evaluation(BASELINES, **UNTRAINED_3B_SUITE):
        untrained.setdefault(row["task"], {})[row["pair_id"]] = _flag(row, "pair_correct")
    return untrained


def _gain_grid() -> dict[str, dict[str, dict[str, Any]]]:
    """Per-task gains of the Geometry3K-trained 3B conditions at step 100.

    Three-seed mean gains with paired-item intervals, plus each task's share of the
    condition's pair-count-weighted movement over the whole suite.
    """
    global _GRID
    if _GRID is not None:
        return _GRID

    untrained = _untrained_suite()
    task_ids = sorted(untrained)
    total_pairs = sum(len(untrained[task_id]) for task_id in task_ids)
    rng = np.random.default_rng(GAIN_SEED)

    grid: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in GAIN_CONDITIONS:
        per_task: dict[str, list[np.ndarray]] = {task_id: [] for task_id in task_ids}
        per_seed_movement: list[float] = []
        for seed in GAIN_SEEDS:
            scored: dict[str, dict[str, float]] = {}
            for row in _evaluation(
                SUITE_GEOMETRY3K_3B, train_condition=condition, seed=seed, **GAIN_ARM
            ):
                scored.setdefault(row["task"], {})[row["pair_id"]] = _flag(row, "pair_correct")
            moved = 0.0
            for task_id in task_ids:
                pair_ids = sorted(untrained[task_id])
                answers = scored.get(task_id, {})
                if set(answers) != set(pair_ids):
                    raise ValueError(f"{condition} seed {seed} {task_id}: pair sets differ")
                gains = np.array(
                    [answers[pair] - untrained[task_id][pair] for pair in pair_ids],
                    dtype=np.float64,
                )
                per_task[task_id].append(gains)
                moved += float(gains.sum())
            per_seed_movement.append(moved / total_pairs)
        movement = float(np.mean(per_seed_movement))
        entry: dict[str, dict[str, Any]] = {}
        for task_id in task_ids:
            stacked = np.mean(per_task[task_id], axis=0)
            mean = float(stacked.mean())
            low, high = _interval(
                _bootstrap_means(stacked, rng, draws=GAIN_DRAWS, block=GAIN_DRAWS)
            )
            contribution = mean * len(untrained[task_id]) / total_pairs
            entry[task_id] = {
                "mean": mean,
                "ci": (low, high),
                "per_seed": [float(gains.mean()) for gains in per_task[task_id]],
                "n_pairs": len(untrained[task_id]),
                "share_percent": 100.0 * contribution / movement if movement else None,
            }
        entry["movement"] = {"mean": movement, "per_seed": per_seed_movement}
        grid[condition] = entry
    _GRID = grid
    return grid


# Printed cells of Table B.3: three-seed mean gain, its interval, and the per-seed gains.
B3_PRINTED = {
    ("real", "cued-readout"): (0.143, (0.102, 0.188), (0.157, 0.127, 0.147)),
    ("real", "grounding"): (0.018, (-0.007, 0.044), (0.012, 0.022, 0.022)),
    ("real", "header-cued table"): (0.019, (-0.002, 0.042), (0.030, 0.013, 0.013)),
    ("caption", "cued-readout"): (0.107, (0.066, 0.149), (0.130, 0.113, 0.077)),
    ("caption", "grounding"): (0.006, (-0.018, 0.030), (0.003, 0.000, 0.013)),
    ("caption", "header-cued table"): (0.021, (-0.001, 0.044), (0.023, 0.027, 0.013)),
    ("none", "cued-readout"): (0.108, (0.070, 0.148), (0.093, 0.097, 0.133)),
    ("none", "grounding"): (-0.013, (-0.039, 0.013), (-0.010, -0.010, -0.020)),
    ("none", "header-cued table"): (0.023, (0.000, 0.049), (0.023, 0.020, 0.027)),
    ("gray", "cued-readout"): (0.138, (0.096, 0.182), (0.140, 0.137, 0.137)),
    ("gray", "grounding"): (-0.028, (-0.055, -0.002), (-0.030, -0.032, -0.023)),
    ("gray", "header-cued table"): (0.023, (0.002, 0.047), (0.027, 0.017, 0.027)),
}

B3_UNTRAINED = {"cued-readout": 0.320, "grounding": 0.455, "header-cued table": 0.867}
B3_CONDITION_ORDER = ("real", "caption", "none", "gray")
B3_TASK_ORDER = ("cued-readout", "grounding", "header-cued table")
B3_READOUT_SHARE = 72.1


def table_b3() -> list[Row]:
    """Task-layer levels and gains of the Geometry3K-trained 3B conditions (Table B.3)."""
    untrained = _untrained_suite()
    grid = _gain_grid()
    rows: list[Row] = []

    for task in B3_TASK_ORDER:
        task_id = SUITE_TASKS[task]
        outcomes = list(untrained[task_id].values())
        rows.append(
            {
                "quantity": "untrained pair accuracy",
                "task": task,
                "task_id": task_id,
                "train_condition": "base",
                "seed": None,
                "n_pairs": len(outcomes),
                "value": float(np.mean(outcomes)),
                "ci_low": None,
                "ci_high": None,
                "paper_value": B3_UNTRAINED[task],
                "paper_ci_low": None,
                "paper_ci_high": None,
                "tolerance": PRINTED_TOLERANCE,
            }
        )

    for condition in B3_CONDITION_ORDER:
        for task in B3_TASK_ORDER:
            task_id = SUITE_TASKS[task]
            cell = grid[condition][task_id]
            mean, interval, per_seed = B3_PRINTED[(condition, task)]
            rows.append(
                {
                    "quantity": "three-seed mean gain",
                    "task": task,
                    "task_id": task_id,
                    "train_condition": condition,
                    "seed": None,
                    "n_pairs": cell["n_pairs"],
                    "value": cell["mean"],
                    "ci_low": cell["ci"][0],
                    "ci_high": cell["ci"][1],
                    "paper_value": mean,
                    "paper_ci_low": interval[0],
                    "paper_ci_high": interval[1],
                    "tolerance": PRINTED_TOLERANCE,
                }
            )
            for index, seed in enumerate(GAIN_SEEDS):
                rows.append(
                    {
                        "quantity": "per-seed gain",
                        "task": task,
                        "task_id": task_id,
                        "train_condition": condition,
                        "seed": seed,
                        "n_pairs": cell["n_pairs"],
                        "value": cell["per_seed"][index],
                        "ci_low": None,
                        "ci_high": None,
                        "paper_value": per_seed[index],
                        "paper_ci_low": None,
                        "paper_ci_high": None,
                        "tolerance": PRINTED_TOLERANCE,
                    }
                )

    rows.append(
        {
            "quantity": "cued-readout share of movement (%)",
            "task": "cued-readout",
            "task_id": SUITE_TASKS["cued-readout"],
            "train_condition": "real",
            "seed": None,
            "n_pairs": sum(len(pairs) for pairs in untrained.values()),
            "value": grid["real"][SUITE_TASKS["cued-readout"]]["share_percent"],
            "ci_low": None,
            "ci_high": None,
            "paper_value": B3_READOUT_SHARE,
            "paper_ci_low": None,
            "paper_ci_high": None,
            "tolerance": 0.05,
        }
    )
    return rows


# Printed rows of Table C.1: instrument, task, role, pair count, untrained 3B level (None where
# the table prints a dash), untrained 7B level and its interval.
C1_PRINTED = (
    ("grounding-suite", "grounding", "find-and-bind (primary)",
     600, 0.455, 0.768, (0.733, 0.802)),
    ("grounding-suite", "header-cued table", "cued control, high baseline",
     300, 0.867, 0.993, (0.983, 1.000)),
    ("grounding-suite", "cued-readout", "cued readout, location marked",
     300, 0.320, 0.673, (0.620, 0.727)),
    ("grounding-twin", "grounding", "regenerated twin",
     600, None, 0.728, (0.692, 0.762)),
    ("grounding-twin", "cued-readout", "twin control",
     300, None, 0.623, (0.570, 0.677)),
    ("grounding-twin", "header-cued table", "twin control",
     300, None, 0.997, (0.990, 1.000)),
)

# Evaluations behind Table C.1. Pair counts, the untrained 3B levels and the image-removed
# floors come from each instrument's release evaluation, which carries the released pair ids.
# The untrained 7B levels come from the 7B scale study, whose record order the printed
# intervals were drawn in; the release evaluations give the same levels but other intervals.
RELEASE_EVALUATION = {"grounding-suite": "instrument-release", "grounding-twin": "twin-release"}
UNTRAINED_7B_LEVELS = "scale-study"
# Validation figures of the caption, both stated as bounds: the 7B model answering from
# question-blind 72B captions reaches at most 0.054, and both untrained models score exactly
# 0.000 with the image replaced by a gray canvas.
CAPTION_STRESS = {
    "run": "base-7b", "measurement": "caption-stress-72b-captions", "test_condition": "caption"
}
C1_SUITE_PAIRS = 1200
C1_CAPTION_CEILING = 0.054
C1_IMAGE_REMOVED_FLOOR = 0.0


def _pair_accuracy(rows: list[Row]) -> float:
    return float(np.mean(_flags(rows, "pair_correct")))


def _baseline(model: str, measurement: str, test_condition: str, eval_set: str) -> list[Row]:
    return _evaluation(
        BASELINES,
        run=f"base-{model}",
        measurement=measurement,
        test_condition=test_condition,
        eval_set=eval_set,
    )


def _c1_row(
    quantity: str,
    value: float | int,
    *,
    eval_set: str,
    test_condition: str,
    measurement: str,
    n_pairs: int,
    task: str | None = None,
    role: str | None = None,
    model: str | None = None,
    interval: tuple[float, float] | None = None,
    printed: float | int | None = None,
    printed_interval: tuple[float, float] | None = None,
    tolerance: float = PRINTED_TOLERANCE,
    relation: str | None = None,
) -> Row:
    return {
        "quantity": quantity,
        "task": task,
        "task_id": SUITE_TASKS.get(task) if task else None,
        "eval_set": eval_set,
        "role": role,
        "model": model,
        "test_condition": test_condition,
        "measurement": measurement,
        "n_pairs": n_pairs,
        "value": value,
        "ci_low": interval[0] if interval else None,
        "ci_high": interval[1] if interval else None,
        "paper_value": printed,
        "paper_ci_low": printed_interval[0] if printed_interval else None,
        "paper_ci_high": printed_interval[1] if printed_interval else None,
        "tolerance": tolerance if printed is not None else None,
        "relation": relation,
    }


def table_c1() -> list[Row]:
    """Untrained levels on the counterfactual suite and its twin, with the validation
    figures of the table caption (Table C.1)."""
    rows: list[Row] = []
    for eval_set, task, role, pairs, untrained_3b, untrained_7b, interval in C1_PRINTED:
        task_id = SUITE_TASKS[task]
        release = RELEASE_EVALUATION[eval_set]
        common = {"task": task, "role": role, "eval_set": eval_set, "test_condition": "real"}

        instrument = _task(_baseline("7b", release, "real", eval_set), task_id)
        rows.append(
            _c1_row(
                "pairs", len(instrument), measurement=release,
                n_pairs=len(instrument), printed=pairs, tolerance=0.0, **common,
            )
        )

        # The twin column of the 3B model is a dash in the table; its level is reported
        # without a printed value.
        small = _task(_baseline("3b", release, "real", eval_set), task_id)
        rows.append(
            _c1_row(
                "untrained pair accuracy", _pair_accuracy(small), model="3b",
                measurement=release, n_pairs=len(small), printed=untrained_3b, **common,
            )
        )

        large = _task(_baseline("7b", UNTRAINED_7B_LEVELS, "real", eval_set), task_id)
        outcomes = np.asarray(_flags(large, "pair_correct"), dtype=np.float64)
        rebuilt = _interval(
            _bootstrap_means(
                outcomes,
                np.random.default_rng(LEVEL_SEED),
                draws=LEVEL_DRAWS,
                block=LEVEL_BLOCK,
            )
        )
        rows.append(
            _c1_row(
                "untrained pair accuracy", float(outcomes.mean()), model="7b",
                measurement=UNTRAINED_7B_LEVELS, n_pairs=len(large), interval=rebuilt,
                printed=untrained_7b, printed_interval=interval, **common,
            )
        )

    suite = _baseline("7b", RELEASE_EVALUATION["grounding-suite"], "real", "grounding-suite")
    rows.append(
        _c1_row(
            "pairs", len(suite), eval_set="grounding-suite", test_condition="real",
            measurement=RELEASE_EVALUATION["grounding-suite"], n_pairs=len(suite),
            printed=C1_SUITE_PAIRS, tolerance=0.0,
        )
    )

    # The printed ceiling bounds the larger of the two instruments' caption-only pair accuracies.
    stress = {
        eval_set: _evaluation(BASELINES, eval_set=eval_set, **CAPTION_STRESS)
        for eval_set in RELEASE_EVALUATION
    }
    highest = max(stress, key=lambda eval_set: _pair_accuracy(stress[eval_set]))
    rows.append(
        _c1_row(
            "caption-only pair accuracy, maximum over suite and twin",
            _pair_accuracy(stress[highest]), eval_set=highest, model="7b",
            test_condition=CAPTION_STRESS["test_condition"],
            measurement=CAPTION_STRESS["measurement"], n_pairs=len(stress[highest]),
            printed=C1_CAPTION_CEILING, relation="le",
        )
    )

    # "Every image-removed condition scores exactly 0.000": one row per evaluation, with the
    # share of pairs whose two members received the same answer.
    for eval_set, release in RELEASE_EVALUATION.items():
        for model in ("3b", "7b"):
            blind = _baseline(model, release, "gray", eval_set)
            collapsed = [pair["answer_a"] == pair["answer_b"] for pair in blind]
            row = _c1_row(
                "image-removed pair accuracy", _pair_accuracy(blind), eval_set=eval_set,
                model=model, test_condition="gray", measurement=release, n_pairs=len(blind),
                printed=C1_IMAGE_REMOVED_FLOOR, tolerance=0.0, relation="zero",
            )
            row["collapsed_share"] = float(np.mean(collapsed))
            rows.append(row)
    return rows


# Plotted markers of Figure 2(a): gain, then the 95% interval where the panel draws one.
FIGURE2A_PRINTED = {
    ("cued_readout", "3b", "real"): (0.1433, (0.1022, 0.1878)),
    ("cued_readout", "3b", "caption"): (0.1067, (0.0656, 0.1489)),
    ("cued_readout", "3b", "none"): (0.1078, (0.0700, 0.1478)),
    ("cued_readout", "3b", "gray"): (0.1378, (0.0956, 0.1822)),
    ("find_and_bind", "3b", "real"): (0.0183, (-0.0067, 0.0444)),
    ("find_and_bind", "3b", "caption"): (0.0056, (-0.0183, 0.0300)),
    ("find_and_bind", "3b", "none"): (-0.0133, (-0.0389, 0.0128)),
    ("find_and_bind", "3b", "gray"): (-0.0283, (-0.0550, -0.0017)),
    ("discovery", "3b", "real"): (0.0850, None),
    ("discovery", "3b", "caption"): (0.0950, None),
    ("discovery", "3b", "none"): (0.0725, None),
    ("discovery", "3b", "gray"): (0.0700, None),
    ("discovery", "7b", "real"): (-0.0100, None),
    ("discovery", "7b", "gray"): (-0.0100, None),
}

FIGURE2A_TASKS = {"cued_readout": "cued-readout", "find_and_bind": "grounding"}
FIGURE2A_DENSITY = "n20"


def figure_2a() -> list[Row]:
    """Gains by task layer, the markers of Figure 2(a)."""
    grid = _gain_grid()
    untrained = {model: _untrained_ladder(model) for model in ("3b", "7b")}
    discovery = ("discovery", FIGURE2A_DENSITY)

    rows: list[Row] = []
    for (operation, model, condition), (printed, interval) in FIGURE2A_PRINTED.items():
        row: Row = {
            "operation": operation,
            "model": model,
            "train_condition": condition,
            "task": FIGURE2A_TASKS.get(operation, "coordinate scenes at 20 points"),
            "seeds": None,
            "n": None,
            "value": None,
            "ci_low": None,
            "ci_high": None,
            "paper_value": printed,
            "paper_ci_low": interval[0] if interval else None,
            "paper_ci_high": interval[1] if interval else None,
            "tolerance": PLOTTED_TOLERANCE,
        }
        if operation in FIGURE2A_TASKS:
            cell = grid[condition][SUITE_TASKS[FIGURE2A_TASKS[operation]]]
            row.update(
                {
                    "seeds": len(GAIN_SEEDS),
                    "n": cell["n_pairs"],
                    "value": cell["mean"],
                    "ci_low": cell["ci"][0],
                    "ci_high": cell["ci"][1],
                }
            )
        elif model == "3b":
            levels = [_virl39k_ladder(condition, seed)[discovery] for seed in (1, 2)]
            row.update(
                {
                    "seeds": len(levels),
                    "n": levels[0][1],
                    "value": float(np.mean([level for level, _ in levels]))
                    - untrained["3b"][discovery][0],
                }
            )
        else:
            level, members = _geometry3k_7b_ladder(condition)[discovery]
            row.update(
                {"seeds": 1, "n": members, "value": level - untrained["7b"][discovery][0]}
            )
        rows.append(row)
    return rows

TARGETS = {
    "tableB2": table_b2,
    "tableB3": table_b3,
    "tableC1": table_c1,
    "figure2a": figure_2a,
}

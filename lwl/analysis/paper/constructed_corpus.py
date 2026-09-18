"""Rebuilds of the constructed-scene results: the evaluation criteria of Appendix B, the
development grid, the confirmatory cells, the training-image attribution and transfer to the
independent grounding instruments (Tables G.1-G.4), and Figure 4. Every outcome is scored with
the canonical matcher (correct_a, correct_b, pair_correct).
"""
from __future__ import annotations

from typing import Any

from lwl.analysis.load import load_pairs, select
from lwl.analysis.paper import NOT_REBUILDABLE
from lwl.analysis.stats import clopper_pearson_interval, mean_with_paired_bootstrap

Row = dict[str, Any]

RUNS = "constructed_corpus_runs"
BASELINES = "instrument_baselines"

UNTRAINED_7B = "base-7b"
STANDARD_RUN1 = "constructed-standard-7b-run1"
STANDARD_RUN2 = "constructed-standard-7b-run2"
GRAY_CONTROL = "constructed-gray-control-7b"

DEVELOPMENT = "scenes-development"
CONFIRMATORY = "scenes-confirmatory"
SUITE = "grounding-suite"
TWIN = "grounding-twin"

# Every evaluation is chosen by run and measurement and must hold each pair once. Trained
# checkpoints are read from their primary evaluation; run 1's separate early read of step 20 on
# the development set comes from an abandoned first attempt at that checkpoint and is not used.
# The untrained 7B model on the scenes is the scene baseline. On the grounding suite and twin it
# is the scale-study evaluation, whose record order the printed intervals resample and whose
# twin pairs carry the same identifiers as the trained runs'.
PRIMARY = "primary"
SCENE_BASELINE = "scene-baseline"
TRANSFER_BASELINE = "scale-study"
CAPTION_STRESS = "caption-stress-72b-captions"

DISCOVERY = "l3"
PROBE = "probe"
CUED_LEVELS = ("l1", "l2")
LAYER_LABEL = {DISCOVERY: "discovery", PROBE: "identification probe"}

# Each density holds 150 scene programs, 50 per role, and each program yields one pair per cue
# level. The printed cells are member accuracy pooled over the target_stable and invariance
# programs: 100 pairs, 200 answers per density. The target_switch pairs are not part of them.
CELL_ROLES = ("target_stable", "invariance")

DENSITIES = ("n8", "n12", "n20")
DENSITY_LABEL = {"n8": "8-pt", "n12": "12-pt", "n20": "20-pt"}
STEPS = (10, 20, 30, 50, 75, 100)
RUN_LABELS = (("run 1", STANDARD_RUN1), ("run 2", STANDARD_RUN2))
UNTRAINED_LABEL = "untrained 7B"

GROUNDING_TASK = "coordinate_register_twenty_point_x_v02"
CUED_READOUT_TASK = "starred_series_value_nine_v07"
HEADER_TABLE_TASK = "header_cued_table_code_v02"

BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260716

# Half of the last printed digit, with a guard for bounds that sit on the rounding boundary.
INTERVAL_TOLERANCE = 5e-4 + 1e-9

ATTRIBUTION_STEP = 30

# The appendix grids print the same checkpoints; Figure 4 draws every checkpoint.
GRID_STEPS = (30, 100)

_LOADED: dict[str, list[Row]] = {}


def _pairs(family: str) -> list[Row]:
    """A family's pair records, decompressed once and then filtered in memory."""
    if family not in _LOADED:
        _LOADED[family] = load_pairs(family)
    return _LOADED[family]


def _source(run: str, eval_set: str) -> tuple[str, str]:
    """The family and measurement holding the evaluation of `run` on `eval_set` that the paper
    reports."""
    if run == UNTRAINED_7B:
        scenes = eval_set in (DEVELOPMENT, CONFIRMATORY)
        return BASELINES, SCENE_BASELINE if scenes else TRANSFER_BASELINE
    return RUNS, PRIMARY


def _records(run: str, step: int, eval_set: str, test_condition: str, **fields: Any) -> list[Row]:
    family, measurement = _source(run, eval_set)
    return select(
        _pairs(family),
        run=run,
        measurement=measurement,
        step=step,
        eval_set=eval_set,
        test_condition=test_condition,
        **fields,
    )


def _keyed(rows: list[Row], label: str) -> dict[str, Row]:
    """Records keyed by pair id, in file order; a pair seen twice is an error."""
    cells: dict[str, Row] = {}
    for row in rows:
        if row["pair_id"] in cells:
            raise ValueError(f"{label}: pair {row['pair_id']} appears twice")
        cells[row["pair_id"]] = row
    return cells


def _cell(run: str, step: int, eval_set: str, test_condition: str, **fields: Any) -> dict[str, Row]:
    """One evaluation keyed by pair id, in file order."""
    return _keyed(
        _records(run, step, eval_set, test_condition, **fields), f"{run} step {step} on {eval_set}"
    )


def _scene_cell(run: str, step: int, eval_set: str, density: str, test_condition: str = "real",
                layer: str = DISCOVERY) -> dict[str, Row]:
    return _cell(run, step, eval_set, test_condition, layer=layer, density=density, role=CELL_ROLES)


def _instrument(run: str, step: int, eval_set: str, task: str) -> dict[str, Row]:
    return _cell(run, step, eval_set, "real", task=task)


def _summarise(contributions: list[float]) -> tuple[float, list[float] | None]:
    """Mean per-pair contribution with its paired bootstrap interval; a constant vector has none."""
    estimate = sum(contributions) / len(contributions)
    if len(set(contributions)) <= 1:
        return estimate, None
    summary = mean_with_paired_bootstrap(
        contributions, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    return summary["estimate"], summary["ci95"]


def _member_accuracy(cells: dict[str, Row]) -> tuple[int, int, float]:
    """Correct members, members, and their ratio."""
    successes = sum(
        int(bool(row["correct_a"])) + int(bool(row["correct_b"])) for row in cells.values()
    )
    total = 2 * len(cells)
    return successes, total, successes / total


def _member_delta(base: dict[str, Row], trained: dict[str, Row]):
    """Member-accuracy difference over the pairs both sides evaluated, resampled by pair."""
    shared = sorted(set(base) & set(trained))
    contributions = []
    for pair_id in shared:
        before, after = base[pair_id], trained[pair_id]
        contributions.append(
            (
                (float(bool(after["correct_a"])) - float(bool(before["correct_a"])))
                + (float(bool(after["correct_b"])) - float(bool(before["correct_b"])))
            )
            / 2.0
        )
    estimate, interval = _summarise(contributions)
    return estimate, interval, len(shared)


def _pair_accuracy(cells: dict[str, Row]):
    """Pair accuracy with its bootstrap interval, or the exact interval when every pair agrees."""
    flags = [float(bool(row["pair_correct"])) for row in cells.values()]
    successes = int(sum(flags))
    estimate, interval = _summarise(flags)
    if interval is None:
        interval = clopper_pearson_interval(successes, len(flags))
    return successes, len(flags), estimate, interval


def _pair_delta(base: dict[str, Row], trained: dict[str, Row]):
    shared = sorted(set(base) & set(trained))
    contributions = [
        float(bool(trained[pair_id]["pair_correct"])) - float(bool(base[pair_id]["pair_correct"]))
        for pair_id in shared
    ]
    estimate, interval = _summarise(contributions)
    return estimate, interval, len(shared)


def _low(interval):
    return None if interval is None else interval[0]


def _high(interval):
    return None if interval is None else interval[1]


def _estimate(value, interval, printed=(None, None, None), relation=None, **extra) -> Row:
    """The value and interval columns every row shares, plus the row's own fields.

    A rebuilt interval sits beside the printed one under ci_low and ci_high; where the paper
    prints no interval, a rebuilt one goes under ci_low_rebuilt and ci_high_rebuilt instead.
    """
    compared = printed[1] is not None
    row = {
        "value": value,
        "ci_low": _low(interval) if compared else None,
        "ci_high": _high(interval) if compared else None,
        "paper_value": printed[0],
        "paper_ci_low": printed[1],
        "paper_ci_high": printed[2],
        "interval_tolerance": INTERVAL_TOLERANCE if compared else None,
        "relation": relation,
        "ci_low_rebuilt": None if compared else _low(interval),
        "ci_high_rebuilt": None if compared else _high(interval),
    }
    row.update(extra)
    return row


# Appendix B: the five criteria each evaluation cell had to meet before training. Criteria 1-3
# are read from the untrained 7B model: discovery between 0.20 and 0.80, at least 0.05 below both
# cued levels, and exactly zero on discovery and its probe with the image replaced or removed.
# Criterion 4 compares caption-only discovery accuracy (captions written by a 72B model without
# the question) against half of the with-image discovery accuracy. The released caption-only
# evaluation covers the development set, on the target_stable and target_switch programs.
DISCOVERY_RANGE = (0.20, 0.80)
CUED_MARGIN = 0.05
CRITERIA_SETS = (("development", DEVELOPMENT), ("confirmatory", CONFIRMATORY))
IMAGE_REMOVED = ("gray", "none")
PRINTED_CAPTION_ONLY = {("development", "n8"): 0.580, ("development", "n12"): 0.445}
PRINTED_CAPTION_THRESHOLD = {
    ("development", "n8"): 0.330,
    ("development", "n12"): 0.288,
    ("confirmatory", "n8"): 0.293,
    ("confirmatory", "n12"): 0.280,
    ("confirmatory", "n20"): 0.213,
}
# Criterion 4 fails on the development set at 8 and 12 points and holds at 20.
CAPTION_CRITERION_MET = {"n8": False, "n12": False, "n20": True}
# Printed in Appendix B without released records behind them: caption-only accuracy on the
# confirmatory set and the construction-artifact checks.
UNRELEASED_CRITERIA = (
    ("confirmatory", "n8", 4, "caption-only discovery accuracy", 0.570),
    ("confirmatory", "n12", 4, "caption-only discovery accuracy", 0.410),
    ("confirmatory", "n20", 4, "caption-only discovery accuracy", 0.130),
    ("development", "n12", 5, "artifact check, worst of three seeds", 0.557),
    ("development", "n20", 5, "artifact check, worst of three seeds", 0.565),
)

# Table G.1: (run, step, density) -> level, its interval, the difference, its interval; then the
# identification probe at 8, 12 and 20 points per checkpoint.
PRINTED_DEVELOPMENT: dict[tuple[str, int, str], tuple] = {
    (UNTRAINED_LABEL, 0, "n8"): (0.660, 0.590, 0.725, None, None, None),
    (UNTRAINED_LABEL, 0, "n12"): (0.575, 0.503, 0.644, None, None, None),
    (UNTRAINED_LABEL, 0, "n20"): (0.470, 0.399, 0.542, None, None, None),
    ("run 1", 30, "n8"): (0.965, 0.929, 0.986, 0.305, 0.235, 0.380),
    ("run 1", 30, "n12"): (0.890, 0.838, 0.930, 0.315, 0.230, 0.400),
    ("run 1", 30, "n20"): (0.800, 0.738, 0.853, 0.330, 0.250, 0.410),
    ("run 1", 100, "n8"): (1.000, 0.982, 1.000, 0.340, 0.260, 0.420),
    ("run 1", 100, "n12"): (0.940, 0.898, 0.969, 0.365, 0.275, 0.455),
    ("run 1", 100, "n20"): (0.910, 0.861, 0.946, 0.440, 0.355, 0.525),
    ("run 2", 30, "n8"): (0.960, 0.923, 0.983, 0.300, 0.225, 0.380),
    ("run 2", 30, "n12"): (0.860, 0.804, 0.905, 0.285, 0.205, 0.365),
    ("run 2", 30, "n20"): (0.795, 0.732, 0.849, 0.325, 0.245, 0.405),
    ("run 2", 100, "n8"): (0.995, 0.972, 1.000, 0.335, 0.255, 0.415),
    ("run 2", 100, "n12"): (0.950, 0.910, 0.976, 0.375, 0.290, 0.465),
    ("run 2", 100, "n20"): (0.905, 0.856, 0.942, 0.435, 0.350, 0.525),
}
PRINTED_DEVELOPMENT_PROBE: dict[tuple[str, int], tuple[float, float, float]] = {
    (UNTRAINED_LABEL, 0): (0.940, 0.910, 0.840),
    ("run 1", 30): (1.000, 0.975, 0.975),
    ("run 1", 100): (1.000, 0.990, 0.995),
    ("run 2", 30): (1.000, 0.960, 0.960),
    ("run 2", 100): (0.995, 0.985, 0.990),
}
# The grid caption: 0 of 200 with the image removed, on discovery and the probe, at every one
# of the six checkpoints of both runs.
PRINTED_IMAGE_REMOVED_COUNT = 0

# Table G.2: (run, step, density, test image) -> level, its interval, the difference, its
# interval. The gray-canvas changes are printed for the gray-trained control only; every other
# gray-canvas cell is exactly 0 of 200.
PRINTED_CONFIRMATORY: dict[tuple[str, int, str, str], tuple] = {
    (UNTRAINED_LABEL, 0, "n8", "real"): (0.585, 0.513, 0.654, None, None, None),
    (UNTRAINED_LABEL, 0, "n12", "real"): (0.560, 0.488, 0.630, None, None, None),
    (UNTRAINED_LABEL, 0, "n20", "real"): (0.425, 0.356, 0.497, None, None, None),
    (UNTRAINED_LABEL, 0, "n8", "gray"): (0.000, 0.000, 0.018, None, None, None),
    (UNTRAINED_LABEL, 0, "n12", "gray"): (0.000, 0.000, 0.018, None, None, None),
    (UNTRAINED_LABEL, 0, "n20", "gray"): (0.000, 0.000, 0.018, None, None, None),
    ("run 1", 30, "n8", "real"): (0.935, 0.891, 0.965, 0.350, 0.270, 0.430),
    ("run 1", 30, "n12", "real"): (0.860, 0.804, 0.905, 0.300, 0.225, 0.375),
    ("run 1", 30, "n20", "real"): (0.740, 0.673, 0.799, 0.315, 0.230, 0.400),
    ("run 1", 30, "n8", "gray"): (0.000, 0.000, 0.018, None, None, None),
    ("run 1", 30, "n12", "gray"): (0.000, 0.000, 0.018, None, None, None),
    ("run 1", 30, "n20", "gray"): (0.000, 0.000, 0.018, None, None, None),
    ("run 1", 100, "n8", "real"): (0.990, 0.964, 0.999, 0.405, 0.320, 0.490),
    ("run 1", 100, "n12", "real"): (0.940, 0.898, 0.969, 0.380, 0.295, 0.465),
    ("run 1", 100, "n20", "real"): (0.875, 0.821, 0.917, 0.450, 0.365, 0.540),
    ("run 1", 100, "n8", "gray"): (0.000, 0.000, 0.018, None, None, None),
    ("run 1", 100, "n12", "gray"): (0.000, 0.000, 0.018, None, None, None),
    ("run 1", 100, "n20", "gray"): (0.000, 0.000, 0.018, None, None, None),
    ("run 2", 30, "n8", "real"): (0.915, 0.867, 0.950, 0.330, 0.245, 0.415),
    ("run 2", 30, "n12", "real"): (0.850, 0.793, 0.896, 0.290, 0.215, 0.365),
    ("run 2", 30, "n20", "real"): (0.720, 0.652, 0.781, 0.295, 0.215, 0.380),
    ("run 2", 30, "n8", "gray"): (0.000, 0.000, 0.018, None, None, None),
    ("run 2", 30, "n12", "gray"): (0.000, 0.000, 0.018, None, None, None),
    ("run 2", 30, "n20", "gray"): (0.000, 0.000, 0.018, None, None, None),
    ("gray-trained control", 30, "n8", "real"): (0.590, 0.518, 0.659, 0.005, -0.035, 0.050),
    ("gray-trained control", 30, "n12", "real"): (0.580, 0.508, 0.649, 0.020, -0.030, 0.070),
    ("gray-trained control", 30, "n20", "real"): (0.385, 0.317, 0.456, -0.040, -0.085, 0.000),
    ("gray-trained control", 30, "n8", "gray"): (0.050, 0.024, 0.090, 0.050, 0.020, 0.090),
    ("gray-trained control", 30, "n12", "gray"): (0.060, 0.031, 0.102, 0.060, 0.025, 0.105),
    ("gray-trained control", 30, "n20", "gray"): (0.090, 0.054, 0.139, 0.090, 0.045, 0.145),
}
# The confirmatory caption: 150 scene programs per density.
PRINTED_PROGRAMS_PER_DENSITY = 150

CONFIRMATORY_RUNS = (
    (UNTRAINED_LABEL, UNTRAINED_7B, 0),
    ("run 1", STANDARD_RUN1, 30),
    ("run 1", STANDARD_RUN1, 100),
    ("run 2", STANDARD_RUN2, 30),
    ("gray-trained control", GRAY_CONTROL, 30),
)

# Only the 20-point cell met all five criteria.
ACCEPTED_DENSITY = "n20"

# Table G.3 at matched step 30: (set, density, cue level) -> the real run's gain, the
# gray-trained control's change and their difference, each as (value, low, high), then the
# gray-canvas correct counts of the real run and the control out of 200. The confirmatory
# differences are printed without an interval.
PRINTED_ATTRIBUTION_TABLE: dict[tuple[str, str, str], tuple] = {
    ("dev", "n8", DISCOVERY): (
        (0.305, 0.235, 0.380), (0.030, -0.005, 0.065), (0.275, 0.200, 0.350), 0, 11
    ),
    ("dev", "n8", PROBE): (
        (0.060, 0.025, 0.105), (0.005, -0.020, 0.030), (0.055, 0.020, 0.100), 0, 0
    ),
    ("dev", "n12", DISCOVERY): (
        (0.315, 0.230, 0.400), (-0.035, -0.080, 0.005), (0.350, 0.265, 0.435), 0, 9
    ),
    ("dev", "n12", PROBE): (
        (0.065, 0.025, 0.115), (-0.005, -0.035, 0.020), (0.070, 0.030, 0.120), 0, 0
    ),
    ("dev", "n20", DISCOVERY): (
        (0.330, 0.250, 0.410), (-0.030, -0.070, 0.005), (0.360, 0.275, 0.445), 0, 8
    ),
    ("dev", "n20", PROBE): (
        (0.135, 0.085, 0.195), (0.020, 0.000, 0.045), (0.115, 0.065, 0.170), 0, 0
    ),
    ("conf", "n8", DISCOVERY): (
        (0.350, 0.270, 0.430), (0.005, -0.035, 0.050), (0.345, None, None), 0, 10
    ),
    ("conf", "n12", DISCOVERY): (
        (0.300, 0.225, 0.375), (0.020, -0.030, 0.070), (0.280, None, None), 0, 12
    ),
    ("conf", "n20", DISCOVERY): (
        (0.315, 0.230, 0.400), (-0.040, -0.085, 0.000), (0.355, None, None), 0, 18
    ),
}
ATTRIBUTION_SETS = (
    ("dev", DEVELOPMENT, (DISCOVERY, PROBE)),
    ("conf", CONFIRMATORY, (DISCOVERY,)),
)
TASK_LABEL = {DISCOVERY: "L3", PROBE: "probe"}

INSTRUMENTS = (
    ("grounding task", SUITE, GROUNDING_TASK),
    ("twin", TWIN, GROUNDING_TASK),
    ("cued-readout task", SUITE, CUED_READOUT_TASK),
    ("header-cued table", SUITE, HEADER_TABLE_TASK),
)

# Table G.4: (run, step, instrument) -> pair accuracy and its interval, then the change from the
# untrained model and its interval. The untrained row is shared by both runs; run 2's twin was
# not evaluated.
PRINTED_TRANSFER: dict[tuple[str, int, str], tuple] = {
    (UNTRAINED_LABEL, 0, "grounding task"): (0.768, 0.733, 0.802, None, None, None),
    (UNTRAINED_LABEL, 0, "twin"): (0.728, 0.692, 0.762, None, None, None),
    (UNTRAINED_LABEL, 0, "cued-readout task"): (0.673, 0.620, 0.727, None, None, None),
    (UNTRAINED_LABEL, 0, "header-cued table"): (0.993, 0.983, 1.000, None, None, None),
    ("run 1", 30, "grounding task"): (0.930, 0.908, 0.950, 0.162, 0.130, 0.193),
    ("run 1", 30, "twin"): (0.905, 0.882, 0.928, 0.177, 0.143, 0.210),
    ("run 1", 30, "cued-readout task"): (0.680, 0.627, 0.733, 0.007, -0.023, 0.037),
    ("run 1", 30, "header-cued table"): (0.997, 0.990, 1.000, 0.003, 0.000, 0.010),
    ("run 1", 100, "grounding task"): (0.948, 0.930, 0.965, 0.180, 0.147, 0.215),
    ("run 1", 100, "twin"): (0.937, 0.917, 0.955, 0.208, 0.175, 0.243),
    ("run 1", 100, "cued-readout task"): (0.707, 0.657, 0.757, 0.033, 0.000, 0.070),
    ("run 1", 100, "header-cued table"): (0.997, 0.990, 1.000, 0.003, 0.000, 0.010),
    ("run 2", 30, "grounding task"): (0.935, 0.915, 0.955, 0.167, 0.137, 0.198),
    ("run 2", 30, "cued-readout task"): (0.690, 0.640, 0.743, 0.017, -0.013, 0.047),
    ("run 2", 30, "header-cued table"): (0.997, 0.990, 1.000, 0.003, 0.000, 0.010),
    ("run 2", 100, "grounding task"): (0.962, 0.947, 0.977, 0.193, 0.162, 0.227),
    ("run 2", 100, "cued-readout task"): (0.727, 0.677, 0.777, 0.053, 0.020, 0.087),
    ("run 2", 100, "header-cued table"): (0.997, 0.990, 1.000, 0.003, 0.000, 0.010),
}
TRANSFER_RUNS = {"run 1": STANDARD_RUN1, "run 2": STANDARD_RUN2}
NOT_EVALUATED = "not evaluated"

# Figure 4 panels (b) and (c), as labelled in the figure and stated in the text. The text gives
# the control's changes as a range, -0.035 to +0.030, whose ends are the 12- and 8-point cells.
# Panel (c) draws the changes at step 100 that Table G.4 prints.
PRINTED_ATTRIBUTION: dict[tuple[str, str], tuple] = {
    ("standard run", "n8"): (0.305, 0.235, 0.380),
    ("standard run", "n12"): (0.315, 0.230, 0.400),
    ("standard run", "n20"): (0.330, 0.250, 0.410),
    ("gray-trained control", "n8"): (0.030, None, None),
    ("gray-trained control", "n12"): (-0.035, None, None),
    ("visually attributable gain", "n20"): (0.360, 0.275, 0.445),
}
TRANSFER_PANEL_STEP = 100

# Values the text prints for figure 4(a): the identification probe on the development set
# before and after training (lowest and highest density), printed to two decimals, and the
# bound on the gap between the two runs' final discovery levels.
PRINTED_PROBE_RANGE: dict[int, tuple[float, float]] = {0: (0.84, 0.94), 100: (0.99, 1.00)}
PRINTED_SEED_AGREEMENT = 0.010
TWO_DECIMAL_TOLERANCE = 5e-3
# The panel's reward annotation; the mean training reward is logged by the trainer and is not
# part of the released per-item outputs.
PRINTED_REWARD_STEP = 13


def _criteria_row(set_label: str, density: str, criterion: int, quantity: str, value, printed,
                  relation: str | None = None, n: int | None = None,
                  note: str | None = None) -> Row:
    return {
        "set": set_label,
        "density": DENSITY_LABEL[density],
        "criterion": criterion,
        "quantity": quantity,
        "value": value,
        "paper_value": printed,
        "tolerance": INTERVAL_TOLERANCE if relation is None and printed is not None else None,
        "relation": relation,
        "n": n,
        "status": None,
        "note": note,
    }


def evaluation_criteria() -> list[Row]:
    """Appendix B: the evaluation criteria of each development and confirmatory cell, as far as
    the released records reach."""
    rows: list[Row] = []
    for set_label, eval_set in CRITERIA_SETS:
        for density in DENSITIES:
            _, total, discovery = _member_accuracy(
                _scene_cell(UNTRAINED_7B, 0, eval_set, density)
            )
            low, high = DISCOVERY_RANGE
            rows.append(_criteria_row(set_label, density, 1, "untrained discovery accuracy",
                                      discovery, low, "ge", total))
            rows.append(_criteria_row(set_label, density, 1, "untrained discovery accuracy",
                                      discovery, high, "le", total))
            cued = min(
                _member_accuracy(_scene_cell(UNTRAINED_7B, 0, eval_set, density, layer=layer))[2]
                for layer in CUED_LEVELS
            )
            rows.append(_criteria_row(set_label, density, 2, "margin below the easier cued level",
                                      cued - discovery, CUED_MARGIN, "ge", total))
            removed = max(
                _member_accuracy(
                    _scene_cell(UNTRAINED_7B, 0, eval_set, density, condition, layer)
                )[0]
                for condition in IMAGE_REMOVED
                for layer in (DISCOVERY, PROBE)
            )
            rows.append(_criteria_row(
                set_label, density, 3, "largest correct count with the image replaced or removed",
                removed, PRINTED_IMAGE_REMOVED_COUNT, "zero", total,
                "discovery and the probe, gray canvas and no image",
            ))
            threshold = discovery / 2
            rows.append(_criteria_row(set_label, density, 4, "caption-only threshold", threshold,
                                      PRINTED_CAPTION_THRESHOLD.get((set_label, density)), n=total,
                                      note="half of the with-image discovery accuracy"))
            if eval_set != DEVELOPMENT:
                continue
            captioned = select(
                _pairs(BASELINES), run=UNTRAINED_7B, measurement=CAPTION_STRESS, step=0,
                eval_set=eval_set, test_condition="caption", layer=DISCOVERY, density=density,
            )
            caption_only = total = None
            if captioned:
                _, total, caption_only = _member_accuracy(_keyed(captioned, CAPTION_STRESS))
            rows.append(_criteria_row(set_label, density, 4, "caption-only discovery accuracy",
                                      caption_only, PRINTED_CAPTION_ONLY.get((set_label, density)),
                                      n=total, note="target_stable and target_switch programs"))
            met = CAPTION_CRITERION_MET[density]
            margin = None if caption_only is None else caption_only - threshold
            rows.append(_criteria_row(
                set_label, density, 4, "caption-only accuracy minus threshold", margin, 0.0,
                "le" if met else "gt", total, "criterion met" if met else "criterion not met",
            ))
    for set_label, density, criterion, quantity, printed in UNRELEASED_CRITERIA:
        row = _criteria_row(set_label, density, criterion, quantity, None, printed,
                            note="no released records behind this value")
        row["status"] = NOT_REBUILDABLE
        rows.append(row)
    return rows


def _image_removed_development() -> list[Row]:
    """The grid caption's claim: one row per image-removed cell, then the largest correct count
    over all of them against the printed 0 of 200."""
    rows: list[Row] = []
    for label, run in RUN_LABELS:
        for step in STEPS:
            for layer in (DISCOVERY, PROBE):
                for density in DENSITIES:
                    cells = _scene_cell(run, step, DEVELOPMENT, density, "gray", layer)
                    successes, total, level = _member_accuracy(cells)
                    rows.append(
                        {
                            "run": label,
                            "step": step,
                            "density": DENSITY_LABEL[density],
                            "layer": LAYER_LABEL[layer],
                            "test_image": "gray",
                            "quantity": "level",
                            **_estimate(level, clopper_pearson_interval(successes, total)),
                            "successes": successes,
                            "n": total,
                        }
                    )
    rows.append(
        {
            "run": "runs 1 and 2",
            "step": None,
            "density": "all",
            "layer": "discovery and identification probe",
            "test_image": "gray",
            "quantity": "largest correct count over every trained checkpoint",
            **_estimate(
                max(row["successes"] for row in rows),
                None,
                (PRINTED_IMAGE_REMOVED_COUNT, None, None),
                "zero",
            ),
            "successes": None,
            "n": max(row["n"] for row in rows),
        }
    )
    return rows


def _grid_checkpoints() -> list[tuple[str, str, int]]:
    """(label, run, step) of each checkpoint the appendix grids print, the untrained model first."""
    return [(UNTRAINED_LABEL, UNTRAINED_7B, 0)] + [
        (label, run, step) for label, run in RUN_LABELS for step in GRID_STEPS
    ]


def development_grid() -> list[Row]:
    """Table G.1: discovery member accuracy and the identification probe on the development set
    at steps 0, 30 and 100, and the image-removed floor at every checkpoint."""
    base = {density: _scene_cell(UNTRAINED_7B, 0, DEVELOPMENT, density) for density in DENSITIES}
    rows: list[Row] = []
    for label, run, step in _grid_checkpoints():
        probes = PRINTED_DEVELOPMENT_PROBE[(label, step)]
        for density, probe_printed in zip(DENSITIES, probes):
            cells = base[density] if step == 0 else _scene_cell(run, step, DEVELOPMENT, density)
            successes, total, level = _member_accuracy(cells)
            printed = PRINTED_DEVELOPMENT[(label, step, density)]
            shared = {
                "run": label,
                "step": step,
                "density": DENSITY_LABEL[density],
                "layer": LAYER_LABEL[DISCOVERY],
                "test_image": "real",
            }
            rows.append(
                {
                    **shared,
                    "quantity": "level",
                    **_estimate(level, clopper_pearson_interval(successes, total), printed[:3]),
                    "successes": successes,
                    "n": total,
                }
            )
            if step > 0:
                delta, interval, pairs = _member_delta(base[density], cells)
                rows.append(
                    {
                        **shared,
                        "quantity": "difference vs untrained",
                        **_estimate(delta, interval, printed[3:]),
                        "successes": None,
                        "n": pairs,
                    }
                )
            successes, total, level = _member_accuracy(
                _scene_cell(run, step, DEVELOPMENT, density, layer=PROBE)
            )
            rows.append(
                {
                    **shared,
                    "layer": LAYER_LABEL[PROBE],
                    "quantity": "level",
                    **_estimate(level, None, (probe_printed, None, None)),
                    "successes": successes,
                    "n": total,
                }
            )
    return rows + _image_removed_development()


def _programs_per_density() -> Row:
    """The confirmatory caption's program count, read from the untrained model's evaluation."""
    counts = [
        len({
            row["scene_id"]
            for row in _records(UNTRAINED_7B, 0, CONFIRMATORY, "real", density=density)
        })
        for density in DENSITIES
    ]
    return {
        "run": "confirmatory set",
        "step": None,
        "density": "each",
        "test_image": None,
        "quantity": "scene programs",
        **_estimate(
            counts[0] if len(set(counts)) == 1 else None,
            None,
            (PRINTED_PROGRAMS_PER_DENSITY, None, None),
        ),
        "accepted": None,
        "successes": None,
        "n": sum(counts),
        "note": ", ".join(
            f"{DENSITY_LABEL[density]}: {count}" for density, count in zip(DENSITIES, counts)
        ),
    }


def confirmatory_cells() -> list[Row]:
    """Table G.2: every confirmatory cell, with real images and with a gray canvas."""
    base = {
        (density, condition): _scene_cell(UNTRAINED_7B, 0, CONFIRMATORY, density, condition)
        for density in DENSITIES
        for condition in ("real", "gray")
    }
    rows: list[Row] = []
    for label, run, step in CONFIRMATORY_RUNS:
        for condition in ("real", "gray"):
            for density in DENSITIES:
                cells = (
                    base[(density, condition)]
                    if run == UNTRAINED_7B
                    else _scene_cell(run, step, CONFIRMATORY, density, condition)
                )
                successes, total, level = _member_accuracy(cells)
                printed = PRINTED_CONFIRMATORY[(label, step, density, condition)]
                shared = {
                    "run": label,
                    "step": step,
                    "density": DENSITY_LABEL[density],
                    "test_image": condition,
                }
                accepted = "yes" if density == ACCEPTED_DENSITY else "no"
                exactly_zero = condition == "gray" and printed[0] == 0
                rows.append(
                    {
                        **shared,
                        "quantity": "level",
                        **_estimate(
                            level, clopper_pearson_interval(successes, total), printed[:3],
                            "zero" if exactly_zero else None,
                        ),
                        "accepted": accepted,
                        "successes": successes,
                        "n": total,
                    }
                )
                if run == UNTRAINED_7B:
                    continue
                delta, interval, pairs = _member_delta(base[(density, condition)], cells)
                rows.append(
                    {
                        **shared,
                        "quantity": "difference vs untrained",
                        **_estimate(delta, interval, printed[3:]),
                        "accepted": accepted,
                        "successes": None,
                        "n": pairs,
                    }
                )
    return rows + [_programs_per_density()]


def attribution() -> list[Row]:
    """Table G.3: at matched step 30, the real run's gain, the gray-trained control's change and
    their difference (tau_vis), with each model's gray-canvas count."""
    step = ATTRIBUTION_STEP
    rows: list[Row] = []
    for set_label, eval_set, layers in ATTRIBUTION_SETS:
        for density in DENSITIES:
            for layer in layers:
                base = _scene_cell(UNTRAINED_7B, 0, eval_set, density, layer=layer)
                standard = _scene_cell(STANDARD_RUN1, step, eval_set, density, layer=layer)
                control = _scene_cell(GRAY_CONTROL, step, eval_set, density, layer=layer)
                gains = (
                    ("real run gain", _member_delta(base, standard)),
                    ("control change", _member_delta(base, control)),
                    ("tau_vis", _member_delta(control, standard)),
                )
                printed = PRINTED_ATTRIBUTION_TABLE[(set_label, density, layer)]
                shared = {
                    "set": set_label,
                    "step": step,
                    "density": DENSITY_LABEL[density],
                    "task": TASK_LABEL[layer],
                }
                for (quantity, (estimate, interval, pairs)), printed_cell in zip(
                    gains, printed[:3]
                ):
                    rows.append(
                        {
                            **shared,
                            "quantity": quantity,
                            "test_image": "real",
                            **_estimate(estimate, interval, printed_cell),
                            "successes": None,
                            "n": pairs,
                        }
                    )
                for model, run, printed_count in (
                    ("real run", STANDARD_RUN1, printed[3]),
                    ("control", GRAY_CONTROL, printed[4]),
                ):
                    cells = _scene_cell(run, step, eval_set, density, "gray", layer)
                    successes, total, _ = _member_accuracy(cells)
                    rows.append(
                        {
                            **shared,
                            "quantity": f"{model}, correct with the image removed",
                            "test_image": "gray",
                            **_estimate(successes, None, (printed_count, None, None)),
                            "successes": successes,
                            "n": total,
                        }
                    )
    return rows


def transfer() -> list[Row]:
    """Table G.4: the grounding instruments at steps 0, 30 and 100, with each checkpoint's
    change from the untrained model."""
    rows: list[Row] = []
    bases = {
        name: _instrument(UNTRAINED_7B, 0, eval_set, task) for name, eval_set, task in INSTRUMENTS
    }
    checkpoints = [(UNTRAINED_LABEL, UNTRAINED_7B, 0)] + [
        (label, run, step) for label, run in TRANSFER_RUNS.items() for step in GRID_STEPS
    ]
    for label, run, step in checkpoints:
        for name, eval_set, task in INSTRUMENTS:
            printed = PRINTED_TRANSFER.get((label, step, name), (None,) * 6)
            cells = _instrument(run, step, eval_set, task)
            shared = {"run": label, "step": step, "instrument": name}
            if not cells:
                # A printed cell without records stays in the table, unmatched.
                note = NOT_EVALUATED if printed[0] is None else "no released records"
                rows.append({**shared, "quantity": "pair accuracy",
                             **_estimate(None, None, printed[:3]), "successes": None, "n": 0,
                             "note": note})
                if step > 0:
                    rows.append({**shared, "quantity": "change from untrained",
                                 **_estimate(None, None, printed[3:]), "successes": None,
                                 "n": 0, "note": note})
                continue
            successes, total, level, interval = _pair_accuracy(cells)
            rows.append(
                {**shared, "quantity": "pair accuracy", **_estimate(level, interval, printed[:3]),
                 "successes": successes, "n": total, "note": None}
            )
            if step == 0:
                continue
            estimate, interval, pairs = _pair_delta(bases[name], cells)
            rows.append(
                {**shared, "quantity": "change from untrained",
                 **_estimate(estimate, interval, printed[3:]), "successes": None, "n": pairs,
                 "note": None}
            )
    return rows


def _figure_row(panel: str, series: str, run: str, step: int | None, cell: str, quantity: str,
                estimate: Row, n: int | None, note: str | None = None) -> Row:
    return {
        "panel": panel,
        "series": series,
        "run": run,
        "step": step,
        "cell": cell,
        "quantity": quantity,
        **estimate,
        "n": n,
        "note": note,
    }


def _acquisition_panel() -> list[Row]:
    rows: list[Row] = []
    for label, run in RUN_LABELS:
        for step in (0,) + STEPS:
            for density in DENSITIES:
                cells = (
                    _scene_cell(UNTRAINED_7B, 0, DEVELOPMENT, density)
                    if step == 0
                    else _scene_cell(run, step, DEVELOPMENT, density)
                )
                successes, total, level = _member_accuracy(cells)
                key = (UNTRAINED_LABEL if step == 0 else label, step, density)
                printed = PRINTED_DEVELOPMENT.get(key)
                rows.append(
                    _figure_row(
                        "a",
                        "development, real image",
                        label,
                        step,
                        DENSITY_LABEL[density],
                        "discovery member accuracy",
                        _estimate(
                            level,
                            clopper_pearson_interval(successes, total),
                            printed[:3] if printed else (None, None, None),
                        ),
                        total,
                        None if printed else "plotted; Table G.1 prints steps 0, 30 and 100",
                    )
                )
    # The panel draws every image-removed discovery cell at the printed "Gray at test: 0".
    for label, run in RUN_LABELS:
        for step in STEPS:
            for density in DENSITIES:
                cells = _scene_cell(run, step, DEVELOPMENT, density, "gray")
                successes, total, level = _member_accuracy(cells)
                rows.append(
                    _figure_row(
                        "a",
                        "development, image removed",
                        label,
                        step,
                        DENSITY_LABEL[density],
                        "discovery member accuracy",
                        _estimate(
                            level, clopper_pearson_interval(successes, total), (0.0, None, None),
                            "zero",
                        ),
                        total,
                    )
                )
    inset = (
        ("run 1", UNTRAINED_7B, 0),
        ("run 1", STANDARD_RUN1, 30),
        ("run 1", STANDARD_RUN1, 100),
        ("run 2", STANDARD_RUN2, 30),
    )
    for label, run, step in inset:
        cells = _scene_cell(run, step, CONFIRMATORY, ACCEPTED_DENSITY)
        successes, total, level = _member_accuracy(cells)
        key = (UNTRAINED_LABEL if run == UNTRAINED_7B else label, step, ACCEPTED_DENSITY, "real")
        rows.append(
            _figure_row(
                "a inset",
                "confirmatory, real image",
                label,
                step,
                DENSITY_LABEL[ACCEPTED_DENSITY],
                "discovery member accuracy",
                _estimate(
                    level, clopper_pearson_interval(successes, total), PRINTED_CONFIRMATORY[key][:3]
                ),
                total,
            )
        )
    return rows + _acquisition_text()


def _acquisition_text() -> list[Row]:
    """Values the text prints for panel (a): the probe range, the second run's agreement and the
    step at which the training reward passes 0.97."""
    rows: list[Row] = []
    for label, run, step in ((UNTRAINED_LABEL, UNTRAINED_7B, 0), ("run 1", STANDARD_RUN1, 100)):
        counts = [
            _member_accuracy(_scene_cell(run, step, DEVELOPMENT, density, layer=PROBE))
            for density in DENSITIES
        ]
        levels = [level for _, _, level in counts]
        low, high = PRINTED_PROBE_RANGE[step]
        for end, value, printed in (("lowest", min(levels), low), ("highest", max(levels), high)):
            row = _figure_row(
                "a text",
                "development, real image",
                label,
                step,
                f"{end} over densities",
                "identification probe member accuracy",
                _estimate(value, None, (printed, None, None)),
                counts[0][1],
            )
            row["tolerance"] = TWO_DECIMAL_TOLERANCE
            rows.append(row)
    final = {
        label: [
            _member_accuracy(_scene_cell(run, 100, DEVELOPMENT, density))
            for density in DENSITIES
        ]
        for label, run in RUN_LABELS
    }
    largest_gap = max(
        abs(one[2] - two[2]) for one, two in zip(final["run 1"], final["run 2"])
    )
    rows.append(
        _figure_row(
            "a text",
            "development, real image",
            "run 2 against run 1",
            100,
            "largest gap over densities",
            "discovery member accuracy",
            _estimate(largest_gap, None, (PRINTED_SEED_AGREEMENT, None, None), "le"),
            final["run 1"][0][1],
            "printed as an upper bound",
        )
    )
    row = _figure_row(
        "a",
        "training reward",
        "run 1",
        None,
        "training batches",
        "first step with mean training reward above 0.97",
        _estimate(None, None, (PRINTED_REWARD_STEP, None, None)),
        None,
        "the mean training reward is logged by the trainer and is not part of the released "
        "per-item outputs",
    )
    row["status"] = NOT_REBUILDABLE
    rows.append(row)
    return rows


def _attribution_panel() -> list[Row]:
    rows: list[Row] = []
    for density in DENSITIES:
        base = _scene_cell(UNTRAINED_7B, 0, DEVELOPMENT, density)
        standard = _scene_cell(STANDARD_RUN1, ATTRIBUTION_STEP, DEVELOPMENT, density)
        control = _scene_cell(GRAY_CONTROL, ATTRIBUTION_STEP, DEVELOPMENT, density)
        estimates = (
            ("standard run", _member_delta(base, standard)),
            ("gray-trained control", _member_delta(base, control)),
            ("visually attributable gain", _member_delta(control, standard)),
        )
        for series, (estimate, interval, pairs) in estimates:
            printed = PRINTED_ATTRIBUTION.get((series, density), (None, None, None))
            rows.append(
                _figure_row(
                    "b",
                    series,
                    "run 1 and control, real image at test",
                    ATTRIBUTION_STEP,
                    DENSITY_LABEL[density],
                    "discovery gain",
                    _estimate(estimate, interval, printed),
                    pairs,
                )
            )
    return rows


def _transfer_panel() -> list[Row]:
    rows: list[Row] = []
    step = TRANSFER_PANEL_STEP
    for name, eval_set, task in INSTRUMENTS:
        base = _instrument(UNTRAINED_7B, 0, eval_set, task)
        for label, run in TRANSFER_RUNS.items():
            printed = PRINTED_TRANSFER.get((label, step, name), (None,) * 6)[3:]
            cells = _instrument(run, step, eval_set, task)
            if cells:
                estimate, interval, pairs = _pair_delta(base, cells)
                note = None
            else:
                estimate, interval, pairs = None, None, 0
                note = NOT_EVALUATED if printed[0] is None else "no released records"
            rows.append(
                _figure_row(
                    "c",
                    name,
                    label,
                    step,
                    "all pairs",
                    "gain in pair accuracy",
                    _estimate(estimate, interval, printed),
                    pairs,
                    note,
                )
            )
    return rows


def acquisition_figure() -> list[Row]:
    """Figure 4: the development curves, the training-image attribution, and transfer."""
    return _acquisition_panel() + _attribution_panel() + _transfer_panel()


TARGETS = {
    "appendixB": evaluation_criteria,
    "tableG1": development_grid,
    "tableG2": confirmatory_cells,
    "tableG3": attribution,
    "tableG4": transfer,
    "figure4": acquisition_figure,
}

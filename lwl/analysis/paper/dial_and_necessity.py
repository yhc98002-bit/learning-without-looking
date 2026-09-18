"""Rebuilds of the dose-response and capacity results (Tables H.1-H.3, Figure 5) and of the
base-model necessity audit (Figure 3(a) and the audit and corpus values quoted in Section 5).
"""
from __future__ import annotations

import json
import tarfile
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Any

import numpy as np

from lwl.analysis.load import load_audits, load_pairs, select
from lwl.analysis.paper import INPUT_MISSING
from lwl.analysis.stats import clopper_pearson_interval, mean_with_paired_bootstrap
from lwl.paths import data_path, predictions_root

Row = dict[str, Any]

DOSE = "dose_mixtures_7b"
RUNS = "constructed_corpus_runs"
BASELINES = "instrument_baselines"
AUDIT = "resolvability_audit"

UNTRAINED_7B = "base-7b"
UNTRAINED_3B = "base-3b"
STANDARD_7B = "constructed-standard-7b-run1"
STANDARD_3B = "constructed-standard-3b"

DEVELOPMENT = "scenes-development"
SUITE = "grounding-suite"
GROUNDING_TASK = "coordinate_register_twenty_point_x_v02"

# Every evaluation is chosen by run and measurement and must hold each pair once. Trained
# checkpoints are read from their primary evaluation; the standard 7B run's separate early-read
# of step 20 is not used. The untrained models on the scenes are the scene baselines. On the
# grounding task the untrained 3B model is the instrument-release evaluation and the untrained
# 7B model the scale-study evaluation, whose record orders the printed intervals resample; the
# 3B remeasure and the 7B release evaluation give the same answers in other orders.
PRIMARY = "primary"
SCENE_MEASUREMENT = {UNTRAINED_3B: "scene-baseline", UNTRAINED_7B: "scene-baseline"}
GROUNDING_MEASUREMENT = {UNTRAINED_3B: "instrument-release", UNTRAINED_7B: "scale-study"}

# Discovery and its probe are read over pair members of the two roles that keep the question
# fixed; the target-switch role is a separate measurement.
COMPOSITION_ROLES = ("target_stable", "invariance")
DENSITIES = ("n8", "n12", "n20")
DENSITY_LABEL = {"n8": "8-pt", "n12": "12-pt", "n20": "20-pt"}

# Training conditions of the dose comparison, by constructed fraction f. The f = 1 point is the
# standard constructed-corpus run read at the same step.
CONDITIONS = ("0", "1/3", "2/3", "1")
CONDITION_RUN = {
    "0": (DOSE, "dose-f0-7b"),
    "1/3": (DOSE, "dose-f1_3-7b"),
    "2/3": (DOSE, "dose-f2_3-7b"),
    "1": (RUNS, STANDARD_7B),
}
DOSE_STEP = 30

# Realized training streams under data/training, as released: each mixture's own 7,200-row
# stream, and the constructed corpus, which the f = 1 run read in order, cycling from the head.
STREAM_DIR = {
    "0": ("mixtures", "f0"),
    "1/3": ("mixtures", "f1_3"),
    "2/3": ("mixtures", "f2_3"),
    "1": ("constructed",),
}
STREAM_ROWS = 7200
PROMPTS_PER_STEP = STREAM_ROWS // DOSE_STEP
VIRL_PREFIX = "virl:"

# A mixture read at step 30 is compared with the f = 1 run at the step where it had seen the same
# number of constructed prompt presentations.
EXPOSURE_MATCH = {"1/3": 10, "2/3": 20}

# The base-model audit: 16 responses of the untrained 7B model per item with the image and 16
# with it removed, on the constructed training rows and the audited ViRL39K items. Audited
# success is scored with the training reward's answer matcher.
AUDIT_SETS = {"constructed": "scenes-training", "virl39k": "virl39k-audit"}
BLIND_CONDITION = "none"
AUDIT_SCORE = "p_sample_reward_matcher"
AUDIT_CORRECT = "correct_samples_reward_matcher"
CONSTRUCTED_ROWS = 2880
AUDIT_SAMPLES = 16
LATTICE_STEP = 1 / 16

BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260716

# Half of the last printed digit, with a guard for values that sit on the rounding boundary. The
# bootstraps replay the original resampling streams, so interval ends use the same tolerance.
PRINTED_TOLERANCE = 5e-4 + 1e-9
INTERVAL_TOLERANCE = PRINTED_TOLERANCE

_PAIRS: dict[str, list[Row]] = {}
_CACHE: dict[str, Any] = {}


def _pairs(family: str) -> list[Row]:
    """A family's pair records, read once and kept in the order of the released file."""
    if family not in _PAIRS:
        _PAIRS[family] = load_pairs(family)
    return _PAIRS[family]


def _by_pair(rows: list[Row], label: str) -> dict[str, Row]:
    """Records of one evaluation keyed by pair id, in record order."""
    cells: dict[str, Row] = {}
    for row in rows:
        if row["pair_id"] in cells:
            raise ValueError(f"{label}: pair {row['pair_id']} appears more than once")
        cells[row["pair_id"]] = row
    if not cells:
        raise ValueError(f"no records for {label}")
    return cells


def _scenes(family: str, run: str, step: int, density: str, layer: str = "l3",
            condition: str = "real") -> dict[str, Row]:
    measurement = SCENE_MEASUREMENT.get(run, PRIMARY)
    rows = select(
        _pairs(family), run=run, measurement=measurement, step=step, eval_set=DEVELOPMENT,
        layer=layer, density=density, role=COMPOSITION_ROLES, test_condition=condition,
    )
    return _by_pair(rows, f"{run} [{measurement}] step {step} {layer} {density} {condition}")


def _grounding(family: str, run: str, step: int) -> dict[str, Row]:
    """Grounding-task pairs of one checkpoint, in the record order of its evaluation."""
    measurement = GROUNDING_MEASUREMENT.get(run, PRIMARY)
    rows = select(_pairs(family), run=run, measurement=measurement, step=step, eval_set=SUITE,
                  task=GROUNDING_TASK, test_condition="real")
    return _by_pair(rows, f"{run} [{measurement}] step {step} grounding task")


def _source(model: str, step: int) -> tuple[str, str, int]:
    """(family, run, step) of the 3B or 7B model trained on the constructed corpus."""
    if step == 0:
        return BASELINES, UNTRAINED_3B if model == "3B" else UNTRAINED_7B, 0
    return RUNS, STANDARD_3B if model == "3B" else STANDARD_7B, step


def _members(row: Row) -> tuple[float, float]:
    return float(bool(row["correct_a"])), float(bool(row["correct_b"]))


def _member_accuracy(cells: dict[str, Row]) -> tuple[int, int, float]:
    """Correct members, members, and their ratio."""
    successes = int(sum(sum(_members(row)) for row in cells.values()))
    total = 2 * len(cells)
    return successes, total, successes / total


def _bootstrap(contributions: list[float]) -> list[float] | None:
    """Percentile interval of the mean over resampled items; none for a constant vector."""
    if len(set(contributions)) <= 1:
        return None
    return mean_with_paired_bootstrap(contributions, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)["ci95"]


def _member_delta(before: dict[str, Row], after: dict[str, Row]):
    """after - before in member accuracy over the shared pairs, with the paired interval."""
    shared = sorted(set(before) & set(after))
    contributions = []
    for pair_id in shared:
        (a0, b0), (a1, b1) = _members(before[pair_id]), _members(after[pair_id])
        contributions.append(((a1 - a0) + (b1 - b0)) / 2.0)
    return sum(contributions) / len(contributions), _bootstrap(contributions), len(shared)


def _pair_accuracy(cells: dict[str, Row]):
    """Pair accuracy with its bootstrap interval, or the exact interval when every pair agrees."""
    flags = [float(bool(row["pair_correct"])) for row in cells.values()]
    successes = int(sum(flags))
    interval = _bootstrap(flags)
    if interval is None:
        interval = clopper_pearson_interval(successes, len(flags))
    return successes, len(flags), successes / len(flags), interval


def _audit_records() -> dict[tuple[str, str], dict[str, Row]]:
    """Audit records per (corpus, test condition), keyed by item id."""
    if "audit_records" not in _CACHE:
        records: dict[tuple[str, str], dict[str, Row]] = {}
        for corpus, eval_set in AUDIT_SETS.items():
            for condition in ("real", BLIND_CONDITION):
                cell: dict[str, Row] = {}
                for row in load_audits(AUDIT, run=UNTRAINED_7B, measurement=PRIMARY, eval_set=eval_set,
                                       test_condition=condition):
                    if row["samples"] != AUDIT_SAMPLES or row.get(AUDIT_SCORE) is None:
                        raise ValueError(f"audit row {row['item_id']} is not a complete "
                                         f"{AUDIT_SAMPLES}-sample record")
                    if row["item_id"] in cell:
                        raise ValueError(f"{eval_set} {condition}: item {row['item_id']} appears twice")
                    cell[row["item_id"]] = row
                records[(corpus, condition)] = cell
            if set(records[(corpus, "real")]) != set(records[(corpus, BLIND_CONDITION)]):
                raise ValueError(f"{corpus} audit: the two conditions cover different items")
        if len(records[("constructed", "real")]) != CONSTRUCTED_ROWS:
            raise ValueError(f"constructed audit holds {len(records[('constructed', 'real')])} items, "
                             f"not {CONSTRUCTED_ROWS}")
        _CACHE["audit_records"] = records
    return _CACHE["audit_records"]


def _audit() -> dict[str, dict[str, tuple[float, float]]]:
    """Base-model audit per corpus: item id -> (b, q_blind), the sampled success with the image
    and with it removed."""
    if "audit" not in _CACHE:
        records = _audit_records()
        _CACHE["audit"] = {
            corpus: {
                item: (float(records[(corpus, "real")][item][AUDIT_SCORE]),
                       float(records[(corpus, BLIND_CONDITION)][item][AUDIT_SCORE]))
                for item in sorted(records[(corpus, "real")])
            }
            for corpus in AUDIT_SETS
        }
    return _CACHE["audit"]


def _necessary_and_learnable(b: float, q_blind: float) -> bool:
    return b - q_blind > 0 and 0 < b < 1


def _stream_text(parts: tuple[str, ...]) -> str:
    """One released train.jsonl, extracted or inside its archive, looked up in the data directory
    and then beside the predictions directory, where the released dataset places it."""
    searched = []
    for root in (data_path("training"), predictions_root().parent / "data" / "training"):
        folder = root.joinpath(*parts)
        extracted = folder / "train.jsonl"
        if extracted.is_file():
            return extracted.read_text(encoding="utf-8")
        archive = folder.parent / f"{folder.name}.tar.gz"
        if archive.is_file():
            with tarfile.open(archive) as tar:
                return tar.extractfile(f"{folder.name}/train.jsonl").read().decode("utf-8")
        searched += [str(extracted), str(archive)]
    raise FileNotFoundError(
        f"no training stream at {' or '.join(searched)}; fetch it with scripts/fetch_data.py --part corpora"
    )


def _stream(condition: str) -> list[Row]:
    """The realized training stream of one condition."""
    text = _stream_text(STREAM_DIR[condition])
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if condition == "1":
        rows = [rows[index % len(rows)] for index in range(STREAM_ROWS)]
    if len(rows) != STREAM_ROWS:
        raise ValueError(f"f = {condition}: stream holds {len(rows)} rows, not {STREAM_ROWS}")
    return rows


def _stream_key(row: Row) -> tuple[str, str]:
    """(corpus, audit item id) of one training row."""
    group = str(row["pair_group_uid"])
    if group.startswith(VIRL_PREFIX):
        return "virl39k", group[len(VIRL_PREFIX):]
    return "constructed", f"{group}:{row['pair_member']}"


def _stratum(b: float) -> tuple[int, int]:
    """Initial-success decile and learnability-band indicator."""
    return min(9, int(b * 10)), 1 if 0.0 < b < 1.0 else 0


def _dial() -> dict[str, Any]:
    """Measured mass m(f), band mass, and the stratified necessity of each realized stream."""
    if "dial" in _CACHE:
        return _CACHE["dial"]
    audit = _audit()
    necessity = {
        corpus: {item: b - q for item, (b, q) in items.items()} for corpus, items in audit.items()
    }
    constructed_mean = float(np.mean(list(necessity["constructed"].values())))
    mass: dict[str, float] = {}
    band: dict[str, float] = {}
    strata: dict[str, dict[tuple[int, int], list[float]]] = {}
    detail: dict[str, dict[str, int]] = {}
    for condition in CONDITIONS:
        rows = _stream(condition)
        fraction = float(Fraction(condition))
        drawn = sorted({key for corpus, key in map(_stream_key, rows) if corpus == "virl39k"})
        audited = [item for item in drawn if item in necessity["virl39k"]]
        if fraction < 1 and not audited:
            raise ValueError(f"f = {condition}: no drawn ViRL39K item is in the audit")
        virl_mean = float(np.mean([necessity["virl39k"][item] for item in audited])) if audited else 0.0
        if fraction == 1:
            mass[condition] = constructed_mean
        else:
            mass[condition] = fraction * constructed_mean + (1 - fraction) * virl_mean
        by: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in rows:
            corpus, key = _stream_key(row)
            if key in audit[corpus]:
                by[_stratum(audit[corpus][key][0])].append(necessity[corpus][key])
        joined = sum(len(values) for values in by.values())
        band[condition] = sum(len(values) for key, values in by.items() if key[1] == 1) / joined
        strata[condition] = dict(by)
        detail[condition] = {
            "virl39k_unique": len(drawn), "virl39k_audited": len(audited), "rows_joined": joined,
        }
    common = sorted(set.intersection(*(set(strata[condition]) for condition in CONDITIONS)))
    pooled = {key: sum(len(strata[condition][key]) for condition in CONDITIONS) for key in common}
    total = float(sum(pooled.values()))
    weights = {key: pooled[key] / total for key in common}
    standardized = {
        condition: sum(weights[key] * float(np.mean(strata[condition][key])) for key in common)
        for condition in CONDITIONS
    }
    _CACHE["dial"] = {
        "mass": mass,
        "standardized": standardized,
        "band": band,
        "strata": strata,
        "common": common,
        "weights": weights,
        "detail": detail,
    }
    return _CACHE["dial"]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope with an intercept."""
    centred = x - x.mean()
    return float((centred * (y - y.mean())).sum() / (centred * centred).sum())


def _slopes(endpoint: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Slopes of the step-30 endpoint on m(f) and on the standardized mass, with intervals from
    one bootstrap that resamples held-out items (paired across conditions) and, for the
    standardized mass, training rows within each condition and stratum."""
    dial = _dial()
    ids = sorted(endpoint["1"])
    if any(set(endpoint[condition]) != set(ids) for condition in CONDITIONS):
        raise ValueError("held-out items differ across conditions")
    outcomes = np.asarray([[endpoint[condition][i] for i in ids] for condition in CONDITIONS], dtype=float)
    raw_x = np.asarray([dial["mass"][condition] for condition in CONDITIONS], dtype=float)
    strata = {
        condition: {key: np.asarray(dial["strata"][condition][key], dtype=float) for key in dial["common"]}
        for condition in CONDITIONS
    }

    def standardized(rng: np.random.Generator) -> np.ndarray:
        values = []
        for condition in CONDITIONS:
            total = 0.0
            for key in dial["common"]:
                sample = strata[condition][key]
                sample = sample[rng.integers(0, len(sample), size=len(sample))]
                total += dial["weights"][key] * float(sample.mean())
            values.append(total)
        return np.asarray(values, dtype=float)

    levels = outcomes.mean(axis=1)
    std_x = np.asarray([dial["standardized"][condition] for condition in CONDITIONS], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    raw_draws, std_draws = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        index = rng.integers(0, len(ids), size=len(ids))
        sampled = outcomes[:, index].mean(axis=1)
        raw_draws.append(_slope(raw_x, sampled))
        std_draws.append(_slope(standardized(rng), sampled))
    def interval(draws: list[float]) -> list[float]:
        return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]

    return {
        "raw": (_slope(raw_x, levels), interval(raw_draws)),
        "standardized": (_slope(std_x, levels), interval(std_draws)),
        "n": len(ids),
    }


def _low(interval):
    return None if interval is None else interval[0]


def _high(interval):
    return None if interval is None else interval[1]


def _intervals_agree(rebuilt, printed, tolerance: float = INTERVAL_TOLERANCE) -> bool | None:
    if rebuilt is None or printed is None or printed[0] is None:
        return None
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(rebuilt, printed))


def _row(fields: Row, value, interval=None, printed=None, printed_interval=None, *,
         tolerance: float = PRINTED_TOLERANCE, interval_tolerance: float = INTERVAL_TOLERANCE,
         relation=None, status=None, successes=None, n=None, note=None) -> Row:
    """One output row: the rebuilt value and interval beside the printed ones."""
    printed_interval = printed_interval or (None, None)
    has_interval = printed_interval[0] is not None
    row = dict(fields)
    row.update(
        {
            "value": value,
            "ci_low": _low(interval),
            "ci_high": _high(interval),
            "paper_value": printed,
            "paper_ci_low": printed_interval[0],
            "paper_ci_high": printed_interval[1],
            "interval_agrees": _intervals_agree(interval, printed_interval, interval_tolerance),
            "tolerance": tolerance if printed is not None else None,
            "interval_tolerance": interval_tolerance if has_interval else None,
            "relation": relation,
            "status": status,
            "successes": successes,
            "n": n,
            "note": note,
        }
    )
    return row


# Table H.1: per condition, the audited mass, the standardized mass and the in-band share of
# the stream (not recorded for f = 1), then per density the discovery level and its gain over
# the untrained 7B model, then the grounding task.
PRINTED_H1: dict[str, dict[str, Any]] = {
    "0": {
        "mass": 0.126, "standardized": 0.094, "band": 0.752,
        "n8": (0.740, 0.080, 0.030, 0.135), "n12": (0.570, -0.005, -0.035, 0.030),
        "n20": (0.460, -0.010, -0.045, 0.025), "grounding": (0.797, 0.763, 0.828),
    },
    "1/3": {
        "mass": 0.165, "standardized": 0.229, "band": 0.900,
        "n8": (0.935, 0.275, 0.205, 0.350), "n12": (0.805, 0.230, 0.155, 0.310),
        "n20": (0.745, 0.275, 0.205, 0.350), "grounding": (0.908, 0.885, 0.930),
    },
    "2/3": {
        "mass": 0.199, "standardized": 0.240, "band": 0.926,
        "n8": (0.950, 0.290, 0.215, 0.365), "n12": (0.850, 0.275, 0.195, 0.355),
        "n20": (0.740, 0.270, 0.195, 0.345), "grounding": (0.927, 0.905, 0.947),
    },
    "1": {
        "mass": 0.249, "standardized": 0.251, "band": None,
        "n8": (0.965, 0.305, 0.235, 0.380), "n12": (0.890, 0.315, 0.230, 0.400),
        "n20": (0.800, 0.330, 0.250, 0.410), "grounding": (0.930, 0.908, 0.950),
    },
}
# With a gray canvas, discovery accuracy is 0 of 200 in every cell of every mixture.
PRINTED_GRAY_CORRECT = 0

# Table H.2: slopes per endpoint (on m(f), then on the standardized mass) and the
# equal-exposure comparisons.
PRINTED_SLOPES: dict[str, dict[str, tuple[float, float, float]]] = {
    "8-pt discovery": {"raw": (1.670, 1.182, 2.192), "standardized": (1.441, 0.990, 1.962)},
    "12-pt discovery": {"raw": (2.446, 1.807, 3.077), "standardized": (1.942, 1.395, 2.533)},
    "20-pt discovery": {"raw": (2.482, 1.886, 3.067), "standardized": (2.079, 1.541, 2.668)},
    "grounding task": {"raw": (1.009, 0.795, 1.230), "standardized": (0.861, 0.659, 1.086)},
}
ENDPOINT_DENSITY = {
    "8-pt discovery": "n8", "12-pt discovery": "n12", "20-pt discovery": "n20", "grounding task": None,
}
# (mixture, density) -> the mixture's level at step 30, the f = 1 run's level at the matched
# step, and their difference with its interval.
PRINTED_DISPLACEMENT: dict[tuple[str, str], tuple[float, ...]] = {
    ("1/3", "n8"): (0.935, 0.870, 0.065, 0.030, 0.105),
    ("1/3", "n12"): (0.805, 0.760, 0.045, 0.015, 0.075),
    ("1/3", "n20"): (0.745, 0.675, 0.070, 0.030, 0.120),
    ("2/3", "n8"): (0.950, 0.945, 0.005, 0.000, 0.015),
    ("2/3", "n12"): (0.850, 0.815, 0.035, -0.010, 0.085),
    ("2/3", "n20"): (0.740, 0.760, -0.020, -0.065, 0.020),
}
# The table's header rows: (constructed, total) prompt presentations of the mixture at step 30
# and of the f = 1 run at the matched step.
PRINTED_PRESENTATIONS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "1/3": ((2400, 7200), (2400, 2400)),
    "2/3": ((4800, 7200), (4800, 4800)),
}
# Appendix H: at equal exposure the one-third mixture is ahead on every density, and the
# two-thirds mixture is within 0.035 of the fully constructed run.
PRINTED_TWO_THIRDS_GAP = 0.035

# Table H.3: (model, step, density) -> discovery level and interval, change from the untrained
# model and interval, probe, discovery with a gray canvas. With a gray canvas the probe is
# exactly zero for both models at every checkpoint.
PRINTED_H3: dict[tuple[str, int, str], tuple] = {
    ("3B", 0, "n8"): (0.330, 0.265, 0.400, None, None, None, 0.705, 0.080),
    ("3B", 0, "n12"): (0.260, 0.201, 0.327, None, None, None, 0.680, 0.025),
    ("3B", 0, "n20"): (0.245, 0.187, 0.311, None, None, None, 0.630, 0.055),
    ("3B", 10, "n8"): (0.475, 0.404, 0.547, 0.145, 0.070, 0.220, 0.875, 0.080),
    ("3B", 10, "n12"): (0.425, 0.356, 0.497, 0.165, 0.090, 0.245, 0.755, 0.025),
    ("3B", 10, "n20"): (0.320, 0.256, 0.389, 0.075, -0.010, 0.160, 0.740, 0.055),
    ("3B", 20, "n8"): (0.750, 0.684, 0.808, 0.420, 0.330, 0.510, 0.960, 0.105),
    ("3B", 20, "n12"): (0.635, 0.564, 0.702, 0.375, 0.285, 0.465, 0.870, 0.030),
    ("3B", 20, "n20"): (0.465, 0.394, 0.537, 0.220, 0.120, 0.320, 0.880, 0.050),
    ("3B", 30, "n8"): (0.865, 0.810, 0.909, 0.535, 0.445, 0.620, 0.985, 0.060),
    ("3B", 30, "n12"): (0.730, 0.663, 0.790, 0.470, 0.375, 0.560, 0.945, 0.040),
    ("3B", 30, "n20"): (0.545, 0.473, 0.615, 0.300, 0.200, 0.400, 0.960, 0.070),
    ("7B", 0, "n8"): (0.660, 0.590, 0.725, None, None, None, 0.940, 0.000),
    ("7B", 0, "n12"): (0.575, 0.503, 0.644, None, None, None, 0.910, 0.000),
    ("7B", 0, "n20"): (0.470, 0.399, 0.542, None, None, None, 0.840, 0.000),
    ("7B", 10, "n8"): (0.870, 0.815, 0.913, 0.210, 0.150, 0.275, 0.985, 0.000),
    ("7B", 10, "n12"): (0.760, 0.695, 0.817, 0.185, 0.115, 0.260, 0.950, 0.000),
    ("7B", 10, "n20"): (0.675, 0.605, 0.739, 0.205, 0.135, 0.280, 0.940, 0.000),
    ("7B", 20, "n8"): (0.945, 0.904, 0.972, 0.285, 0.215, 0.360, 0.995, 0.000),
    ("7B", 20, "n12"): (0.815, 0.754, 0.866, 0.240, 0.165, 0.315, 0.960, 0.000),
    ("7B", 20, "n20"): (0.760, 0.695, 0.817, 0.290, 0.215, 0.370, 0.975, 0.000),
    ("7B", 30, "n8"): (0.965, 0.929, 0.986, 0.305, 0.235, 0.380, 1.000, 0.000),
    ("7B", 30, "n12"): (0.890, 0.838, 0.930, 0.315, 0.230, 0.400, 0.975, 0.000),
    ("7B", 30, "n20"): (0.800, 0.738, 0.853, 0.330, 0.250, 0.410, 0.975, 0.000),
}
# Grounding-task pair accuracy is printed at the untrained and final checkpoints only.
PRINTED_H3_GROUNDING: dict[tuple[str, int], tuple[float, float, float]] = {
    ("3B", 0): (0.455, 0.415, 0.495),
    ("3B", 30): (0.585, 0.547, 0.625),
    ("7B", 0): (0.768, 0.733, 0.802),
    ("7B", 30): (0.930, 0.908, 0.950),
}
PRINTED_PROBE_GRAY = 0.0
CAPACITY_STEPS = (0, 10, 20, 30)
MODELS = ("3B", "7B")

# Figure 5 and Section 7: the lettering of both panels and the values quoted beside them.
PRINTED_GAIN_SHARE = 0.83
MATCHED_STEPS = {"7B": 10, "3B": 30}
PRINTED_TRANSFER = {"3B": 0.130, "7B": 0.162}
TWO_DECIMAL_TOLERANCE = 5e-3 + 1e-9

# Figure 3(a), Section 5 and Appendix B.
PRINTED_SHARE = {"constructed": 0.929, "virl39k": 0.487}
PRINTED_LARGEST_CELL = 0.213
PRINTED_BLIND_OPPORTUNITY = 0.003
PRINTED_BLIND_CORRECT = 133
PRINTED_BLIND_ROLLOUTS = 46080
# The constructed training corpus: 720 scene programs of four prompts each, 960 per role.
PRINTED_PROGRAMS = 720
PRINTED_PROMPTS = 2880
PRINTED_PROMPTS_PER_ROLE = 960
ROLES = ("target_stable", "target_switch", "invariance")
CORPUS_LABEL = {"constructed": "constructed corpus", "virl39k": "ViRL39K"}


def _condition_cells(condition: str, step: int, density: str,
                     test_condition: str = "real") -> dict[str, Row]:
    family, run = CONDITION_RUN[condition]
    return _scenes(family, run, step, density, condition=test_condition)


def dose_response() -> list[Row]:
    """Table H.1: dose response at step 30 by constructed fraction f."""
    try:
        dial = _dial()
        missing = None
    except FileNotFoundError as error:
        dial, missing = None, str(error)
    base = {density: _scenes(BASELINES, UNTRAINED_7B, 0, density) for density in DENSITIES}
    rows: list[Row] = []
    for condition in CONDITIONS:
        printed = PRINTED_H1[condition]
        for column, quantity, key in (
            ("m(f)", "measured resolvability mass", "mass"),
            ("standardized m(f)", "standardized resolvability mass", "standardized"),
            ("in-band", "share of the stream in the learnability band", "band"),
        ):
            if printed[key] is None:
                continue
            value = None if dial is None else dial[key][condition]
            note = missing
            if dial is not None and key != "standardized":
                detail = dial["detail"][condition]
                note = f"{detail['rows_joined']} stream rows joined to the audit"
                if detail["virl39k_unique"]:
                    note += (f"; {detail['virl39k_audited']} of {detail['virl39k_unique']}"
                             " drawn ViRL39K items audited")
            rows.append(_row({"f": condition, "column": column, "quantity": quantity}, value,
                             printed=printed[key], status=INPUT_MISSING if missing else None,
                             note=note))
        for density in DENSITIES:
            cells = _condition_cells(condition, DOSE_STEP, density)
            successes, total, level = _member_accuracy(cells)
            level_printed, gain_printed, gain_low, gain_high = printed[density]
            column = DENSITY_LABEL[density] + (" (never trained)" if density == "n20" else "")
            rows.append(_row({"f": condition, "column": column, "quantity": "discovery level"}, level,
                             printed=level_printed, successes=successes, n=total))
            gain, interval, pairs = _member_delta(base[density], cells)
            rows.append(_row({"f": condition, "column": column, "quantity": "gain over untrained 7B"}, gain,
                             interval, gain_printed, (gain_low, gain_high), n=pairs))
        family, run = CONDITION_RUN[condition]
        grounding = _grounding(family, run, DOSE_STEP)
        successes, total, level, interval = _pair_accuracy(grounding)
        fields = {"f": condition, "column": "grounding task"}
        rows.append(_row({**fields, "quantity": "pair accuracy"}, level, interval, printed["grounding"][0],
                         printed["grounding"][1:], successes=successes, n=total))
    for condition in CONDITIONS:
        for density in DENSITIES:
            successes, total, _ = _member_accuracy(_condition_cells(condition, DOSE_STEP, density, "gray"))
            rows.append(_row({"f": condition, "column": DENSITY_LABEL[density],
                              "quantity": "discovery correct members, image removed"},
                             successes, printed=PRINTED_GRAY_CORRECT, relation="zero",
                             successes=successes, n=total))
    return rows


def _endpoints() -> dict[str, dict[str, dict[str, float]]]:
    """Per endpoint and condition, the step-30 per-pair outcome on the held-out items."""
    endpoints: dict[str, dict[str, dict[str, float]]] = {}
    for name, density in ENDPOINT_DENSITY.items():
        per_condition = {}
        for condition in CONDITIONS:
            if density is None:
                family, run = CONDITION_RUN[condition]
                cells = _grounding(family, run, DOSE_STEP)
                per_condition[condition] = {
                    pair_id: float(bool(row["pair_correct"])) for pair_id, row in cells.items()
                }
            else:
                cells = _condition_cells(condition, DOSE_STEP, density)
                per_condition[condition] = {
                    pair_id: sum(_members(row)) / 2.0 for pair_id, row in cells.items()
                }
        endpoints[name] = per_condition
    return endpoints


def _presentations(condition: str, step: int) -> tuple[int, int]:
    """(constructed, total) prompt presentations of one condition's stream up to `step`."""
    rows = _stream(condition)[: step * PROMPTS_PER_STEP]
    constructed = sum(1 for row in rows if _stream_key(row)[0] == "constructed")
    return constructed, len(rows)


def slopes_and_displacement() -> list[Row]:
    """Table H.2: slopes of step-30 accuracy on the audited and the standardized mass, and the
    equal-exposure comparisons of each mixture with the fully constructed run."""
    rows: list[Row] = []
    for name, per_condition in _endpoints().items():
        try:
            fitted, missing = _slopes(per_condition), None
        except FileNotFoundError as error:
            fitted, missing = None, str(error)
        for kind, label in (("raw", "slope on m(f)"), ("standardized", "slope on standardized m(f)")):
            estimate, interval = (None, None) if fitted is None else fitted[kind]
            printed = PRINTED_SLOPES[name][kind]
            rows.append(_row({"section": "slope", "endpoint": name, "quantity": label}, estimate,
                             interval, printed[0], printed[1:],
                             status=INPUT_MISSING if missing else None,
                             n=None if fitted is None else fitted["n"], note=missing))

    section = "equal exposure"
    differences: dict[str, list[float]] = defaultdict(list)
    for (condition, density), printed in PRINTED_DISPLACEMENT.items():
        reference_step = EXPOSURE_MATCH[condition]
        reference = _condition_cells("1", reference_step, density)
        mixture = _condition_cells(condition, DOSE_STEP, density)
        endpoint = DENSITY_LABEL[density] + " discovery"
        for quantity, cells, level_printed in (
            (f"f = {condition} at step {DOSE_STEP}", mixture, printed[0]),
            (f"f = 1 at step {reference_step}", reference, printed[1]),
        ):
            successes, total, level = _member_accuracy(cells)
            rows.append(_row({"section": section, "endpoint": endpoint, "quantity": quantity}, level,
                             printed=level_printed, successes=successes, n=total))
        estimate, interval, pairs = _member_delta(reference, mixture)
        differences[condition].append(estimate)
        quantity = f"f = {condition} at step {DOSE_STEP} minus f = 1 at step {reference_step}"
        rows.append(_row({"section": section, "endpoint": endpoint, "quantity": quantity}, estimate,
                         interval, printed[2], printed[3:], n=pairs))
    fields = {"section": section, "endpoint": "every density"}
    rows.append(_row({**fields, "quantity": "smallest difference, f = 1/3"},
                     min(differences["1/3"]), printed=0.0, relation="gt"))
    rows.append(_row({**fields, "quantity": "largest absolute difference, f = 2/3"},
                     max(abs(value) for value in differences["2/3"]),
                     printed=PRINTED_TWO_THIRDS_GAP, relation="le"))

    for condition, (mixture_printed, reference_printed) in PRINTED_PRESENTATIONS.items():
        reference_step = EXPOSURE_MATCH[condition]
        for side, stream, step, printed in (
            (f"f = {condition}", condition, DOSE_STEP, mixture_printed),
            ("f = 1", "1", reference_step, reference_printed),
        ):
            try:
                counts, missing = _presentations(stream, step), None
            except FileNotFoundError as error:
                counts, missing = (None, None), str(error)
            for kind, value, count_printed in zip(("constructed", "total"), counts, printed):
                rows.append(_row(
                    {"section": section, "endpoint": f"f = {condition} against f = 1",
                     "quantity": f"{kind} prompt presentations, {side} at step {step}"},
                    value, printed=count_printed, tolerance=0,
                    status=INPUT_MISSING if missing else None, note=missing,
                ))
    return rows


def capacity() -> list[Row]:
    """Table H.3: the 3B and 7B models trained on the same constructed corpus, per checkpoint."""
    rows: list[Row] = []
    for model in MODELS:
        untrained = {density: _scenes(*_source(model, 0), density) for density in DENSITIES}
        for step in CAPACITY_STEPS:
            family, run, read_step = _source(model, step)
            for density in DENSITIES:
                printed = PRINTED_H3[(model, step, density)]
                fields = {"model": model, "step": step, "density": DENSITY_LABEL[density]}
                cells = untrained[density] if step == 0 else _scenes(family, run, read_step, density)
                successes, total, level = _member_accuracy(cells)
                rows.append(_row({**fields, "quantity": "discovery"}, level,
                                 clopper_pearson_interval(successes, total), printed[0], printed[1:3],
                                 successes=successes, n=total))
                if step > 0:
                    delta, interval, pairs = _member_delta(untrained[density], cells)
                    rows.append(_row({**fields, "quantity": "change from untrained"}, delta, interval,
                                     printed[3], printed[4:6], n=pairs))
                probe = _scenes(family, run, read_step, density, layer="probe")
                successes, total, level = _member_accuracy(probe)
                rows.append(_row({**fields, "quantity": "probe"}, level, printed=printed[6],
                                 successes=successes, n=total))
                probe = _scenes(family, run, read_step, density, layer="probe", condition="gray")
                successes, total, level = _member_accuracy(probe)
                rows.append(_row({**fields, "quantity": "probe, gray canvas"}, level,
                                 printed=PRINTED_PROBE_GRAY, relation="zero", successes=successes,
                                 n=total))
                removed = _scenes(family, run, read_step, density, condition="gray")
                successes, total, level = _member_accuracy(removed)
                rows.append(_row({**fields, "quantity": "discovery, gray canvas"}, level,
                                 printed=printed[7], successes=successes, n=total))
            if (model, step) in PRINTED_H3_GROUNDING:
                printed = PRINTED_H3_GROUNDING[(model, step)]
                successes, total, level, interval = _pair_accuracy(_grounding(family, run, read_step))
                fields = {"model": model, "step": step, "density": None, "quantity": "grounding task"}
                rows.append(_row(fields, level, interval, printed[0], printed[1:], successes=successes,
                                 n=total))
    return rows


def dial_figure() -> list[Row]:
    """Figure 5 and the values Section 7 quotes from it: (a) step-30 discovery gain against
    measured mass, (b) discovery trajectories of the 3B and 7B models."""
    try:
        mass = _dial()["mass"]
        missing = None
    except FileNotFoundError as error:
        mass, missing = None, str(error)
    base = {density: _scenes(BASELINES, UNTRAINED_7B, 0, density) for density in DENSITIES}
    rows: list[Row] = []
    gains: dict[tuple[str, str], float] = {}
    for condition in CONDITIONS:
        rows.append(_row({"panel": "a", "series": "all densities", "f": condition, "step": DOSE_STEP,
                          "density": None, "quantity": "m(f)"},
                         None if mass is None else mass[condition], printed=PRINTED_H1[condition]["mass"],
                         status=INPUT_MISSING if missing else None, note=missing))
        for density in DENSITIES:
            cells = _condition_cells(condition, DOSE_STEP, density)
            gain, interval, pairs = _member_delta(base[density], cells)
            gains[(condition, density)] = gain
            printed = PRINTED_H1[condition][density]
            fields = {"panel": "a", "series": DENSITY_LABEL[density], "f": condition, "step": DOSE_STEP,
                      "density": density, "quantity": "discovery gain"}
            rows.append(_row(fields, gain, interval, printed[1], printed[2:], n=pairs))
    share = gains[("1/3", "n20")] / gains[("1", "n20")]
    rows.append(_row({"panel": "a", "series": "20-pt", "f": "1/3", "step": DOSE_STEP, "density": "n20",
                      "quantity": "share of the f = 1 gain"},
                     share, printed=PRINTED_GAIN_SHARE, tolerance=TWO_DECIMAL_TOLERANCE))
    for model in MODELS:
        for step in CAPACITY_STEPS:
            family, run, read_step = _source(model, step)
            for density in DENSITIES:
                cells = _scenes(family, run, read_step, density)
                successes, total, level = _member_accuracy(cells)
                printed = PRINTED_H3[(model, step, density)]
                series = f"{model} step {step}" if MATCHED_STEPS[model] == step else model
                fields = {"panel": "b", "series": series, "f": "1", "step": step, "density": density,
                          "quantity": "discovery accuracy"}
                rows.append(_row(fields, level, clopper_pearson_interval(successes, total), printed[0],
                                 printed[1:3], successes=successes, n=total))
        before = _pair_accuracy(_grounding(*_source(model, 0)))[2]
        after = _pair_accuracy(_grounding(*_source(model, 30)))[2]
        rows.append(_row({"panel": "b", "series": model, "f": "1", "step": 30, "density": None,
                          "quantity": "grounding-task pair-accuracy gain"},
                         after - before, printed=PRINTED_TRANSFER[model]))
    return rows


def audit_plane() -> list[Row]:
    """Figure 3(a): the base-model audit of the constructed corpus and ViRL39K on the plane of
    visual necessity against initial success, one row per occupied lattice cell, and the shares."""
    audit = _audit()
    rows: list[Row] = []
    largest = 0.0
    for corpus in ("constructed", "virl39k"):
        items = audit[corpus]
        cells: dict[tuple[float, float], int] = defaultdict(int)
        for b, q_blind in items.values():
            cells[(round(b / LATTICE_STEP) * LATTICE_STEP,
                   round((b - q_blind) / LATTICE_STEP) * LATTICE_STEP)] += 1
        largest = max(largest, max(cells.values()) / len(items))
        for (b, necessity), count in sorted(cells.items()):
            rows.append(_row({"corpus": CORPUS_LABEL[corpus], "quantity": "item share", "b": b,
                              "delta_q": necessity}, count / len(items), successes=count, n=len(items)))
        resolvable = sum(1 for b, q in items.values() if _necessary_and_learnable(b, q))
        rows.append(_row({"corpus": CORPUS_LABEL[corpus], "quantity": "share necessary and learnable",
                          "b": None, "delta_q": None},
                         resolvable / len(items), printed=PRINTED_SHARE[corpus], successes=resolvable,
                         n=len(items)))
    rows.append(_row({"corpus": "both", "quantity": "largest cell share", "b": None, "delta_q": None},
                     largest, printed=PRINTED_LARGEST_CELL))
    return rows


def audit_summary() -> list[Row]:
    """Section 5 and Appendix B: the constructed corpus's image-free rollouts and blind reward
    opportunity, the necessary-and-learnable share of both audited corpora, the corpus-level
    audit means, and the composition of the constructed training corpus."""
    audit = _audit()
    records = _audit_records()
    rows: list[Row] = []
    for corpus in ("constructed", "virl39k"):
        items = audit[corpus]
        n = len(items)
        blind = records[(corpus, BLIND_CONDITION)].values()
        blind_correct = sum(int(row[AUDIT_CORRECT]) for row in blind)
        blind_rollouts = sum(int(row["samples"]) for row in blind)
        resolvable = sum(1 for b, q in items.values() if _necessary_and_learnable(b, q))
        constructed = corpus == "constructed"
        label = CORPUS_LABEL[corpus]
        rows.append(_row({"corpus": label, "quantity": "audited items"}, n,
                         printed=CONSTRUCTED_ROWS if constructed else None, tolerance=0))
        rows.append(_row({"corpus": label, "quantity": "image-free rollouts"}, blind_rollouts,
                         printed=PRINTED_BLIND_ROLLOUTS if constructed else None, tolerance=0))
        rows.append(_row({"corpus": label, "quantity": "correct image-free rollouts"}, blind_correct,
                         printed=PRINTED_BLIND_CORRECT if constructed else None, tolerance=0,
                         successes=blind_correct, n=blind_rollouts))
        rows.append(_row({"corpus": label, "quantity": "blind reward opportunity"},
                         float(np.mean([q for _, q in items.values()])),
                         printed=PRINTED_BLIND_OPPORTUNITY if constructed else None))
        rows.append(_row({"corpus": label, "quantity": "mean initial success b"},
                         float(np.mean([b for b, _ in items.values()]))))
        rows.append(_row({"corpus": label, "quantity": "mean visual necessity"},
                         float(np.mean([b - q for b, q in items.values()]))))
        rows.append(_row({"corpus": label, "quantity": "share in the learnability band"},
                         sum(1 for b, _ in items.values() if 0 < b < 1) / n))
        rows.append(_row({"corpus": label, "quantity": "share necessary and learnable"}, resolvable / n,
                         printed=PRINTED_SHARE[corpus], successes=resolvable, n=n,
                         note=None if constructed else "printed in Figure 3(a)"))
    return rows + _corpus_composition()


def _corpus_composition() -> list[Row]:
    """Scene programs, prompts and prompts per role of the constructed training corpus."""
    label = CORPUS_LABEL["constructed"]
    try:
        text, missing = _stream_text(STREAM_DIR["1"]), None
    except FileNotFoundError as error:
        text, missing = None, str(error)
    rows = [] if text is None else [json.loads(line) for line in text.splitlines() if line.strip()]
    programs = {str(row["pair_group_uid"]) for row in rows}
    # The role is part of each program's identifier.
    per_role = Counter(
        next((role for role in ROLES if role in str(row["pair_group_uid"])), None) for row in rows
    )
    status = INPUT_MISSING if missing else None
    counts = [("scene programs", len(programs), PRINTED_PROGRAMS),
              ("training prompts", len(rows), PRINTED_PROMPTS)]
    counts += [(f"training prompts, {role} programs", per_role[role], PRINTED_PROMPTS_PER_ROLE)
               for role in ROLES]
    return [
        _row({"corpus": label, "quantity": quantity}, None if missing else value, printed=printed,
             tolerance=0, status=status, note=missing)
        for quantity, value, printed in counts
    ]


TARGETS = {
    "tableH1": dose_response,
    "tableH2": slopes_and_displacement,
    "tableH3": capacity,
    "figure5": dial_figure,
    "figure3a": audit_plane,
    "section5": audit_summary,
}

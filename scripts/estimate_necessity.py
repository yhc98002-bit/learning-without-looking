#!/usr/bin/env python3
"""How well a k-rollout subsample of the visual-necessity audit reproduces the full audit's per-item
necessity ranking, per corpus and per k. Correctness is read from the recorded samples, not re-scored.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.analysis.resample import (  # noqa: E402
    deterministic_seed,
    mean_with_paired_bootstrap,
    tied_spearman,
)
from lwl.paths import repo_root  # noqa: E402

ROOT = repo_root()

SCHEMA_VERSION = "lwl.necessity-estimator.v1"

K_GRID = (1, 2, 4, 8, 16)
FULL_ROLLOUTS = 16
REPLICATES = 20
SUBSAMPLE_SEED = 20260823
RHO_MIN = 0.85
TOPQ_MIN = 0.70
QBLIND_ABS_MAX = 0.02
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260716

ACCEPTED_SCHEMAS = ("lwl.blind-solvability-pilot.v1",)
REAL_CONDITION = "real"
BLIND_CONDITIONS = ("none", "gray", "noise", "caption")
JOIN_KEYS = ("auto", "qid", "row_index", "split_row_index")
SUB_K = tuple(k for k in K_GRID if k < FULL_ROLLOUTS)   # the k values that need a random subsample
MIN_ITEMS = 8
MIN_STRATUM_ITEMS = 25
DIGITS = 6                     # rank statistics need more precision than accuracy readouts
# Scoring settings that the with-image and image-removed pass of one corpus must share.
LOCK_FIELDS = ("schema_version", "scoring_mode", "parser_version", "pilot_reward_version",
               "prompt_contract_sha256", "symbolic_grader_guard_version", "sample_count")

BLIND_CONDITION_NOTE = (
    "Where a corpus's image-removed condition is `gray`, that is the base model measured on gray "
    "inputs inside this audit. It is not a trained model evaluated on gray inputs, and not a "
    "gray-trained control arm."
)
NO_CROSS_CORPUS_NOTE = (
    "No statistic in this file is averaged across corpora; each corpus carries its own numbers, and "
    "strata are reported inside their corpus only, never pooled."
)
NESTING_NOTE = (
    "Δq̂_k is drawn from inside the same 16 rollouts that define Δq_16, so ρ(Δq̂_k, Δq_16) is optimistic "
    "relative to k fresh rollouts; the disjoint-halves companion ρ(Δq̂_k^A, Δq̂_k^B) carries no nesting "
    "and is reported beside it at every k ≤ 8."
)
COST_NOTE = (
    "Cost is rollouts, not wall-clock: 2k rollouts per item (with image + image removed) against the "
    "audit's 32; no other cost model is asserted here."
)


class EstimatorError(ValueError):
    """Input, schema, join or discipline error that stops the build."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    """Exact binomial interval (scipy.stats.beta)."""
    from scipy.stats import beta

    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"invalid counts k={k} n={n}")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(lo, 6), round(hi, 6)]


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - environment dependent
        return None


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _num(value: Any) -> Any:
    """JSON-safe float (numpy scalars become python floats; NaN never reaches the file)."""
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        raise EstimatorError(f"non-finite statistic: {value!r}")
    return round(out, DIGITS)


def _summary(values: list[float | None]) -> dict[str, Any]:
    """mean / sd / min / max over replicates; None replicates are counted, never imputed."""
    good = [float(v) for v in values if v is not None]
    n_null = len(values) - len(good)
    if not good:
        return {"mean": None, "sd": None, "min": None, "max": None, "n_replicates": len(values),
                "null_replicates": n_null, "per_replicate": [None] * len(values)}
    arr = np.asarray(good, dtype=np.float64)
    return {
        "mean": _num(arr.mean()),
        "sd": _num(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": _num(arr.min()),
        "max": _num(arr.max()),
        "n_replicates": len(values),
        "null_replicates": n_null,
        "per_replicate": [None if v is None else _num(v) for v in values],
    }


def resolve_run_dirs(pattern: str, label: str) -> list[Path]:
    """Run directories matching `pattern`, which is absolute or relative to the repository."""
    if Path(pattern).is_absolute():
        matches = sorted(Path(match) for match in glob.glob(pattern, recursive=True))
    else:
        matches = sorted(ROOT.glob(pattern))
    hits: list[Path] = []
    for match in matches:
        cand = match.parent if match.name == "per_item.jsonl" else match
        if (cand / "per_item.jsonl").is_file() and cand not in hits:
            hits.append(cand)
    if not hits:
        raise EstimatorError(f"{label}: no run directory with per_item.jsonl matched {pattern}")
    return hits


def load_condition(pattern: str, condition: str, label: str) -> dict[str, Any]:
    """Rollout-level records for one condition; every consumed file sha256-pinned."""
    records: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    run_dirs: list[str] = []
    locks: dict[str, set] = {field: set() for field in LOCK_FIELDS}
    for run_dir in resolve_run_dirs(pattern, label):
        per_item = run_dir / "per_item.jsonl"
        inputs[rel(per_item)] = sha256_file(per_item)
        manifest = run_dir / "run_manifest.json"
        if manifest.is_file():
            inputs[rel(manifest)] = sha256_file(manifest)
        run_dirs.append(rel(run_dir))
        with per_item.open() as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                where = f"{label} {rel(per_item)}:{lineno}"
                schema = str(row.get("schema_version"))
                if schema not in ACCEPTED_SCHEMAS:
                    raise EstimatorError(f"{where}: unexpected schema_version {schema!r} (accepted {ACCEPTED_SCHEMAS})")
                if str(row.get("condition")) != condition:
                    raise EstimatorError(f"{where}: condition is {row.get('condition')!r}, expected {condition!r}")
                n = int(row.get("sample_count", -1))
                if n != FULL_ROLLOUTS:
                    raise EstimatorError(f"{where}: sample_count {n} != {FULL_ROLLOUTS}; this build subsamples the "
                                         f"16-rollout audit only")
                correct = row.get("sample_correct")
                if not isinstance(correct, list) or len(correct) != n:
                    raise EstimatorError(
                        f"{where}: no rollout-level `sample_correct` array of length {n}. This artifact stores "
                        f"per-item aggregates only; a binomial draw from a stored count is not a rollout "
                        f"subsample and would misreport the estimator's error.")
                bits = np.asarray([bool(c) for c in correct], dtype=bool)
                count = int(row.get("sample_correct_count", -1))
                if count != int(bits.sum()):
                    raise EstimatorError(f"{where}: sample_correct_count {count} != sum(sample_correct) {int(bits.sum())}")
                p_sample = float(row.get("p_sample", float("nan")))
                if not math.isfinite(p_sample) or abs(p_sample - count / n) > 1e-9:
                    raise EstimatorError(f"{where}: p_sample {p_sample} != sample_correct_count/sample_count")
                for field in LOCK_FIELDS:
                    locks[field].add(json.dumps(row.get(field), sort_keys=True))
                split = row.get("split")
                index = row.get("row_index")
                records.append({
                    "qid": row.get("qid"),
                    "row_index": index,
                    # The geometry3k audit restarts row_index inside each split (1,288 train + 601
                    # test rows share indices 0...), so the composite key is the join that survives there.
                    "split_row_index": None if (split is None or index is None) else f"{split}#{index}",
                    "correct": bits,
                    "count": count,
                    "metadata": row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else None,
                })
    return {"condition": condition, "records": records, "inputs": inputs, "run_dirs": run_dirs,
            "locks": {field: sorted(values) for field, values in locks.items()}}


def choose_join_key(real: dict, blind: dict, requested: str, label: str) -> str:
    def usable(side: dict, key: str) -> bool:
        values = [rec[key] for rec in side["records"]]
        return all(v is not None for v in values) and len(set(map(str, values))) == len(values)

    if requested != "auto":
        for side, name in ((real, "real"), (blind, "blind")):
            if not usable(side, requested):
                raise EstimatorError(f"{label}: --join-key {requested} is not unique/non-null on the {name} pass")
        return requested
    for key in ("qid", "split_row_index", "row_index"):
        if usable(real, key) and usable(blind, key):
            return key
    raise EstimatorError(f"{label}: no join key is unique and non-null on both passes "
                         f"(tried qid, split_row_index, row_index)")


def join_corpus(label: str, real: dict, blind: dict, join_key: str, stratum_key: str | None) -> dict[str, Any]:
    for field in LOCK_FIELDS:
        if real["locks"][field] != blind["locks"][field] or len(real["locks"][field]) != 1:
            raise EstimatorError(
                f"{label}: the real and blind passes disagree on the locked field {field!r} "
                f"(real={real['locks'][field]}, blind={blind['locks'][field]}); Δq would mix scoring settings")
    by_key: dict[Any, dict[str, Any]] = {}
    for side_name, side in (("real", real), ("blind", blind)):
        for rec in side["records"]:
            key = rec[join_key]
            slot = by_key.setdefault(key, {})
            if side_name in slot:
                raise EstimatorError(f"{label}: duplicate {join_key} {key!r} on the {side_name} pass")
            slot[side_name] = rec
    only_real = sorted(str(k) for k, v in by_key.items() if "blind" not in v)
    only_blind = sorted(str(k) for k, v in by_key.items() if "real" not in v)
    if only_real or only_blind:
        raise EstimatorError(f"{label}: the two passes do not carry the same {join_key} set "
                             f"(only-real={only_real[:3]} n={len(only_real)}; only-blind={only_blind[:3]} n={len(only_blind)})")
    if len(by_key) < MIN_ITEMS:
        raise EstimatorError(f"{label}: {len(by_key)} joined items is below the {MIN_ITEMS}-item floor")

    keys = sorted(by_key)
    strata: dict[Any, str] | None = None
    if stratum_key is not None:
        strata = {}
        for key in keys:
            meta = by_key[key]["real"]["metadata"] or {}
            if stratum_key not in meta:
                raise EstimatorError(f"{label}: --stratify key {stratum_key!r} missing from source_metadata of item {key!r}")
            strata[key] = str(meta[stratum_key])
    real_mat = np.stack([by_key[k]["real"]["correct"] for k in keys])
    blind_mat = np.stack([by_key[k]["blind"]["correct"] for k in keys])
    return {
        "label": label, "join_key": join_key, "keys": keys, "n_items": len(keys),
        "real_matrix": real_mat, "blind_matrix": blind_mat,
        "dq_full": real_mat.mean(axis=1) - blind_mat.mean(axis=1),
        "q_real_full": float(real_mat.mean()), "q_blind_full": float(blind_mat.mean()),
        "blind_condition": blind["condition"], "stratum_key": stratum_key, "strata": strata,
        "run_dirs": {"real": real["run_dirs"], "blind": blind["run_dirs"]},
        "locks": {field: json.loads(real["locks"][field][0]) for field in LOCK_FIELDS},
    }


def permutation_schedule(label: str, condition: str, item_key: Any) -> np.ndarray:
    """(len(SUB_K) × REPLICATES, 16) rollout permutations for ONE item, from ONE per-item stream.

    The stream is seeded by (corpus label, condition, item key) alone - never by the item's
    position, the corpus's size, or the number of replicates actually requested - so the draw for
    a given (k, replicate) is fixed once and for all, and a mixture or a stratum inherits exactly
    the draws the whole corpus would have given that item.
    """
    seed = deterministic_seed(SUBSAMPLE_SEED, f"{label}|{condition}|{item_key}")
    base = np.tile(np.arange(FULL_ROLLOUTS, dtype=np.int8), (len(SUB_K) * REPLICATES, 1))
    return np.random.default_rng(seed).permuted(base, axis=1)


def schedule_row(k: int, replicate: int) -> int:
    if k not in SUB_K:
        raise EstimatorError(f"no subsample schedule for k={k} (k = {FULL_ROLLOUTS} is the full record set)")
    if not 1 <= replicate <= REPLICATES:
        raise EstimatorError(f"replicate {replicate} outside the pinned 1..{REPLICATES} schedule")
    return SUB_K.index(k) * REPLICATES + (replicate - 1)


def subsample_indices(label: str, condition: str, k: int, replicate: int, item_key: Any) -> np.ndarray:
    """The permutation of the 16 rollout slots this item gets at (k, replicate); first k = the subsample."""
    return permutation_schedule(label, condition, item_key)[schedule_row(k, replicate)]


def corpus_schedule(corpus: dict, side: str) -> np.ndarray:
    condition = REAL_CONDITION if side == "real" else corpus["blind_condition"]
    return np.stack([permutation_schedule(corpus["label"], condition, key) for key in corpus["keys"]])


def subsample_means(corpus: dict, side: str, k: int, replicate: int,
                    schedule: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-item success rate over a k-subsample (and over a DISJOINT second k-subsample when 2k ≤ 16)."""
    matrix = corpus["real_matrix"] if side == "real" else corpus["blind_matrix"]
    if k == FULL_ROLLOUTS:
        return matrix.mean(axis=1), None
    if schedule is None:
        schedule = corpus_schedule(corpus, side)
    perm = schedule[:, schedule_row(k, replicate), :].astype(np.intp)
    first = np.take_along_axis(matrix, perm[:, :k], axis=1).mean(axis=1)
    second = (np.take_along_axis(matrix, perm[:, k:2 * k], axis=1).mean(axis=1)
              if 2 * k <= FULL_ROLLOUTS else None)
    return first, second


def _ordered(values: np.ndarray, keys: list, *, top: bool) -> list[int]:
    sign = -1.0 if top else 1.0
    return sorted(range(len(keys)), key=lambda i: (sign * float(values[i]), str(keys[i])))


def reference_sets(values: np.ndarray, keys: list) -> dict[str, Any]:
    m = math.ceil(len(keys) / 4)
    top = _ordered(values, keys, top=True)[:m]
    bottom = _ordered(values, keys, top=False)[:m]
    return {
        "m": m,
        "top": set(top), "bottom": set(bottom),
        "top_boundary_tie_mass": _num(float(np.sum(values == values[top[-1]])) / len(keys)),
        "bottom_boundary_tie_mass": _num(float(np.sum(values == values[bottom[-1]])) / len(keys)),
    }


def retrieval_deterministic(values: np.ndarray, keys: list, reference: set[int], m: int, *, top: bool) -> float:
    picked = _ordered(values, keys, top=top)[:m]
    return len(set(picked) & reference) / m


def retrieval_expected_random_tiebreak(values: np.ndarray, reference: set[int], m: int, *, top: bool) -> tuple[float, float]:
    """Expected |picked ∩ reference| / m when boundary ties are broken uniformly at random.

    Returns (expected retrieval, boundary tie mass of this ranking). At small k the estimate
    takes few distinct values; scoring an arbitrary tie order as skill is the bug this avoids.
    """
    order = np.argsort(-values if top else values, kind="stable")
    boundary = float(values[order[m - 1]])
    strict = np.where(values > boundary)[0] if top else np.where(values < boundary)[0]
    tied = np.where(values == boundary)[0]
    slots = m - len(strict)
    if slots < 0 or slots > len(tied):
        raise EstimatorError("retrieval: boundary partition is inconsistent")
    hit = len(set(strict.tolist()) & reference)
    if tied.size:
        hit += (slots / len(tied)) * len(set(tied.tolist()) & reference)
    return hit / m, float(len(tied)) / len(values)


def _index_subset(corpus: dict, member: set[str] | None) -> list[int]:
    if member is None:
        return list(range(corpus["n_items"]))
    return [i for i, key in enumerate(corpus["keys"]) if corpus["strata"][key] in member]


def block_metrics(dq_full: np.ndarray, dq_hat: list[np.ndarray], dq_hat_b: list[np.ndarray | None],
                  keys: list) -> dict[str, Any]:
    """Rank stability + quartile retrieval for one (corpus | stratum) × k, over all replicates.

    Retrieval is reported three ways: the expected value under uniform random tie-breaking, the
    deterministic join-key tie-break beside it, and the ratio of the first to the CEILING - the
    expected retrieval of a perfect estimator, i.e. of Δq_16 itself scored under the same random
    tie-breaking. The ceiling is below 1 whenever Δq_16 ties across the quartile boundary (16
    rollouts give at most 17 distinct values), so an absolute threshold would be unreachable by
    construction; the ratio is the comparable number.
    """
    ref = reference_sets(dq_full, keys)
    ceil_top, _ = retrieval_expected_random_tiebreak(dq_full, ref["top"], ref["m"], top=True)
    ceil_bottom, _ = retrieval_expected_random_tiebreak(dq_full, ref["bottom"], ref["m"], top=False)
    rho, rho_disjoint = [], []
    top_exp, top_det, bot_exp, bot_det, tie_top, tie_bot = [], [], [], [], [], []
    for rep, hat in enumerate(dq_hat):
        rho.append(tied_spearman(hat, dq_full))
        other = dq_hat_b[rep]
        rho_disjoint.append(None if other is None else tied_spearman(hat, other))
        te, tt = retrieval_expected_random_tiebreak(hat, ref["top"], ref["m"], top=True)
        be, bt = retrieval_expected_random_tiebreak(hat, ref["bottom"], ref["m"], top=False)
        top_exp.append(te)
        bot_exp.append(be)
        tie_top.append(tt)
        tie_bot.append(bt)
        top_det.append(retrieval_deterministic(hat, keys, ref["top"], ref["m"], top=True))
        bot_det.append(retrieval_deterministic(hat, keys, ref["bottom"], ref["m"], top=False))
    disjoint_available = any(v is not None for v in rho_disjoint)
    if ceil_top <= 0 or ceil_bottom <= 0:
        raise EstimatorError("retrieval ceiling is zero: the reference quartile is degenerate")
    return {
        "n_items": len(keys), "quartile_m": ref["m"],
        "spearman_vs_full_audit": _summary(rho),
        "spearman_disjoint_halves": _summary(rho_disjoint) if disjoint_available else {
            "available": False, "note": "2k > 16: no disjoint second k-subsample exists"},
        "retrieval_top_quartile": {
            "expected_random_tiebreak": _summary(top_exp),
            "deterministic_tiebreak": _summary(top_det),
            "ceiling_expected_random_tiebreak": _num(ceil_top),
            "ratio_to_ceiling": _summary([v / ceil_top for v in top_exp]),
        },
        "retrieval_bottom_quartile": {
            "expected_random_tiebreak": _summary(bot_exp),
            "deterministic_tiebreak": _summary(bot_det),
            "ceiling_expected_random_tiebreak": _num(ceil_bottom),
            "ratio_to_ceiling": _summary([v / ceil_bottom for v in bot_exp]),
        },
        "reference_boundary_tie_mass": {"top": ref["top_boundary_tie_mass"], "bottom": ref["bottom_boundary_tie_mass"]},
        "estimator_boundary_tie_mass": {"top": _summary(tie_top)["mean"], "bottom": _summary(tie_bot)["mean"]},
    }


def q_blind_block(corpus: dict, blind_hat: list[np.ndarray], k: int, *, draws: int, seed: int) -> dict[str, Any]:
    """Corpus-level q_blind error, paired per item on replicate 1."""
    per_item_full = corpus["blind_matrix"].mean(axis=1)
    full_mean = float(per_item_full.mean())
    errors = [float(hat.mean()) - full_mean for hat in blind_hat]
    contributions = (blind_hat[0] - per_item_full).astype(np.float64)
    n_items = corpus["n_items"]
    khat = int(round(float(blind_hat[0].sum()) * k))
    kfull = int(corpus["blind_matrix"].sum())
    block: dict[str, Any] = {
        "q_blind_full_audit": _num(full_mean),
        "q_blind_hat": _summary([float(hat.mean()) for hat in blind_hat]),
        "signed_error": _summary(errors),
        "abs_error_mean": _num(float(np.mean([abs(e) for e in errors]))),
        "counts": {"subsample": {"k": khat, "n": n_items * k}, "full_audit": {"k": kfull, "n": n_items * FULL_ROLLOUTS}},
    }
    if len(set(contributions.tolist())) <= 1:
        block["paired_bootstrap_replicate1"] = {
            "zero_variance": True, "ci95": None,
            "method": "exact counts; Clopper–Pearson 95% per pass; no bootstrap, no paired test "
                      "(paired contributions identical)",
            "subsample_clopper_pearson_95": clopper_pearson(khat, n_items * k),
            "full_audit_clopper_pearson_95": clopper_pearson(kfull, n_items * FULL_ROLLOUTS),
        }
    else:
        boot = mean_with_paired_bootstrap(contributions, draws=draws, seed=seed)
        block["paired_bootstrap_replicate1"] = {
            "zero_variance": False, "estimate": _num(boot["estimate"]),
            "ci95": [_num(boot["ci95"][0]), _num(boot["ci95"][1])],
            "paired_se": _num(boot["paired_se"]), "bootstrap_draws": boot["bootstrap_draws"], "seed": seed,
            "method": "paired per-item bootstrap over items (mean_with_paired_bootstrap)",
        }
    return block


def corpus_metrics(corpus: dict, *, replicates: int, draws: int, seed: int) -> dict[str, Any]:
    dq_full = corpus["dq_full"]
    keys = corpus["keys"]
    schedules = {side: corpus_schedule(corpus, side) for side in ("real", "blind")}
    per_k: dict[str, Any] = {}
    for k in K_GRID:
        reps = 1 if k == FULL_ROLLOUTS else replicates
        dq_hat: list[np.ndarray] = []
        dq_hat_b: list[np.ndarray | None] = []
        blind_hat: list[np.ndarray] = []
        for rep in range(1, reps + 1):
            real_a, real_b = subsample_means(corpus, "real", k, rep, schedules["real"])
            blind_a, blind_b = subsample_means(corpus, "blind", k, rep, schedules["blind"])
            dq_hat.append(real_a - blind_a)
            dq_hat_b.append(None if (real_b is None or blind_b is None) else real_b - blind_b)
            blind_hat.append(blind_a)
        if k == FULL_ROLLOUTS:
            worst = float(np.max(np.abs(dq_hat[0] - dq_full)))
            if worst != 0.0:
                raise EstimatorError(f"{corpus['label']}: k=16 must reproduce Δq_16 exactly, max |Δ| = {worst}")
        block = block_metrics(dq_full, dq_hat, dq_hat_b, keys)
        block.update({
            "k": k, "replicates": reps,
            "cost": {"rollouts_per_item": 2 * k, "rollouts_corpus": 2 * k * corpus["n_items"],
                     "fraction_of_full_audit": _num(k / FULL_ROLLOUTS),
                     "rollout_saving_factor": _num(FULL_ROLLOUTS / k)},
            "q_blind": q_blind_block(corpus, blind_hat, k, draws=draws, seed=seed),
        })
        if corpus["strata"] is not None:
            block["strata"] = {}
            for name in sorted(set(corpus["strata"].values())):
                idx = _index_subset(corpus, {name})
                if len(idx) < MIN_STRATUM_ITEMS:
                    block["strata"][name] = {"skipped": True, "n_items": len(idx),
                                             "note": f"below the {MIN_STRATUM_ITEMS}-item floor; not reported"}
                    continue
                sub_keys = [keys[i] for i in idx]
                block["strata"][name] = block_metrics(dq_full[idx], [h[idx] for h in dq_hat],
                                                      [None if h is None else h[idx] for h in dq_hat_b], sub_keys)
        per_k[str(k)] = block
    return {
        "label": corpus["label"], "n_items": corpus["n_items"], "join_key": corpus["join_key"],
        "blind_condition": corpus["blind_condition"], "stratum_key": corpus["stratum_key"],
        "q_real_full_audit": _num(corpus["q_real_full"]), "q_blind_full_audit": _num(corpus["q_blind_full"]),
        "delta_q_full_audit_mean": _num(float(dq_full.mean())),
        "delta_q_full_audit_positive_items": int(np.sum(dq_full > 0)),
        "locks": corpus["locks"], "run_dirs": corpus["run_dirs"],
        "per_k": per_k,
    }


def qualification_table(results: dict[str, Any]) -> dict[str, Any]:
    """Per k and corpus: the statistics compared with the pinned thresholds."""
    table: dict[str, Any] = {}
    for k in K_GRID:
        row = {}
        for label, res in results.items():
            block = res["per_k"][str(k)]
            rho = block["spearman_vs_full_audit"]["mean"]
            topq = block["retrieval_top_quartile"]["ratio_to_ceiling"]["mean"]
            err = block["q_blind"]["signed_error"]["mean"]
            row[label] = {
                "spearman_mean": rho, "spearman_ge_RHO_MIN": bool(rho is not None and rho >= RHO_MIN),
                "top_quartile_ratio_to_ceiling_mean": topq,
                "top_quartile_expected_mean": block["retrieval_top_quartile"]["expected_random_tiebreak"]["mean"],
                "top_quartile_ceiling": block["retrieval_top_quartile"]["ceiling_expected_random_tiebreak"],
                "top_quartile_ge_TOPQ_MIN": bool(topq is not None and topq >= TOPQ_MIN),
                "qualifies": bool(rho is not None and rho >= RHO_MIN and topq is not None and topq >= TOPQ_MIN),
                "q_blind_signed_error_mean": err,
                "q_blind_within_QBLIND_ABS_MAX": bool(err is not None and abs(err) <= QBLIND_ABS_MAX),
            }
        table[str(k)] = row
    return table


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not args.corpus:
        raise EstimatorError("--corpus LABEL REAL_GLOB BLIND_GLOB BLIND_CONDITION is required (repeatable)")
    if args.join_key not in JOIN_KEYS:
        raise EstimatorError(f"--join-key must be one of {JOIN_KEYS}")
    if not 1 <= args.replicates <= REPLICATES:
        raise EstimatorError(f"--replicates must be between 1 and the pinned REPLICATES ({REPLICATES}); "
                             f"the draw schedule is fixed at {REPLICATES} replicates per k")
    stratify = {}
    for label, key in args.stratify or []:
        if label in stratify:
            raise EstimatorError(f"--stratify given twice for corpus {label}")
        stratify[label] = key

    inputs: dict[str, str] = {}
    results: dict[str, Any] = {}
    for spec in args.corpus:
        label, real_glob, blind_glob, blind_condition = spec
        if label in results:
            raise EstimatorError(f"duplicate corpus label {label}")
        if blind_condition == REAL_CONDITION or blind_condition not in BLIND_CONDITIONS:
            raise EstimatorError(f"{label}: blind condition must be one of {BLIND_CONDITIONS}, got {blind_condition!r}")
        real = load_condition(real_glob, REAL_CONDITION, f"{label}/real")
        blind = load_condition(blind_glob, blind_condition, f"{label}/{blind_condition}")
        inputs.update(real["inputs"])
        inputs.update(blind["inputs"])
        join_key = choose_join_key(real, blind, args.join_key, label)
        corpus = join_corpus(label, real, blind, join_key, stratify.get(label))
        results[label] = corpus_metrics(corpus, replicates=args.replicates, draws=args.draws, seed=args.seed)
    unused = sorted(set(stratify) - set(results))
    if unused:
        raise EstimatorError(f"--stratify names corpora that were not built: {unused}")

    return {
        "schema_version": SCHEMA_VERSION,
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git_head(),
        "blind_condition_note": BLIND_CONDITION_NOTE,
        "no_cross_corpus_aggregate": NO_CROSS_CORPUS_NOTE,
        "nesting_caveat": NESTING_NOTE,
        "cost_note": COST_NOTE,
        "estimator": {
            "k_grid": list(K_GRID), "full_rollouts": FULL_ROLLOUTS, "replicates": args.replicates,
            "subsample_seed": SUBSAMPLE_SEED,
            "draw": "k of the 16 banked rollout records, WITHOUT replacement, from one permutation stream "
                    "per (corpus, condition, item key) with the (k, replicate) schedule pinned at "
                    f"{REPLICATES} replicates per k; no binomial resampling of stored counts",
            "reference": "Δq_16 = p_sample(real) − p_sample(blind), the banked full audit",
            "scoring": "banked `sample_correct` from lwl/audit/sampling.py; nothing re-scored here",
        },
        "bootstrap": {"draws": args.draws, "seed": args.seed,
                      "method": "paired per-item bootstrap (mean_with_paired_bootstrap); zero-variance cells as "
                                "exact counts + Clopper–Pearson"},
        "corpora": results,
        "qualification": {
            "thresholds": {"RHO_MIN": RHO_MIN, "TOPQ_MIN": TOPQ_MIN, "QBLIND_ABS_MAX": QBLIND_ABS_MAX},
            "qualification_table": qualification_table(results),
        },
        "inputs_sha256": dict(sorted(inputs.items())),
        "command": " ".join(sys.argv),
    }


def _f(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _pm(block: dict[str, Any], digits: int = 3) -> str:
    if not block or block.get("mean") is None:
        return "—"
    return f"{_f(block['mean'], digits)} [{_f(block['min'], digits)}, {_f(block['max'], digits)}]"


def render_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    est = payload["estimator"]
    lines.append("# Necessity estimator and cost curve\n")
    lines.append(f"Built {payload['built_utc']} · git `{(payload['git_head'] or '')[:12]}`\n")
    lines.append(f"Estimator: Δq̂_k from k ∈ {est['k_grid']} of {est['full_rollouts']} banked rollouts, "
                 f"{est['replicates']} replicates, subsample seed {est['subsample_seed']}; {est['draw']}. "
                 f"Reference: {est['reference']}. Scoring: {est['scoring']}.\n")
    lines.append(f"{payload['nesting_caveat']}\n")
    lines.append(f"{payload['cost_note']} {payload['no_cross_corpus_aggregate']}\n")
    lines.append(f"Blind condition: {payload['blind_condition_note']}\n")

    lines.append("## Corpora\n")
    lines.append("| corpus | items | join key | blind condition | q_real (16) | q_blind (16) | mean Δq_16 | items Δq_16 > 0 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, res in payload["corpora"].items():
        lines.append(f"| {label} | {res['n_items']} | `{res['join_key']}` | `{res['blind_condition']}` | "
                     f"{_f(res['q_real_full_audit'], 4)} | {_f(res['q_blind_full_audit'], 4)} | "
                     f"{_f(res['delta_q_full_audit_mean'], 4)} | {res['delta_q_full_audit_positive_items']} |")
    lines.append("")

    for label, res in payload["corpora"].items():
        lines.append(f"## {label} — rank stability and retrieval by k (mean [min, max] over replicates)\n")
        lines.append("| k | rollouts/item | cost vs full audit | ρ(Δq̂_k, Δq_16) | ρ disjoint halves | "
                     "top-quartile ÷ ceiling | top-quartile (random tie-break) | "
                     "bottom-quartile ÷ ceiling | top-quartile (key tie-break) | q̂_blind | signed q_blind error |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for k in est["k_grid"]:
            b = res["per_k"][str(k)]
            dis = b["spearman_disjoint_halves"]
            dis_txt = "—" if dis.get("available") is False else _pm(dis)
            top, bot, q = b["retrieval_top_quartile"], b["retrieval_bottom_quartile"], b["q_blind"]
            lines.append(
                f"| {k} | {b['cost']['rollouts_per_item']} | {_f(b['cost']['fraction_of_full_audit'], 3)} "
                f"({_f(b['cost']['rollout_saving_factor'], 1)}×) | {_pm(b['spearman_vs_full_audit'])} | {dis_txt} | "
                f"{_pm(top['ratio_to_ceiling'])} | {_pm(top['expected_random_tiebreak'])} | "
                f"{_pm(bot['ratio_to_ceiling'])} | {_pm(top['deterministic_tiebreak'])} | "
                f"{_pm(q['q_blind_hat'], 4)} | {_f(q['signed_error']['mean'], 4)} |")
        lines.append("")
        first = res["per_k"][str(est["k_grid"][0])]
        ties = first["reference_boundary_tie_mass"]
        lines.append(f"Reference Δq_16 boundary tie mass: top {_f(ties['top'])}, bottom {_f(ties['bottom'])} "
                     f"(quartile size m = {first['quartile_m']}); retrieval ceiling (a perfect estimator, same "
                     f"random tie-breaking): top {_f(first['retrieval_top_quartile']['ceiling_expected_random_tiebreak'])}, "
                     f"bottom {_f(first['retrieval_bottom_quartile']['ceiling_expected_random_tiebreak'])}; "
                     "estimator boundary tie mass per k is in the JSON.\n")
        lines.append("Corpus-level q_blind. The error column is the mean over replicates; the interval is the "
                     "paired per-item bootstrap of ONE replicate (the first), whose own error is given beside it:\n")
        lines.append("| k | q̂_blind | q_blind (16) | signed error (mean over reps) | replicate-1 error | "
                     "replicate-1 95% interval | method |")
        lines.append("|---|---|---|---|---|---|---|")
        for k in est["k_grid"]:
            q = res["per_k"][str(k)]["q_blind"]
            pb = q["paired_bootstrap_replicate1"]
            rep1 = q["signed_error"]["per_replicate"][0]
            ci = ("—" if pb.get("ci95") is None
                  else f"[{_f(pb['ci95'][0], 4)}, {_f(pb['ci95'][1], 4)}]")
            if pb.get("zero_variance"):
                ci = (f"exact: subsample {q['counts']['subsample']['k']}/{q['counts']['subsample']['n']} "
                      f"CP {pb['subsample_clopper_pearson_95']}, audit {q['counts']['full_audit']['k']}/"
                      f"{q['counts']['full_audit']['n']} CP {pb['full_audit_clopper_pearson_95']}")
            lines.append(f"| {k} | {_f(q['q_blind_hat']['mean'], 4)} | {_f(q['q_blind_full_audit'], 4)} | "
                         f"{_f(q['signed_error']['mean'], 4)} | {_f(rep1, 4)} | {ci} | {pb['method']} |")
        lines.append("")
        if res["stratum_key"]:
            lines.append(f"### {label} — strata by `{res['stratum_key']}` (never pooled; same per-item draws)\n")
            lines.append("| stratum | items | k | ρ(Δq̂_k, Δq_16) | top-quartile ÷ ceiling |")
            lines.append("|---|---|---|---|---|")
            for k in est["k_grid"]:
                for name, sb in (res["per_k"][str(k)].get("strata") or {}).items():
                    if sb.get("skipped"):
                        lines.append(f"| {name} | {sb['n_items']} | {k} | skipped | {sb['note']} |")
                        continue
                    lines.append(f"| {name} | {sb['n_items']} | {k} | {_pm(sb['spearman_vs_full_audit'])} | "
                                 f"{_pm(sb['retrieval_top_quartile']['ratio_to_ceiling'])} |")
            lines.append("")

    qual = payload["qualification"]
    lines.append("## Qualification table (arithmetic against the pinned thresholds)\n")
    thr = qual["thresholds"]
    lines.append(f"Qualifying k: mean ρ ≥ {thr['RHO_MIN']} AND mean top-quartile retrieval divided by its "
                 f"ceiling ≥ {thr['TOPQ_MIN']}; corpus-level cell: |signed q_blind error| ≤ "
                 f"{thr['QBLIND_ABS_MAX']}.\n")
    labels = list(payload["corpora"])
    lines.append("| k | " + " | ".join(f"{lab}: ρ / topQ÷ceiling / qualifies / q_blind ok" for lab in labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for k in est["k_grid"]:
        cells = []
        for lab in labels:
            row = qual["qualification_table"][str(k)][lab]
            cells.append(f"{_f(row['spearman_mean'])} / {_f(row['top_quartile_ratio_to_ceiling_mean'])} / "
                         f"{'yes' if row['qualifies'] else 'no'} / {'yes' if row['q_blind_within_QBLIND_ABS_MAX'] else 'no'}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Runs\n")
    for label, res in payload["corpora"].items():
        lines.append(f"- {label} real: " + ", ".join(f"`{d}`" for d in res["run_dirs"]["real"]))
        lines.append(f"- {label} {res['blind_condition']}: " + ", ".join(f"`{d}`" for d in res["run_dirs"]["blind"]))
    lines.append("")
    lines.append(f"Inputs: {len(payload['inputs_sha256'])} files sha256-pinned in the JSON. "
                 f"Command: `{payload['command']}`\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", nargs=4, action="append",
                        metavar=("LABEL", "REAL_GLOB", "BLIND_GLOB", "BLIND_CONDITION"))
    parser.add_argument("--stratify", nargs=2, action="append", metavar=("LABEL", "METADATA_KEY"))
    parser.add_argument("--join-key", default="auto", choices=JOIN_KEYS)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--md-output", required=True)
    args = parser.parse_args(argv)
    for out in (args.json_output, args.md_output):
        if Path(out).exists():
            raise SystemExit(f"report already exists: {out}")
    payload = build(args)
    markdown = render_md(payload)                 # render first; write both only when both exist
    Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    Path(args.md_output).write_text(markdown)
    print(f"wrote {args.json_output} and {args.md_output}: corpora {list(payload['corpora'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

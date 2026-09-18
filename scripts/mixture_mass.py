#!/usr/bin/env python3
"""Resolvability mass m(f) of each dose mixture: f times the constructed corpus's mean audited
necessity plus (1 - f) times the mean over the audited ViRL39K rows that mixture actually draws.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.paths import repo_root, results_root

ROOT = repo_root()
DRAWS = 5000
SEED = 20260716


def _load_estimator():
    """Import scripts/estimate_necessity.py so the join uses its own code path."""
    path = ROOT / "scripts" / "estimate_necessity.py"
    spec = importlib.util.spec_from_file_location("lwl_necessity_estimator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MassError(ValueError):
    pass


def bootstrap_mean_ci(values: np.ndarray, *, draws: int, seed: int) -> dict:
    """Paired per-item bootstrap over items. Zero-variance cells report exactly, never a fake CI."""
    n = int(values.size)
    if n == 0:
        raise MassError("cannot take a mean over an empty subset")
    point = float(values.mean())
    if float(values.var()) == 0.0:
        return {"mean": point, "ci95": [point, point], "n_items": n,
                "zero_variance": True,
                "note": "every item in this cell carries the same dq; the interval is the point "
                        "(exact), not a bootstrap"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": point, "ci95": [float(lo), float(hi)], "n_items": n,
            "zero_variance": False, "draws": draws, "seed": seed}


def arm_mass(arm: str, report_path: Path, joined: dict, dq_constructed: float, *,
             draws: int, seed: int) -> dict:
    """m(f) and its ViRL-side inputs for one mixture arm's build report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fraction = Fraction(str(report["fraction"]))
    stream = list(report["virl_qids"])
    if not stream and fraction != 1:
        raise MassError(f"{arm}: mixture report carries no virl_qids")

    keys = list(joined["keys"])
    index_of = {str(k): i for i, k in enumerate(keys)}
    dq_full = np.asarray(joined["dq_full"], dtype=float)

    unique_drawn = sorted(set(str(q) for q in stream))
    audited = [q for q in unique_drawn if q in index_of]
    if not audited:
        raise MassError(
            f"{arm}: none of its {len(unique_drawn)} unique virl_qids intersect the banked audit "
            f"sample ({len(keys)} items); the restricted estimand is undefined, and 0.0 is not a "
            f"stand-in for it")
    idx = np.array([index_of[q] for q in audited], dtype=int)
    restricted = bootstrap_mean_ci(dq_full[idx], draws=draws, seed=seed)

    # descriptive cross-check only: the same audited items weighted by how often the arm draws them
    mult = {}
    for q in stream:
        q = str(q)
        if q in index_of:
            mult[q] = mult.get(q, 0) + 1
    weights = np.array([mult[q] for q in audited], dtype=float)
    row_weighted = float((dq_full[idx] * weights).sum() / weights.sum())

    f = float(fraction)
    dq_virl = restricted["mean"]
    return {
        "arm": arm,
        "fraction": str(fraction),
        "fraction_float": f,
        "virl_rows_realized": len(stream),
        "virl_unique_qids": len(unique_drawn),
        "audited_subset_n": len(audited),
        "audited_subset_share_of_unique": len(audited) / len(unique_drawn),
        "dq_virl_restricted": restricted,
        "dq_virl_row_weighted_descriptive": row_weighted,
        "m_f": f * dq_constructed + (1.0 - f) * dq_virl,
        "source_report": str(report_path),
    }


def build(args: argparse.Namespace) -> dict:
    est = _load_estimator()
    out_json, out_md = Path(args.json_output), Path(args.md_output)
    for path in (out_json, out_md):
        if path.exists():
            raise MassError(f"{path} already exists")

    estimates = json.loads(Path(args.estimator_report).read_text(encoding="utf-8"))
    constructed = estimates["corpora"][args.constructed_label]
    dq_constructed = float(constructed["delta_q_full_audit_mean"])

    real = est.load_condition(args.virl_real, "real", args.virl_label)
    blind = est.load_condition(args.virl_blind, args.virl_blind_condition, args.virl_label)
    joined = est.join_corpus(args.virl_label, real, blind, "qid", None)

    pool_mean = float(np.asarray(joined["dq_full"], dtype=float).mean())
    banked = estimates["corpora"][args.virl_label]
    banked_mean = float(banked["delta_q_full_audit_mean"])
    # The estimator stores this rounded to 6 dp, so 6 dp is the tightest agreement that can be
    # demanded; anything tighter would fail on the rounding rather than on a disagreement. The
    # observed residual is recorded below so genuine drift stays visible under the bound.
    residual = abs(pool_mean - banked_mean)
    if residual > 5e-7:
        raise MassError(
            f"pool-level dq for {args.virl_label} recomputed as {pool_mean!r} but the estimator "
            f"banked {banked_mean!r} (residual {residual:.3e} > 5e-7 at its 6-dp storage "
            f"precision); the loaders disagree")

    rows = []
    for arm, report_path in args.arm:
        rows.append(arm_mass(arm, Path(report_path), joined, dq_constructed,
                             draws=args.draws, seed=args.seed))
    # the f = 1 point is the standard arm, banked and never re-run: it draws no ViRL rows,
    # so m(1) = dq_constructed
    rows.append({"arm": args.banked_label, "fraction": "1", "fraction_float": 1.0,
                 "virl_rows_realized": 0, "virl_unique_qids": 0, "audited_subset_n": 0,
                 "audited_subset_share_of_unique": None,
                 "dq_virl_restricted": None, "dq_virl_row_weighted_descriptive": None,
                 "m_f": dq_constructed,
                 "source_report": "banked arm 1; no ViRL side by construction"})
    rows.sort(key=lambda r: r["fraction_float"])

    masses = [r["m_f"] for r in rows]
    strictly_increasing = all(b > a for a, b in zip(masses, masses[1:]))

    payload = {
        "schema_version": "lwl.mixture-mass.v1",
        "built_utc": args.built_utc,
        "git_head": est.git_head(),
        "estimator_path": "scripts/estimate_necessity.py (imported)",
        "estimator_report": args.estimator_report,
        "dq_constructed": {"label": args.constructed_label, "mean": dq_constructed,
                           "n_items": constructed["n_items"], "source": "delta_q_full_audit_mean"},
        "dq_virl_pool_unrestricted": {"label": args.virl_label, "mean": pool_mean,
                                      "n_items": joined["n_items"],
                                      "banked_mean": banked_mean,
                                      "recompute_residual": residual,
                                      "agreement_bound": 5e-7},
        "estimand_note": ("dq_virl per arm is the unweighted mean over the audited rows the arm "
                          "actually draws: the audit is a proportional-allocation sample, so the "
                          "unweighted domain mean is the estimator and the domain restriction is "
                          "reported rather than corrected. The row-weighted figure beside it is "
                          "descriptive and feeds nothing."),
        "bootstrap": {"draws": args.draws, "seed": args.seed, "kind": "paired per-item"},
        "rows": rows,
        "m_strictly_increasing_in_f": strictly_increasing,
        "locks": joined["locks"],
        "run_dirs": joined["run_dirs"],
    }
    if not strictly_increasing:
        payload["blocked"] = "m(f) is not strictly increasing in f"
    return payload


def render_md(p: dict) -> str:
    lines = [
        "# Resolvability mass per mixture arm",
        "",
        f"Built {p['built_utc']} · git `{p['git_head']}`",
        f"Estimator: {p['estimator_path']} · inputs: `{p['estimator_report']}`",
        "",
        f"- constructed side Δq̄ = **{p['dq_constructed']['mean']:.6f}** "
        f"({p['dq_constructed']['n_items']} items)",
        f"- ViRL side Δq̄ over the whole banked audit = **{p['dq_virl_pool_unrestricted']['mean']:.6f}** "
        f"({p['dq_virl_pool_unrestricted']['n_items']} items) — reported beside the restricted values",
        "",
        "| arm | f | audited subset n | Δq̄ ViRL (restricted) | 95% CI | m(f) |",
        "|---|---|---|---|---|---|",
    ]
    for r in p["rows"]:
        if r["dq_virl_restricted"] is None:
            lines.append(f"| `{r['arm']}` | {r['fraction']} | — | — (no ViRL side) | — | "
                         f"**{r['m_f']:.6f}** |")
        else:
            d = r["dq_virl_restricted"]
            lines.append(
                f"| `{r['arm']}` | {r['fraction']} | {r['audited_subset_n']} | {d['mean']:.6f} | "
                f"[{d['ci95'][0]:.6f}, {d['ci95'][1]:.6f}] | **{r['m_f']:.6f}** |")
    lines += ["", f"m(f) strictly increasing in f: **{p['m_strictly_increasing_in_f']}**", "",
              p["estimand_note"], ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimator-report",
                    default=str(results_root() / "necessity_estimator_v1.json"),
                    help="JSON written by scripts/estimate_necessity.py")
    ap.add_argument("--constructed-label", default="constructed", metavar="LABEL",
                    help="corpus label of the constructed scenes in the estimator report")
    ap.add_argument("--virl-label", default="virl39k_audit_7b_base")
    ap.add_argument("--virl-real", required=True)
    ap.add_argument("--virl-blind", required=True)
    ap.add_argument("--virl-blind-condition", default="none")
    ap.add_argument("--arm", nargs=2, action="append", metavar=("ARM", "MIXTURE_REPORT"),
                    required=True)
    ap.add_argument("--banked-label", default="arm 1 (f = 1, banked)")
    ap.add_argument("--built-utc", required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json-output", required=True)
    ap.add_argument("--md-output", required=True)
    args = ap.parse_args(argv)
    try:
        payload = build(args)
    except (MassError, KeyError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3
    Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    Path(args.md_output).write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({r["arm"]: round(r["m_f"], 6) for r in payload["rows"]}, indent=2))
    print("strictly increasing:", payload["m_strictly_increasing_in_f"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-mixture base-difficulty and learnability-band profiles.

Computed over each arm's realized stream -- the arm's own train.jsonl, 7,200 rows -- not over
the pools it draws from. Each row is joined to the base model's per-item real accuracy through
the necessity estimator's own loaders, so the covariate sits on the same footing as the
resolvability-mass table:

    constructed row -> qid  "{pair_group_uid}:{pair_member}"
    ViRL row        -> qid   the suffix of "virl:<qid>"

An item is in the learnability band iff its base success is strictly inside (0, 1), i.e. it can
produce reward variance and therefore gradient.

Also reports the covariate's common support across arms: if the mixtures overlap too little to
match strata (< 30% of items in common support) the comparison cannot be adjusted, and that is a
corpus property, decidable before any arm runs. Writes its two outputs and nothing else, and
never overwrites.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from lwl.paths import repo_root

ROOT = repo_root()
DECILES = [round(0.1 * i, 1) for i in range(11)]
# Category of the constructed rows, as released.
CONSTRUCTED_CATEGORY = "hier_v1_st3"


class ProfileError(ValueError):
    pass


def _est():
    """Import scripts/estimate_necessity.py so the join uses its own code path."""
    spec = importlib.util.spec_from_file_location(
        "lwl_necessity_estimator", ROOT / "scripts" / "estimate_necessity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def q_real_by_qid(joined) -> dict:
    keys = list(joined["keys"])
    mat = np.asarray(joined["real_matrix"], dtype=float)
    return {str(k): float(mat[i].mean()) for i, k in enumerate(keys)}


def row_key(rec):
    """(side, join key) for one corpus row."""
    cat = rec.get("category", "")
    pgu = str(rec.get("pair_group_uid", ""))
    if cat == CONSTRUCTED_CATEGORY:
        return "constructed", "%s:%s" % (pgu, rec.get("pair_member"))
    if pgu.startswith("virl:"):
        return "virl", pgu[len("virl:"):]
    return "virl", pgu


def profile(arm, corpus_dir, qconstructed, qvirl):
    path = Path(corpus_dir) / "train.jsonl"
    if not path.is_file():
        raise ProfileError("%s: no realized stream at %s" % (arm, path))
    acc, side_n, matched_n = [], {"constructed": 0, "virl": 0}, {"constructed": 0, "virl": 0}
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        side, key = row_key(rec)
        side_n[side] += 1
        table = qconstructed if side == "constructed" else qvirl
        if key in table:
            matched_n[side] += 1
            acc.append(table[key])
    if not acc:
        raise ProfileError("%s: no realized row joined to a base-accuracy record" % arm)
    a = np.array(acc, dtype=float)
    hist, _ = np.histogram(a, bins=np.linspace(0.0, 1.0, 11))
    in_band = (a > 0.0) & (a < 1.0)
    return {
        "arm": arm,
        "corpus_dir": str(corpus_dir),
        "rows_realized": int(sum(side_n.values())),
        "rows_by_side": side_n,
        "rows_joined_by_side": matched_n,
        "rows_joined_total": int(a.size),
        "join_coverage": float(a.size / max(1, sum(side_n.values()))),
        "base_accuracy": {
            "mean": float(a.mean()), "median": float(np.median(a)),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "decile_edges": DECILES,
            "decile_counts": [int(x) for x in hist],
            "decile_shares": [float(x) / a.size for x in hist],
        },
        "band_mass": float(in_band.mean()),
        "band_definition": "base real accuracy strictly inside (0, 1) -- the item can produce reward variance",
        "_strata": None,
    }


def strata_of(arm_rows):
    """(accuracy decile, band) -> count. The covariate the arms are matched on."""
    out = {}
    for v in arm_rows:
        d = min(9, int(v * 10))
        b = 1 if 0.0 < v < 1.0 else 0
        out[(d, b)] = out.get((d, b), 0) + 1
    return out


def build(args):
    est = _est()
    for path in (Path(args.json_output), Path(args.md_output)):
        if path.exists():
            raise ProfileError("%s already exists" % path)

    cj = est.join_corpus("constructed", est.load_condition(args.constructed_real, "real", "constructed"),
                         est.load_condition(args.constructed_blind, args.constructed_blind_condition,
                                            "constructed"), "qid", None)
    vj = est.join_corpus("virl", est.load_condition(args.virl_real, "real", "virl"),
                         est.load_condition(args.virl_blind, args.virl_blind_condition, "virl"), "qid", None)
    qconstructed, qvirl = q_real_by_qid(cj), q_real_by_qid(vj)

    rows, raw = [], {}
    for arm, corpus_dir in args.arm:
        p = profile(arm, corpus_dir, qconstructed, qvirl)
        acc = []
        path = Path(corpus_dir) / "train.jsonl"
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            side, key = row_key(rec)
            t = qconstructed if side == "constructed" else qvirl
            if key in t:
                acc.append(t[key])
        raw[arm] = acc
        p.pop("_strata")
        rows.append(p)

    # common support on the joint covariate, pairwise and across all arms
    strata = {a: strata_of(v) for a, v in raw.items()}
    names = list(strata)
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(strata[a]) & set(strata[b])
            na = sum(strata[a][k] for k in shared) / max(1, sum(strata[a].values()))
            nb = sum(strata[b][k] for k in shared) / max(1, sum(strata[b].values()))
            pairwise["%s|%s" % (a, b)] = {
                "shared_strata": len(shared),
                "share_of_%s_rows_in_common_support" % a: float(na),
                "share_of_%s_rows_in_common_support" % b: float(nb),
                "min_share": float(min(na, nb)),
            }
    allshared = set.intersection(*[set(strata[a]) for a in names]) if names else set()
    across = {a: float(sum(strata[a][k] for k in allshared) / max(1, sum(strata[a].values()))) for a in names}
    min_across = min(across.values()) if across else 0.0

    return {
        "schema_version": "lwl.mixture-profiles.v1",
        "built_utc": args.built_utc,
        "git_head": est.git_head(),
        "estimator_path": "scripts/estimate_necessity.py (imported)",
        "base_model": "Qwen2.5-VL-7B-Instruct, 16 samples, T=1 (the blind-opportunity audit pass)",
        "sides": {
            "constructed": {"n_items_audited": cj["n_items"], "q_real_pool_mean": cj["q_real_full"]},
            "virl": {"n_items_audited": vj["n_items"], "q_real_pool_mean": vj["q_real_full"]},
        },
        "arms": rows,
        "common_support": {
            "covariate": "(base-accuracy decile, in-band indicator) -- the pair the arms are matched on",
            "threshold": 0.30,
            "pairwise": pairwise,
            "across_all_arms": across,
            "min_share_across_all_arms": min_across,
            "below_threshold": bool(min_across < 0.30),
        },
        "caveat": ("ViRL rows join only where the arm draws a row inside the banked audit sample, so "
                   "join coverage is partial on that side and is reported per arm; the constructed "
                   "side is fully audited. Shares are over joined rows."),
    }


def render_md(p):
    L = ["# Per-mixture base-difficulty and learnability-band profiles", "",
         "Built %s · git `%s`" % (p["built_utc"], p["git_head"]),
         "Base accuracy is the 7B base model's per-item real-image success, joined to each arm's realized stream.", "",
         "| arm | rows joined | base acc mean | median | band mass |", "|---|---|---|---|---|"]
    for a in p["arms"]:
        L.append("| `%s` | %d / %d | %.4f | %.4f | **%.4f** |" % (
            a["arm"], a["rows_joined_total"], a["rows_realized"],
            a["base_accuracy"]["mean"], a["base_accuracy"]["median"], a["band_mass"]))
    L += ["", "**Base-accuracy decile shares** (0.0–0.1 … 0.9–1.0):", ""]
    for a in p["arms"]:
        L.append("- `%s`: %s" % (a["arm"], " ".join("%.3f" % x for x in a["base_accuracy"]["decile_shares"])))
    cs = p["common_support"]
    L += ["", "**Common support on (%s)** — strata-matched adjustment needs a minimum share of at least %.2f." % (
        cs["covariate"], cs["threshold"]), ""]
    for k, v in cs["pairwise"].items():
        L.append("- %s: %d shared strata, minimum share **%.4f**" % (k, v["shared_strata"], v["min_share"]))
    L += ["", "Across all arms: %s — minimum **%.4f**. Below the threshold: **%s**." % (
        ", ".join("%s %.4f" % (k, v) for k, v in cs["across_all_arms"].items()),
        cs["min_share_across_all_arms"], cs["below_threshold"]), "", p["caveat"], ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--constructed-real", required=True, metavar="PATTERN",
                    help="audit run directories of the constructed scenes, real images")
    ap.add_argument("--constructed-blind", required=True, metavar="PATTERN",
                    help="audit run directories of the constructed scenes, image withheld")
    ap.add_argument("--constructed-blind-condition", default="none", metavar="CONDITION")
    ap.add_argument("--virl-real", required=True); ap.add_argument("--virl-blind", required=True)
    ap.add_argument("--virl-blind-condition", default="none")
    ap.add_argument("--arm", nargs=2, action="append", metavar=("ARM", "CORPUS_DIR"), required=True)
    ap.add_argument("--built-utc", required=True)
    ap.add_argument("--json-output", required=True); ap.add_argument("--md-output", required=True)
    a = ap.parse_args(argv)
    try:
        payload = build(a)
    except (ProfileError, KeyError) as exc:
        print("BLOCKED: %s" % exc, file=sys.stderr); return 3
    Path(a.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(a.md_output).write_text(render_md(payload), encoding="utf-8")
    for arm in payload["arms"]:
        print("%-5s band_mass %.4f  base_acc_mean %.4f  joined %d/%d" % (
            arm["arm"], arm["band_mass"], arm["base_accuracy"]["mean"],
            arm["rows_joined_total"], arm["rows_realized"]))
    print("min common support across arms: %.4f (below threshold: %s)" % (
        payload["common_support"]["min_share_across_all_arms"],
        payload["common_support"]["below_threshold"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

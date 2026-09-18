#!/usr/bin/env python3
"""Combine scored pair files into one metrics file: aggregate and per-template accuracy,
a bootstrap interval, and the two permutation nulls.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.evaluation.metrics import (
    aggregate_pair_metrics,
    aggregate_pair_metrics_by_template,
    pair_accuracy_ci,
    permutation_null_pair_accuracy,
    template_key_shuffle_null_pair_accuracy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    rows = []
    matched_paths: list[Path] = []
    for pattern in args.inputs:
        # glob.glob, unlike Path.glob, also takes absolute patterns.
        for path in sorted(Path(match) for match in glob.glob(pattern, recursive=True)):
            matched_paths.append(path)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
    if not matched_paths or not rows:
        raise ValueError("inputs matched no nonempty rows")

    metrics = aggregate_pair_metrics(rows)
    metrics["per_template"] = aggregate_pair_metrics_by_template(rows)
    lo, hi = pair_accuracy_ci(rows, n_boot=args.bootstrap)
    swap_null = permutation_null_pair_accuracy(rows, n_perm=args.permutations)
    key_shuffle_null = template_key_shuffle_null_pair_accuracy(rows, n_perm=args.permutations)
    metrics.update(
        {
            "pair_accuracy_ci95_low": lo,
            "pair_accuracy_ci95_high": hi,
            "swap_null_mean": swap_null["null_mean"],
            "swap_null_p_ge": swap_null["p_ge"],
            "key_shuffle_null_mean": key_shuffle_null["null_mean"],
            "key_shuffle_null_p_ge": key_shuffle_null["p_ge"],
            "permutation_null_mean": key_shuffle_null["null_mean"],
            "permutation_p_ge": key_shuffle_null["p_ge"],
        }
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{out_path}.partial")
    if out_path.exists() or partial.exists():
        raise FileExistsError(f"aggregate already exists: {out_path}")
    partial.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, out_path)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()

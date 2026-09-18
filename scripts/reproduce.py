#!/usr/bin/env python3
"""Rebuild the paper's tables and figure values from the released per-item outputs.

    python scripts/reproduce.py --list
    python scripts/reproduce.py all
    python scripts/reproduce.py table1 figure4 --check

With --check, each target reports how many printed values it compares and how many match,
separating those that match only after the paper's own rounding, printed values the released
records cannot rebuild, and verified inconsistencies in the paper. The exit status is nonzero
when a target fails, a needed input is missing or a mismatch is unexplained.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.analysis.paper import (
    INPUT_MISSING,
    MATCHED,
    MATCHED_AFTER_ROUNDING,
    MISMATCH,
    NOT_REBUILDABLE,
    PAPER_ISSUE,
    comparison_outcome,
    import_errors,
    registry,
)
from lwl.analysis.report import write_table


def column_order(rows: list[dict[str, Any]]) -> list[str]:
    """Columns in the order the rows introduce them."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def check_counts(rows: list[dict[str, Any]]) -> Counter:
    """Rows carrying a printed value, counted by how they compare with the paper."""
    return Counter(
        outcome for outcome in map(comparison_outcome, rows) if outcome is not None
    )


def check_summary(counts: Counter) -> str:
    checked = sum(counts.values())
    matched = counts[MATCHED] + counts[MATCHED_AFTER_ROUNDING]
    line = f"  checked={checked:5d}  matched={matched:5d}"
    extras = (
        (MATCHED_AFTER_ROUNDING, "of them after the paper's rounding"),
        (NOT_REBUILDABLE, "not rebuildable from the release"),
        (PAPER_ISSUE, "paper issues"),
        (INPUT_MISSING, "inputs missing"),
        (MISMATCH, "unexplained"),
    )
    notes = [f"{counts[outcome]} {label}" for outcome, label in extras if counts[outcome]]
    return line + (f"  ({'; '.join(notes)})" if notes else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", help='target names, or "all"')
    parser.add_argument("--list", action="store_true", help="list the available targets")
    parser.add_argument(
        "--check", action="store_true", help="compare each rebuilt value against the printed one"
    )
    args = parser.parse_args()

    targets = registry()
    for module, problem in sorted(import_errors().items()):
        print(f"skipped {module}: {problem}", file=sys.stderr)

    if args.list:
        if not targets:
            print("no targets available")
        for name in sorted(targets):
            print(f"{name:28s} {targets[name].__module__}")
        return 0

    if not args.targets:
        parser.error('name at least one target, or "all"')
    selected = sorted(targets) if "all" in args.targets else list(dict.fromkeys(args.targets))
    unknown = [name for name in selected if name not in targets]
    if unknown:
        parser.error(f"unknown targets: {', '.join(unknown)}; --list shows what is available")

    failures = 0
    for name in selected:
        try:
            rows = list(targets[name]())
            columns = column_order(rows)
            if rows:
                write_table(name, rows, columns)
        except Exception as error:
            failures += 1
            print(f"{name:28s} failed: {type(error).__name__}: {error}")
            continue
        line = f"{name:28s} values={len(rows):5d}"
        if args.check:
            counts = check_counts(rows)
            line += check_summary(counts)
            if counts[MISMATCH] or counts[INPUT_MISSING]:
                failures += 1
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

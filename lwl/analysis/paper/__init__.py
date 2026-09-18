"""Registry of the targets that rebuild the paper's tables and figure values.

Each module in this package exposes ``TARGETS = {"<target name>": callable}``. The callable
takes no arguments, returns a list of row dicts and writes nothing itself; the caller writes
the rows. A row reports its rebuilt number under ``value``; when the paper prints that number,
the row carries it under ``paper_value`` and may set its own ``tolerance``. When the paper also
prints an interval, the row carries it under ``paper_ci_low`` and ``paper_ci_high`` next to the
rebuilt ``ci_low`` and ``ci_high``, and may set ``interval_tolerance`` for the bounds.

A row may also carry:
  ``relation``        how the printed value is stated: eq (default, within the tolerance), le, lt,
                      ge, gt (a bound on the rebuilt value), zero (exactly zero) or undefined (the
                      paper prints the quantity as undefined, so no value is rebuilt);
  ``paper_rounding``  the value the paper's own rounding produces from the rebuilt numbers, where
                      it rounded an intermediate to four decimals before printing three;
  ``paper_issue``     a verified inconsistency in the paper that explains a mismatch;
  ``status``          "not rebuildable" for a printed value the released records cannot rebuild,
                      "input missing" when a needed input file is absent.
"""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

Row = dict[str, Any]
Builder = Callable[[], list[Row]]

VALUE_FIELD = "value"
PAPER_VALUE_FIELD = "paper_value"
TOLERANCE_FIELD = "tolerance"
INTERVAL_FIELDS = (("ci_low", "paper_ci_low"), ("ci_high", "paper_ci_high"))
INTERVAL_TOLERANCE_FIELD = "interval_tolerance"
RELATION_FIELD = "relation"
PAPER_ROUNDING_FIELD = "paper_rounding"
PAPER_ISSUE_FIELD = "paper_issue"
STATUS_FIELD = "status"

NOT_REBUILDABLE = "not rebuildable"
INPUT_MISSING = "input missing"

# Outcomes of comparing one row with the paper.
MATCHED = "matched"
MATCHED_AFTER_ROUNDING = "matched after the paper's rounding"
PAPER_ISSUE = "paper issue"
MISMATCH = "mismatch"

# The paper prints three decimals, so a rebuilt value within half of the last printed digit
# agrees with it.
DEFAULT_TOLERANCE = 5e-4
# Bounds are compared exactly, up to floating-point noise.
BOUND_GUARD = 1e-9

_IMPORT_ERRORS: dict[str, str] = {}


def registry() -> dict[str, Builder]:
    """Every target exposed by the modules in this package, keyed by target name.

    Modules that fail to import or break the contract are skipped and recorded in
    `import_errors`, so the remaining targets stay usable.
    """
    _IMPORT_ERRORS.clear()
    found: dict[str, Builder] = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda entry: entry.name):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{info.name}")
        except Exception as error:
            _IMPORT_ERRORS[info.name] = f"{type(error).__name__}: {error}"
            continue
        targets = getattr(module, "TARGETS", None)
        if not isinstance(targets, dict):
            _IMPORT_ERRORS[info.name] = "no TARGETS mapping"
            continue
        for name, builder in targets.items():
            if name in found:
                _IMPORT_ERRORS[f"{info.name}.{name}"] = "target name already taken"
                continue
            if not callable(builder):
                _IMPORT_ERRORS[f"{info.name}.{name}"] = "target is not callable"
                continue
            found[name] = builder
    return found


def import_errors() -> dict[str, str]:
    """What the last `registry` call had to skip."""
    return dict(_IMPORT_ERRORS)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _agrees(value: Any, printed: Any, tolerance: float) -> bool:
    if _is_number(value) and _is_number(printed):
        return abs(float(value) - float(printed)) <= tolerance
    return value == printed


def _holds(value: Any, printed: Any, relation: str, tolerance: float) -> bool:
    """Whether a rebuilt value satisfies the printed statement."""
    if relation == "undefined":
        return value is None
    if value is None:
        return False
    if relation == "eq":
        return _agrees(value, printed, tolerance)
    if not (_is_number(value) and _is_number(printed)):
        return False
    value, printed = float(value), float(printed)
    if relation == "zero":
        return value == 0.0
    if relation == "le":
        return value <= printed + BOUND_GUARD
    if relation == "lt":
        return value < printed
    if relation == "ge":
        return value >= printed - BOUND_GUARD
    if relation == "gt":
        return value > printed
    raise ValueError(f"unknown relation: {relation!r}")


def comparison_outcome(row: Row) -> str | None:
    """How a row compares with the paper: MATCHED, MATCHED_AFTER_ROUNDING, NOT_REBUILDABLE,
    INPUT_MISSING, PAPER_ISSUE or MISMATCH, or None when the row carries nothing printed.

    Each printed interval bound must have a rebuilt bound within ``interval_tolerance``, which
    defaults to the row's value tolerance.
    """
    printed = row.get(PAPER_VALUE_FIELD)
    bounds = [
        (row.get(rebuilt), row.get(paper))
        for rebuilt, paper in INTERVAL_FIELDS
        if row.get(paper) is not None
    ]
    if printed is None and not bounds:
        return None
    value = row.get(VALUE_FIELD)
    status = row.get(STATUS_FIELD)
    if value is None and status in (NOT_REBUILDABLE, INPUT_MISSING):
        return status
    relation = row.get(RELATION_FIELD) or "eq"
    tolerance = row.get(TOLERANCE_FIELD)
    tolerance = DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    interval_tolerance = row.get(INTERVAL_TOLERANCE_FIELD)
    interval_tolerance = tolerance if interval_tolerance is None else float(interval_tolerance)
    bounds_agree = all(
        rebuilt is not None and _agrees(rebuilt, paper, interval_tolerance)
        for rebuilt, paper in bounds
    )
    if bounds_agree and (printed is None or _holds(value, printed, relation, tolerance)):
        return MATCHED
    rounded = row.get(PAPER_ROUNDING_FIELD)
    if bounds_agree and rounded is not None and _agrees(rounded, printed, tolerance):
        return MATCHED_AFTER_ROUNDING
    if row.get(PAPER_ISSUE_FIELD):
        return PAPER_ISSUE
    return MISMATCH


def paper_comparison(row: Row) -> bool | None:
    """Whether a row matches the paper, directly or after the paper's documented rounding; None
    when the row carries nothing printed."""
    outcome = comparison_outcome(row)
    if outcome is None:
        return None
    return outcome in (MATCHED, MATCHED_AFTER_ROUNDING)

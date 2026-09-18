"""Readers for the released per-item outputs: the gzipped JSONL families, their manifest,
and small helpers for filtering and grouping the rows.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from lwl.paths import predictions_root

Row = dict[str, Any]

FAMILIES = (
    "access_matrix_geometry3k_3b",
    "access_matrix_virl39k_3b",
    "access_pair_geometry3k_7b",
    "constructed_corpus_runs",
    "dose_mixtures_7b",
    "instrument_baselines",
    "long_horizon_geometry3k_3b",
    "necessity_audit_benchmarks",
    "necessity_audit_cross_family",
    "resolvability_audit",
)

KINDS = ("pairs", "items", "audits")

MANIFEST_NAME = "manifest.json"


def family_path(family: str, kind: str) -> Path:
    """Path of one released file; raises when the file is not present."""
    if kind not in KINDS:
        raise ValueError(f"unknown record kind: {kind}")
    path = predictions_root() / family / f"{kind}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"no {kind} file for {family}: {path}")
    return path


def available_families(kind: str | None = None) -> list[str]:
    """Families present under the predictions directory, released ones first."""
    root = predictions_root()
    if not root.is_dir():
        return []
    present = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    ordered = [name for name in FAMILIES if name in present]
    ordered += [name for name in present if name not in FAMILIES]
    if kind is None:
        return ordered
    return [name for name in ordered if (root / name / f"{kind}.jsonl.gz").exists()]


def iter_records(family: str, kind: str) -> Iterator[Row]:
    """Records of one file, decompressed and parsed one line at a time."""
    with gzip.open(family_path(family, kind), "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _matches(row: Row, filters: dict[str, Any]) -> bool:
    for field, wanted in filters.items():
        value = row.get(field)
        if callable(wanted):
            if not wanted(value):
                return False
        elif isinstance(wanted, (list, tuple, set, frozenset)):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


def select(rows: Iterable[Row], **filters: Any) -> list[Row]:
    """Rows matching every filter: a value, a collection of accepted values, or a predicate."""
    return [row for row in rows if _matches(row, filters)]


def group_by(rows: Iterable[Row], *keys: str) -> dict[Any, list[Row]]:
    """Group rows by one field (scalar keys) or several (tuple keys), preserving row order."""
    if not keys:
        raise ValueError("group_by needs at least one key")
    grouped: dict[Any, list[Row]] = {}
    for row in rows:
        if len(keys) == 1:
            key: Any = row.get(keys[0])
        else:
            key = tuple(row.get(field) for field in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def _load(kind: str, family: str | Sequence[str] | None, filters: dict[str, Any]) -> list[Row]:
    if family is None:
        families: Sequence[str] = available_families(kind)
    elif isinstance(family, str):
        families = [family]
    else:
        families = list(family)
    rows: list[Row] = []
    for name in families:
        rows.extend(row for row in iter_records(name, kind) if _matches(row, filters))
    return rows


def load_pairs(family: str | Sequence[str] | None = None, **filters: Any) -> list[Row]:
    """Counterfactual pair records, from one family, several, or all of them."""
    return _load("pairs", family, filters)


def load_items(family: str | Sequence[str] | None = None, **filters: Any) -> list[Row]:
    """Single-image item records, from one family, several, or all of them."""
    return _load("items", family, filters)


def load_audits(family: str | Sequence[str] | None = None, **filters: Any) -> list[Row]:
    """Sampled audit records, from one family, several, or all of them."""
    return _load("audits", family, filters)


def manifest() -> dict[str, dict[str, dict[str, Any]]]:
    """The released manifest: record count, byte size and sha256 per family and kind."""
    path = predictions_root() / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_rows() -> list[Row]:
    """The manifest flattened to one row per file."""
    rows: list[Row] = []
    for family, kinds in manifest().items():
        for kind, entry in kinds.items():
            rows.append(
                {
                    "family": family,
                    "kind": kind,
                    "records": entry.get("records"),
                    "bytes": entry.get("bytes"),
                    "sha256": entry.get("sha256"),
                }
            )
    return rows


def file_sha256(family: str, kind: str, block_size: int = 1 << 20) -> str:
    """sha256 of one released file as it sits on disk."""
    digest = hashlib.sha256()
    with family_path(family, kind).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(family: str | None = None) -> list[Row]:
    """Compare the files on disk against the manifest, one row per file."""
    rows: list[Row] = []
    for entry in manifest_rows():
        if family is not None and entry["family"] != family:
            continue
        row = dict(entry)
        try:
            path = family_path(entry["family"], entry["kind"])
        except (FileNotFoundError, ValueError) as error:
            row.update({"present": False, "sha256_on_disk": None, "matches": False, "problem": str(error)})
            rows.append(row)
            continue
        on_disk = file_sha256(entry["family"], entry["kind"])
        row.update(
            {
                "present": True,
                "bytes_on_disk": path.stat().st_size,
                "sha256_on_disk": on_disk,
                "matches": on_disk == entry["sha256"],
                "problem": None,
            }
        )
        rows.append(row)
    return rows

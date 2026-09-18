"""Locations of the repository and of the downloaded data.

The data directory (instruments and corpora) defaults to ``<repo>/data`` and moves with
``LWL_DATA``; the per-item predictions default to ``<repo>/predictions`` (``LWL_PREDICTIONS``) and
the rebuilt tables to ``<repo>/results`` (``LWL_RESULTS``).
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return _REPO_ROOT


def data_root() -> Path:
    override = os.environ.get("LWL_DATA")
    return Path(override).expanduser().resolve() if override else _REPO_ROOT / "data"


def data_path(*parts) -> Path:
    return data_root().joinpath(*parts)


def predictions_root() -> Path:
    override = os.environ.get("LWL_PREDICTIONS")
    return Path(override).expanduser().resolve() if override else _REPO_ROOT / "predictions"


def results_root() -> Path:
    override = os.environ.get("LWL_RESULTS")
    return Path(override).expanduser().resolve() if override else _REPO_ROOT / "results"


def repo_relative(path) -> str:
    """A path as a manifest records it: relative to the repository, or unchanged when it lies outside."""
    candidate = Path(path)
    return str(candidate.relative_to(_REPO_ROOT)) if candidate.is_relative_to(_REPO_ROOT) else str(candidate)


def resolve_data_reference(reference, base=None) -> Path:
    """Where a path recorded in a manifest is. With LWL_DATA set, a `data/...` reference always
    lies under it; otherwise the path as written when that resolves, else beside `base`, else
    under the data directory."""
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate
    parts = candidate.parts
    under_data = bool(parts) and parts[0] == "data"
    if under_data and os.environ.get("LWL_DATA"):
        return data_root().joinpath(*parts[1:])
    if candidate.exists():
        return candidate
    if base is not None:
        beside = Path(base) / candidate
        if beside.exists():
            return beside
    if under_data:
        return data_root().joinpath(*parts[1:])
    return _REPO_ROOT / candidate

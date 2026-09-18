"""Readers for the released per-item files: filtering, grouping and the manifest check."""
from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from lwl.analysis import load
from lwl.paths import predictions_root


PAIRS = [
    {"pair_id": "p1", "run": "run-a", "seed": 1, "layer": "l3", "pair_correct": True},
    {"pair_id": "p2", "run": "run-a", "seed": 1, "layer": "probe", "pair_correct": False},
    {"pair_id": "p3", "run": "run-a", "seed": 2, "layer": "l3", "pair_correct": True},
    {"pair_id": "p4", "run": "run-b", "seed": 2, "layer": "l3", "pair_correct": False},
]
ITEMS = [
    {"item_id": "i1", "run": "run-a", "test_condition": "real", "correct": True},
    {"item_id": "i2", "run": "run-a", "test_condition": "none", "correct": False},
]
OTHER_PAIRS = [
    {"pair_id": "q1", "run": "run-c", "seed": 1, "layer": "l3", "pair_correct": True},
]


def write_records(path, rows):
    blob = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(blob)
    return {
        "records": len(rows),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.fixture
def released(tmp_path, monkeypatch):
    """A two-family stand-in for the released files, with its own manifest."""
    manifest = {
        "instrument_baselines": {
            "pairs": write_records(tmp_path / "instrument_baselines" / "pairs.jsonl.gz", PAIRS),
            "items": write_records(tmp_path / "instrument_baselines" / "items.jsonl.gz", ITEMS),
        },
        "extra_family": {
            "pairs": write_records(tmp_path / "extra_family" / "pairs.jsonl.gz", OTHER_PAIRS),
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("LWL_PREDICTIONS", str(tmp_path))
    return tmp_path


def test_the_released_families_are_listed_first(released):
    assert load.available_families() == ["instrument_baselines", "extra_family"], (
        f"unexpected order: {load.available_families()}"
    )
    assert load.available_families("items") == ["instrument_baselines"], (
        f"only one family has item records, got {load.available_families('items')}"
    )


def test_records_are_read_in_the_order_of_the_file(released):
    rows = list(load.iter_records("instrument_baselines", "pairs"))
    assert [row["pair_id"] for row in rows] == ["p1", "p2", "p3", "p4"], (
        f"unexpected order: {[row['pair_id'] for row in rows]}"
    )


def test_a_missing_file_and_an_unknown_kind_are_named(released):
    with pytest.raises(ValueError, match="unknown record kind"):
        load.family_path("instrument_baselines", "scores")
    with pytest.raises(FileNotFoundError, match="no items file for extra_family"):
        load.family_path("extra_family", "items")


def ids(rows):
    return [row.get("pair_id") or row.get("item_id") for row in rows]


def test_filters_accept_a_value_a_collection_or_a_test(released):
    rows = load.load_pairs("instrument_baselines")
    assert ids(load.select(rows, run="run-a")) == ["p1", "p2", "p3"], (
        "filtering on one value did not keep the three rows of that run"
    )
    assert ids(load.select(rows, layer=("l3", "probe"), seed=1)) == ["p1", "p2"], (
        "filtering on two fields did not keep the rows that satisfy both"
    )
    assert ids(load.select(rows, seed=lambda value: value > 1)) == ["p3", "p4"], (
        "filtering with a test did not keep the later seeds"
    )
    assert load.select(rows, run="run-z") == [], "an unmatched filter should keep nothing"
    assert load.select(rows, missing_field=None) == rows, (
        "a filter on an absent field compares against None and should keep every row"
    )


def test_grouping_takes_one_key_or_several(released):
    rows = load.load_pairs()
    by_run = load.group_by(rows, "run")
    assert sorted(by_run) == ["run-a", "run-b", "run-c"], f"unexpected groups: {sorted(by_run)}"
    assert ids(by_run["run-a"]) == ["p1", "p2", "p3"], "a group did not keep its rows in file order"
    by_run_and_seed = load.group_by(rows, "run", "seed")
    assert sorted(by_run_and_seed) == [("run-a", 1), ("run-a", 2), ("run-b", 2), ("run-c", 1)], (
        f"unexpected keys: {sorted(by_run_and_seed)}"
    )
    assert sum(len(group) for group in by_run_and_seed.values()) == len(rows), (
        "grouping lost or duplicated rows"
    )
    with pytest.raises(ValueError, match="at least one key"):
        load.group_by(rows)


def test_loading_reads_one_family_several_or_all_of_them(released):
    assert len(load.load_pairs("instrument_baselines")) == 4, "one family holds four pairs"
    assert len(load.load_pairs()) == 5, "both families together hold five pairs"
    assert len(load.load_pairs(["instrument_baselines", "extra_family"])) == 5, (
        "naming both families should read both"
    )
    filtered = load.load_pairs("instrument_baselines", seed=1, pair_correct=True)
    assert [row["pair_id"] for row in filtered] == ["p1"], (
        f"the filters were not applied while reading: {[row['pair_id'] for row in filtered]}"
    )
    assert len(load.load_items()) == 2, "one family holds two items"
    assert load.load_audits() == [], "no family holds audit records"


def test_the_manifest_describes_the_files_on_disk(released):
    rows = load.manifest_rows()
    assert sorted((row["family"], row["kind"]) for row in rows) == [
        ("extra_family", "pairs"),
        ("instrument_baselines", "items"),
        ("instrument_baselines", "pairs"),
    ], f"unexpected manifest rows: {[(row['family'], row['kind']) for row in rows]}"
    checked = load.verify()
    assert all(row["matches"] for row in checked), (
        f"a file does not match its manifest entry: {[row for row in checked if not row['matches']]}"
    )
    assert load.file_sha256("instrument_baselines", "pairs") == load.manifest()[
        "instrument_baselines"
    ]["pairs"]["sha256"], "the hash on disk differs from the recorded one"


def test_a_changed_file_fails_the_manifest_check(released):
    write_records(released / "extra_family" / "pairs.jsonl.gz", OTHER_PAIRS + PAIRS)
    checked = {(row["family"], row["kind"]): row for row in load.verify()}
    assert checked[("extra_family", "pairs")]["matches"] is False, (
        "a rewritten file should not match its manifest entry"
    )
    assert checked[("instrument_baselines", "pairs")]["matches"] is True, (
        "the untouched file should still match"
    )


def test_a_missing_manifest_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv("LWL_PREDICTIONS", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no manifest"):
        load.manifest()


PUBLISHED_MANIFEST = (predictions_root() / load.MANIFEST_NAME).exists()
published = pytest.mark.skipif(not PUBLISHED_MANIFEST, reason="the released outputs are not downloaded")


@published
def test_the_published_files_hold_the_records_the_manifest_counts():
    entries = load.manifest_rows()
    assert entries, "the published manifest lists no files"
    problems = []
    for entry in entries:
        name = f"{entry['family']}/{entry['kind']}"
        try:
            path = load.family_path(entry["family"], entry["kind"])
        except (FileNotFoundError, ValueError) as error:
            problems.append(f"{name}: {error}")
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            found = sum(1 for line in handle if line.strip())
        if found != entry["records"]:
            problems.append(f"{name}: {found} records, the manifest says {entry['records']}")
    assert not problems, "; ".join(problems)


@published
def test_the_published_files_match_their_recorded_hashes():
    mismatched = [
        f"{row['family']}/{row['kind']}: {row['problem'] or 'sha256 differs'}"
        for row in load.verify()
        if not row["matches"]
    ]
    assert not mismatched, "; ".join(mismatched)

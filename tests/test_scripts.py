"""Command-line entry points: the rebuild of the paper's numbers, the checkpoint merger and the
placement of the upstream ViRL39K images."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile

import pytest

from lwl.analysis.paper import (
    INPUT_MISSING,
    MATCHED,
    MATCHED_AFTER_ROUNDING,
    MISMATCH,
    NOT_REBUILDABLE,
    PAPER_ISSUE,
    comparison_outcome,
    registry,
)
from lwl.paths import predictions_root, repo_root


REBUILD_SCRIPT = repo_root() / "scripts" / "reproduce.py"


@pytest.mark.skipif(not REBUILD_SCRIPT.exists(), reason="the scripts directory is not next to the package")
def test_the_rebuild_script_lists_its_targets():
    result = subprocess.run(
        [sys.executable, str(REBUILD_SCRIPT), "--list"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"listing the targets exited {result.returncode}; stderr: {result.stderr.strip()}"
    )
    listed = [line for line in result.stdout.splitlines() if line.strip()]
    assert listed and listed != ["no targets available"], (
        f"no target was listed; stderr: {result.stderr.strip()}"
    )
    skipped = [line for line in result.stderr.splitlines() if line.startswith("skipped ")]
    assert not skipped, f"some table modules failed to import: {skipped}"


def run_script(name, *args, env=None, cwd=None):
    return subprocess.run(
        [sys.executable, str(repo_root() / "scripts" / name), *args],
        cwd=cwd or repo_root(), env=env, capture_output=True, text=True, timeout=600,
    )


def test_the_checkpoint_merger_answers_help_without_the_trainer(tmp_path):
    env = dict(os.environ, LWL_EASYR1=str(tmp_path / "no_trainer_here"))
    result = run_script("merge_checkpoint.py", "--help", env=env)
    assert result.returncode == 0, f"--help exited {result.returncode}: {result.stderr.strip()}"
    assert "Merge a sharded trainer checkpoint" in result.stdout, f"unexpected help text: {result.stdout!r}"
    missing = run_script("merge_checkpoint.py", "--local_dir", str(tmp_path), env=env)
    assert missing.returncode != 0 and "setup_easyr1.sh" in missing.stderr, (
        f"a merge without the trainer should fail and say how to get it: {missing.stderr!r}"
    )


@pytest.mark.parametrize(
    ("row", "outcome"),
    [
        ({"value": 0.1234, "paper_value": 0.123}, MATCHED),
        ({"value": 0.1240, "paper_value": 0.123}, MISMATCH),
        ({"value": None, "paper_value": 0.123}, MISMATCH),
        ({"value": 0.0533, "paper_value": 0.054, "relation": "le"}, MATCHED),
        ({"value": 0.0541, "paper_value": 0.054, "relation": "le"}, MISMATCH),
        ({"value": -0.005, "paper_value": 0.0, "relation": "lt"}, MATCHED),
        ({"value": 0.0004, "paper_value": 0.0, "relation": "zero"}, MISMATCH),
        ({"value": None, "paper_value": "undefined", "relation": "undefined"}, MATCHED),
        ({"value": 1.371, "paper_value": "undefined", "relation": "undefined"}, MISMATCH),
        ({"value": 0.11148, "paper_value": 0.112, "paper_rounding": 0.112}, MATCHED_AFTER_ROUNDING),
        ({"value": 0.11148, "paper_value": 0.113, "paper_rounding": 0.112}, MISMATCH),
        ({"value": 0.2, "paper_value": 0.2, "ci_low": 0.1, "paper_ci_low": 0.12}, MISMATCH),
        ({"value": None, "paper_value": 13, "status": NOT_REBUILDABLE}, NOT_REBUILDABLE),
        ({"value": None, "paper_value": 0.126, "status": INPUT_MISSING}, INPUT_MISSING),
        ({"value": 0.3, "paper_value": 0.2, "paper_issue": "printed from another run"}, PAPER_ISSUE),
        ({"value": 0.3}, None),
    ],
)
def test_printed_values_are_compared_as_the_paper_states_them(row, outcome):
    assert comparison_outcome(row) == outcome, f"{row} should compare as {outcome}"


RELEASED_BASELINES = predictions_root() / "instrument_baselines" / "pairs.jsonl.gz"
RELEASED_PREDICTIONS = predictions_root() / "manifest.json"


@pytest.mark.skipif(not RELEASED_BASELINES.exists(), reason="the released predictions are not downloaded")
def test_the_grounding_table_is_rebuilt_to_the_printed_values(tmp_path):
    env = dict(os.environ, LWL_RESULTS=str(tmp_path))
    result = run_script("reproduce.py", "tableC1", "--check", env=env)
    assert result.returncode == 0, (
        f"tableC1 did not reproduce:\n{result.stdout}{result.stderr}"
    )
    assert (tmp_path / "tableC1.csv").exists(), "the rebuilt table was not written to LWL_RESULTS"


def _released_targets() -> list[str]:
    return sorted(registry()) if RELEASED_PREDICTIONS.exists() else []


@pytest.mark.skipif(not RELEASED_PREDICTIONS.exists(), reason="the released predictions are not downloaded")
@pytest.mark.parametrize("name", _released_targets())
def test_every_printed_value_is_rebuilt_or_explained(name):
    """Each printed value matches, matches after the paper's rounding, or carries its reason."""
    try:
        rows = registry()[name]()
    except FileNotFoundError as error:
        pytest.skip(f"{name} needs a file that is not present: {error}")
    outcomes = [comparison_outcome(row) for row in rows]
    if INPUT_MISSING in outcomes:
        pytest.skip(f"{name} needs inputs that are not unpacked under LWL_DATA")
    unexplained = [row for row, outcome in zip(rows, outcomes) if outcome == MISMATCH]
    assert not unexplained, (
        f"{name}: {len(unexplained)} printed values are not rebuilt:\n"
        + "\n".join(json.dumps(row, default=str) for row in unexplained[:10])
    )


def test_the_virl39k_images_are_placed_and_checked(tmp_path):
    images = {"images/one.png": b"first image", "images/two.jpg": b"second image", "images/three.png": b"third"}
    digests = {name: hashlib.sha256(blob).hexdigest() for name, blob in images.items()}
    rows = [
        {"qid": "q1", "images": ["data/virl39k/images/one.png"],
         "metadata": {"relative_image_paths": ["images/one.png"], "image_sha256": [digests["images/one.png"]]}},
        {"qid": "q2", "images": ["data/virl39k/images/two.jpg", "data/virl39k/images/three.png"],
         "metadata": {"relative_image_paths": ["images/two.jpg", "images/three.png"],
                      "image_sha256": [digests["images/two.jpg"], digests["images/three.png"]]}},
    ]
    rows_path = tmp_path / "filtered_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        for name, blob in images.items():
            archive.writestr(name, blob)
    dest = tmp_path / "virl39k"

    placed = run_script("fetch_virl39k.py", "--rows", str(rows_path), "--archive", str(good), "--dest", str(dest))
    assert placed.returncode == 0, f"placing the images failed: {placed.stdout}{placed.stderr}"
    for name, blob in images.items():
        assert (dest / name).read_bytes() == blob, f"{name} was not placed"
    assert json.loads(placed.stdout)["placed"] == 3, f"unexpected summary: {placed.stdout}"
    again = run_script("fetch_virl39k.py", "--rows", str(rows_path), "--archive", str(good), "--dest", str(dest))
    assert json.loads(again.stdout)["already_present"] == 3, f"verified files should be kept: {again.stdout}"

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("images/one.png", b"not the recorded image")
        archive.writestr("images/two.jpg", images["images/two.jpg"])
    fresh = tmp_path / "fresh"
    checked = run_script("fetch_virl39k.py", "--rows", str(rows_path), "--archive", str(bad), "--dest", str(fresh))
    summary = json.loads(checked.stdout)
    assert checked.returncode == 1, "a wrong or missing image should fail the run"
    assert (summary["sha256_mismatch"], summary["missing_from_archive"], summary["placed"]) == (1, 1, 1), (
        f"one image differs from its digest and one is absent: {summary}"
    )
    assert not (fresh / "images" / "one.png").exists(), "an image that fails its digest must not be written"

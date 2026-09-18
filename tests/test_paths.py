"""How a path recorded in a manifest or corpus row is found on disk."""
from __future__ import annotations

from pathlib import Path

from lwl.paths import data_path, repo_root, resolve_data_reference

REFERENCE = "data/grounding/suite/images/0001.png"


def place(root: Path, reference: str = REFERENCE) -> Path:
    path = root / reference
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    return path


def test_lwl_data_wins_over_a_copy_under_the_current_directory(tmp_path, monkeypatch):
    data = tmp_path / "elsewhere"
    expected = place(data, REFERENCE.removeprefix("data/"))
    workdir = tmp_path / "work"
    place(workdir)
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("LWL_DATA", str(data))
    found = resolve_data_reference(REFERENCE)
    assert found == expected.resolve(), f"with LWL_DATA set, {REFERENCE} resolved to {found}"
    assert resolve_data_reference(REFERENCE, base=workdir) == expected.resolve(), (
        "a copy beside the base should not win over LWL_DATA either"
    )


def test_lwl_data_holds_a_reference_even_before_the_file_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    place(tmp_path)
    monkeypatch.setenv("LWL_DATA", str(tmp_path / "empty"))
    found = resolve_data_reference(REFERENCE)
    assert found == (tmp_path / "empty").resolve() / REFERENCE.removeprefix("data/"), (
        f"a missing file should be looked for under LWL_DATA, not found at {found}"
    )


def test_without_lwl_data_a_reference_resolves_as_written_then_under_the_data_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("LWL_DATA", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_data_reference(REFERENCE) == data_path("grounding", "suite", "images", "0001.png"), (
        "a missing data/ reference should fall back to the data directory"
    )
    place(tmp_path)
    assert resolve_data_reference(REFERENCE) == Path(REFERENCE), (
        "an existing relative path should be used as written"
    )


def test_a_reference_beside_its_manifest_and_other_references(tmp_path, monkeypatch):
    monkeypatch.delenv("LWL_DATA", raising=False)
    monkeypatch.chdir(tmp_path)
    beside = place(tmp_path / "manifest_dir", "images/0001.png")
    assert resolve_data_reference("images/0001.png", base=tmp_path / "manifest_dir") == beside, (
        "a path relative to the manifest should resolve beside it"
    )
    assert resolve_data_reference("configs/train") == repo_root() / "configs" / "train", (
        "a reference outside data/ should resolve against the repository"
    )
    absolute = tmp_path / "absolute.png"
    monkeypatch.setenv("LWL_DATA", str(tmp_path / "data"))
    assert resolve_data_reference(str(absolute)) == absolute, "an absolute path is used as it is"

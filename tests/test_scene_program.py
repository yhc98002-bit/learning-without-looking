"""The coordinate scene program: determinism, the four cue levels, the twin pair and the labels,
and a regenerated scene against its published row (one shipped row, plus the first row of every
published scene set that is present)."""
from __future__ import annotations

import importlib.util
import json
import math
import random
import tarfile

import numpy as np
import pytest

from lwl.grounding.build_suite import COLORS
from lwl.evaluation.metrics import match_tier
from lwl.paths import data_path, repo_root
from lwl.scenes.program import (
    COORD_ALLOWED,
    EXTREMUM_KINDS,
    EXTREMUM_ROTATION,
    LABEL_POOL,
    SPLIT_BUCKETS,
    build_coord_geometry,
    coord_extremum,
    coord_hard_negatives,
    coord_read,
    coord_target_px,
    cue_ink_disjoint,
    render_coord_layers,
    render_coordinate_scene,
    split_of,
)

ROLES = ("target_switch", "target_stable", "invariance")
KIND = "largest_y"
N_POINTS = 12


def geometry_for(role, kind=KIND, n_points=N_POINTS):
    """The first scene the program accepts for this role, with the seed that produced it."""
    for seed in range(400):
        built = build_coord_geometry(role, kind, n_points, random.Random(seed))
        if built is not None:
            return seed, built
    raise AssertionError(f"no {role} scene was accepted in 400 attempts")


def test_one_seed_gives_one_scene():
    seed, first = geometry_for("target_stable")
    second = build_coord_geometry("target_stable", KIND, N_POINTS, random.Random(seed))
    assert second == first, f"seed {seed} gave two different scenes"
    one = np.asarray(render_coordinate_scene(first["points_a"]))
    other = np.asarray(render_coordinate_scene(first["points_a"]))
    assert np.array_equal(one, other), "the same points rendered to two different images"


def test_a_different_seed_gives_a_different_scene():
    _, first = geometry_for("target_stable")
    others = [
        build_coord_geometry("target_stable", KIND, N_POINTS, random.Random(seed))
        for seed in range(400, 410)
    ]
    built = [scene for scene in others if scene is not None]
    assert built, "no scene was accepted for any of ten further seeds"
    assert all(scene["points_a"] != first["points_a"] for scene in built), (
        "another seed reproduced the same points"
    )


def labels_named_in(text, points):
    return sorted(label for label in points if label in text)


@pytest.mark.parametrize("kind", EXTREMUM_ROTATION)
def test_the_four_cue_levels_agree_on_the_target(kind):
    _, geometry = geometry_for("target_stable", kind=kind)
    for side, questions in (("a", geometry["questions"]), ("b", geometry["questions_b_target"])):
        points, target = geometry[f"points_{side}"], geometry[f"target_{side}"]
        discovered = coord_extremum(points, kind)[0]
        assert discovered == target, (
            f"member {side}: discovery and the identification probe resolve to {discovered}, "
            f"the scene's target is {target}"
        )
        assert labels_named_in(questions["l3"], points) == [], (
            f"member {side}: the discovery question names a point: {questions['l3']!r}"
        )
        assert labels_named_in(questions["probe"], points) == [], (
            f"member {side}: the identification question names a point: {questions['probe']!r}"
        )
        assert labels_named_in(questions["l2"], points) == [target], (
            f"member {side}: the find-and-bind question should name {target} alone: {questions['l2']!r}"
        )
        answer = coord_read(points, target, kind)
        assert answer == geometry[f"answer_{side}"], (
            f"member {side}: the target reads {answer}, the scene records {geometry[f'answer_{side}']}"
        )
        axis_name = EXTREMUM_KINDS[kind]["read_name"]
        asked = f"{axis_name}-coordinate"
        assert asked in questions["l3"] and asked in questions["l2"], (
            f"member {side}: discovery and find-and-bind should ask for the same {axis_name}-coordinate"
        )

        layers = render_coord_layers(points, target)
        assert layers is not None, f"member {side}: no cue could be placed on an accepted scene"
        uncued, cued, record = layers
        for index, label in enumerate(points):
            drawn = uncued.getpixel(coord_target_px(points, label))
            assert drawn == COLORS[index % len(COLORS)], (
                f"member {side}: the pixel given for {label} shows {drawn}, not the colour of its dot"
            )
        assert cue_ink_disjoint(uncued, cued, COORD_ALLOWED), (
            f"member {side}: the cue overdrew something other than the background"
        )
        left, top, right, bottom = record["cue_bbox"]
        centre = ((left + right) / 2, (top + bottom) / 2)
        nearest = min(points, key=lambda label: math.dist(centre, coord_target_px(points, label)))
        assert nearest == target, (
            f"member {side}: the cued readout points at {nearest}, not the target {target}"
        )


def accepted_scenes(role, seeds=range(40)):
    """Every scene the program accepts for this role over a range of seeds and all four kinds."""
    scenes = []
    for kind in EXTREMUM_ROTATION:
        for seed in seeds:
            built = build_coord_geometry(role, kind, N_POINTS, random.Random(seed))
            if built is not None:
                scenes.append((f"{kind}/seed {seed}", built))
    assert len(scenes) >= len(seeds), (
        f"only {len(scenes)} {role} scenes were accepted from {4 * len(seeds)} tries, fewer than one in four"
    )
    return scenes


@pytest.mark.parametrize("role", ROLES)
def test_a_twin_pair_differs_in_exactly_one_point(role):
    for name, geometry in accepted_scenes(role):
        points_a, points_b, kind = geometry["points_a"], geometry["points_b"], geometry["extremum_kind"]
        assert list(points_a) == list(points_b), f"{name}: the two members carry different labels"
        moved = [label for label in points_a if points_a[label] != points_b[label]]
        assert moved == [geometry["moved_label"]], (
            f"{name}: exactly the recorded point {geometry['moved_label']} should move, {moved} did"
        )
        for side in ("a", "b"):
            points = geometry[f"points_{side}"]
            target, gap = coord_extremum(points, kind)
            assert target == geometry[f"target_{side}"] and gap >= 1, (
                f"{name}: member {side} should have {geometry[f'target_{side}']} as a clear extremum, "
                f"the points give {target} with a margin of {gap}"
            )
            assert coord_read(points, target, kind) == geometry[f"answer_{side}"], (
                f"{name}: member {side} records an answer that is not its target's coordinate"
            )


def test_a_target_switch_hands_the_extremum_to_the_runner_up():
    for name, scene in accepted_scenes("target_switch"):
        axis = EXTREMUM_KINDS[scene["extremum_kind"]]["axis"]
        best = EXTREMUM_KINDS[scene["extremum_kind"]]["best"]
        others = {label: point for label, point in scene["points_a"].items() if label != scene["target_a"]}
        runner_up = best(sorted(others), key=lambda label: others[label][axis])
        assert scene["moved_label"] == scene["target_a"], f"{name}: the target itself should move"
        assert scene["points_b"][scene["target_b"]] == scene["points_a"][scene["target_b"]], (
            f"{name}: the new target should stay where it was"
        )
        assert scene["points_a"][scene["target_b"]][axis] == others[runner_up][axis], (
            f"{name}: the new target {scene['target_b']} was not the runner-up of the first member"
        )
        answer_a, answer_b = scene["answer_a"], scene["answer_b"]
        crossed = (match_tier(answer_a, answer_b), match_tier(answer_b, answer_a))
        assert crossed == (0, 0), (
            f"{name}: the answers {scene['answer_a']} and {scene['answer_b']} could be matched to each other"
        )


def test_a_target_stable_pair_moves_only_the_read_coordinate():
    for name, scene in accepted_scenes("target_stable"):
        spec = EXTREMUM_KINDS[scene["extremum_kind"]]
        target = scene["target_a"]
        before, after = scene["points_a"][target], scene["points_b"][target]
        assert scene["moved_label"] == target == scene["target_b"], (
            f"{name}: the target should move and stay the target"
        )
        assert before[spec["axis"]] == after[spec["axis"]], (
            f"{name}: the target's extremum coordinate changed from {before} to {after}"
        )
        assert abs(before[spec["read"]] - after[spec["read"]]) >= 3, (
            f"{name}: the read coordinate moved from {before} to {after}, less than three steps"
        )


def test_an_invariance_pair_moves_a_distractor_and_keeps_the_answer():
    for name, scene in accepted_scenes("invariance"):
        target = scene["target_a"]
        assert scene["moved_label"] != target == scene["target_b"], (
            f"{name}: a point other than the target should move, {scene['moved_label']} did"
        )
        assert scene["points_a"][target] == scene["points_b"][target], f"{name}: the target moved"
        assert scene["answer_a"] == scene["answer_b"], (
            f"{name}: the answer changed from {scene['answer_a']} to {scene['answer_b']}"
        )


@pytest.mark.parametrize("role", ROLES)
def test_labels_do_not_collide(role):
    for name, geometry in accepted_scenes(role, seeds=range(10)):
        labels = geometry["labels"]
        assert len(set(labels)) == len(labels) == N_POINTS, (
            f"{name}: {len(labels)} labels with {len(set(labels))} distinct values"
        )
        assert set(labels) <= set(LABEL_POOL), (
            f"{name}: labels outside the pool: {sorted(set(labels) - set(LABEL_POOL))}"
        )
        contained = [(one, other) for one in labels for other in labels if one != other and one in other]
        assert not contained, (
            f"{name}: a label occurs inside another, which a text matcher could confuse: {contained}"
        )
        for side in ("points_a", "points_b"):
            items = sorted(geometry[side].items())
            for index, (label, position) in enumerate(items):
                for other_label, other in items[index + 1:]:
                    gap = max(abs(position[0] - other[0]), abs(position[1] - other[1]))
                    assert gap >= 2, (
                        f"{name}: {label} and {other_label} are {gap} grid step apart in {side}, "
                        "close enough for their drawn labels to overlap"
                    )


def test_the_split_follows_the_scene_alone():
    ids = [f"scene_{index:06d}" for index in range(300)]
    splits = [split_of(one) for one in ids]
    assert set(splits) == set(SPLIT_BUCKETS), (
        f"300 scenes should reach every split, got {sorted(set(splits))}"
    )
    assert [split_of(one) for one in ids] == splits, "the split of a scene changed between calls"


BUILD_SCRIPT = repo_root() / "scripts" / "build_scenes.py"
CELL = "n8"
# The first discovery row of the training set's 8-point cell, copied from the published manifest.
SHIPPED_ROW = repo_root() / "tests" / "fixtures" / "scene_training_n8_l3_first.jsonl"


def is_manifest(name, layer):
    """A split's per-cell manifest, as the build names it: manifest_<family>_<cell>_<layer>.jsonl."""
    return name.startswith("manifest_") and name.endswith(f"_{CELL}_{layer}.jsonl")


def published_available(split):
    unpacked = data_path("scenes", split)
    return (unpacked.is_dir() and any(is_manifest(path.name, "l3") for path in unpacked.iterdir())) or (
        data_path("scenes", f"{split}.tar.gz").exists()
    )


def first_published_rows(split, layers):
    """First row of each named layer's manifest, from the unpacked split or straight from its archive."""
    rows = {}
    unpacked = data_path("scenes", split)
    if unpacked.is_dir():
        for path in sorted(unpacked.iterdir()):
            for layer in layers:
                if layer not in rows and is_manifest(path.name, layer):
                    with path.open(encoding="utf-8") as handle:
                        rows[layer] = json.loads(handle.readline())
    if len(rows) < len(layers):
        with tarfile.open(data_path("scenes", f"{split}.tar.gz"), "r|gz") as archive:
            for member in archive:
                directory, _, name = member.name.rpartition("/")
                for layer in layers:
                    if layer not in rows and directory == split and is_manifest(name, layer):
                        rows[layer] = json.loads(archive.extractfile(member).readline())
                if len(rows) == len(layers):
                    break
    missing = sorted(set(layers) - set(rows))
    assert not missing, f"the {split} scene set has no manifest for {missing}"
    return rows


def load_build_script():
    spec = importlib.util.spec_from_file_location("build_scenes", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regenerate_first_scene(build, split):
    """The first scene of the cell: the first target-switch attempt whose program lands in `split`.

    The build draws the side swap from the same generator right after the geometry, and the first
    scene of a cell cannot be rejected for answer balance.
    """
    n_points = dict(build.CELLS)[CELL]
    kind = EXTREMUM_ROTATION[0]
    for attempt in range(1, 20000):
        rng = build.attempt_rng(CELL, "target_switch", attempt)
        geometry = build_coord_geometry("target_switch", kind, n_points, rng)
        if geometry is None:
            continue
        program_id = build.scene_program_id(sorted(geometry["points_a"].items()))
        if split_of(program_id) == split:
            return geometry, program_id, rng.random() < 0.5
    raise AssertionError(f"no {split} scene was generated for cell {CELL}")


def assert_regenerated(split, row, targets):
    """Regenerate the first scene of the cell and compare it with its published discovery row;
    `targets` maps each published member to the label its identification probe answers."""
    build = load_build_script()
    geometry, program_id, swapped = regenerate_first_scene(build, split)
    first, second = ("b", "a") if swapped else ("a", "b")
    item = row["mother_item_id"]

    assert program_id == row["scene_program_id"], (
        f"{split}: regenerated scene program {program_id}, the published row has {row['scene_program_id']}"
    )
    assert item.endswith(f"_{CELL}_target_switch_{program_id[-12:]}"), (
        f"{split}: the regenerated scene {program_id} would carry another item id than {item}"
    )
    assert swapped == row["provenance"]["semantic_side_assignment_swapped"], (
        f"{split}: {item} regenerated with side swap {swapped}, the published row says "
        f"{row['provenance']['semantic_side_assignment_swapped']}"
    )
    for published_side, generated_side in (("a", first), ("b", second)):
        points = sorted(geometry[f"points_{generated_side}"].items())
        rebuilt = [[label, x, y] for label, (x, y) in points]
        assert rebuilt == row[f"scene_{published_side}"], (
            f"{split}: member {published_side} of {item} regenerated with different points"
        )
        assert geometry[f"answer_{generated_side}"] == row[f"answer_{published_side}"], (
            f"{split}: member {published_side} of {item} regenerated with answer "
            f"{geometry[f'answer_{generated_side}']}, published {row[f'answer_{published_side}']}"
        )
        assert geometry[f"target_{generated_side}"] == targets[published_side], (
            f"{split}: the identification probe of {item} names {targets[published_side]} "
            f"for member {published_side}, the regenerated target is {geometry[f'target_{generated_side}']}"
        )
    assert row["verifier_results"]["extremum_kind"] == geometry["extremum_kind"], (
        f"{split}: {item} was published as {row['verifier_results']['extremum_kind']}, "
        f"regenerated as {geometry['extremum_kind']}"
    )
    assert geometry["questions"]["l3"] == row["question"], (
        f"{split}: the discovery question was regenerated as {geometry['questions']['l3']!r}, "
        f"published as {row['question']!r}"
    )
    assert coord_hard_negatives(geometry) == row["hard_negatives"], (
        f"{split}: the wrong answers regenerated for {item} differ from the published ones"
    )


def test_the_shipped_scene_row_is_regenerated_exactly():
    row = json.loads(SHIPPED_ROW.read_text(encoding="utf-8"))
    assert (row["split"], row["cell"], row["layer"]) == ("training", CELL, "l3"), (
        f"the shipped row is not the training {CELL} discovery row: {row['pair_id']}"
    )
    verified = row["verifier_results"]
    assert_regenerated("training", row, {"a": verified["target_label_a"], "b": verified["target_label_b"]})


@pytest.mark.skipif(not BUILD_SCRIPT.exists(), reason="the scripts directory is not next to the package")
@pytest.mark.parametrize(
    "split",
    [
        pytest.param(split, marks=pytest.mark.skipif(
            not published_available(split),
            reason=f"the published {split} scenes are not in the data directory",
        ))
        for split in SPLIT_BUCKETS
    ],
)
def test_a_published_scene_is_regenerated_exactly(split):
    rows = first_published_rows(split, ("l3", "probe"))
    probe = rows["probe"]
    assert_regenerated(split, rows["l3"], {side: probe[f"answer_{side}"] for side in ("a", "b")})

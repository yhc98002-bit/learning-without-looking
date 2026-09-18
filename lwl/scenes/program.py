"""Scene programs for the constructed coordinate scenes.

One program produces a twin pair of labelled point scenes (a and b) carrying a
target-stable, target-switch or invariance role, the questions for each cue
level, and the offset cue that turns a find-and-bind image into a cued readout.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from lwl.grounding.build_suite import (
    COLORS as _COLORS,
    _answers_distinguishable,
    _font,
    _sample_high_entropy_points,
)

CUE_COLOR = (196, 30, 58)
# Diagonal-first: scene points sit on gridline intersections, and a 45-degree
# ray whose axis offsets stay inside one grid cell crosses no gridline.
CUE_DIRECTIONS = (
    (1, 1), (-1, -1), (1, -1), (-1, 1), (1, 0), (-1, 0), (0, 1), (0, -1),
)
CUE_RADII = ((18, 66), (18, 52), (24, 78))


def _draw_cue(draw: ImageDraw.ImageDraw, target: tuple[int, int],
              direction: tuple[int, int], radii: tuple[int, int]) -> None:
    norm = math.hypot(*direction)
    ux, uy = direction[0] / norm, direction[1] / norm
    near, far = radii
    tip = (round(target[0] + ux * near), round(target[1] + uy * near))
    tail = (round(target[0] + ux * far), round(target[1] + uy * far))
    draw.line((tail, tip), fill=CUE_COLOR, width=3)
    # open arrowhead: two wings back from the tip
    for wing_sign in (1, -1):
        angle = math.atan2(uy, ux) + wing_sign * 0.45
        wing = (
            round(tip[0] + math.cos(angle) * 14),
            round(tip[1] + math.sin(angle) * 14),
        )
        draw.line((tip, wing), fill=CUE_COLOR, width=3)


def add_offset_cue(
    base: Image.Image,
    target_px: tuple[int, int],
    allowed_colors: frozenset[tuple[int, int, int]],
) -> tuple[Image.Image, dict[str, Any]] | None:
    """Return (cued image, cue record), or None when no placement leaves the
    existing ink intact. The cue may only overdraw background or plot fill, the
    search order is deterministic, and a cue is never forced."""
    base_array = np.asarray(base, dtype=np.uint8)
    for radii in CUE_RADII:
        for direction in CUE_DIRECTIONS:
            candidate = base.copy()
            _draw_cue(ImageDraw.Draw(candidate), target_px, direction, radii)
            cand_array = np.asarray(candidate, dtype=np.uint8)
            changed = np.any(cand_array != base_array, axis=2)
            if not changed.any():
                continue
            replaced = base_array[changed]
            replaced_colors = {tuple(int(v) for v in px) for px in replaced}
            if replaced_colors <= allowed_colors:
                ys, xs = np.nonzero(changed)
                return candidate, {
                    "cue_direction": list(direction),
                    "cue_radii": list(radii),
                    "cue_pixel_count": int(changed.sum()),
                    "cue_bbox": [int(xs.min()), int(ys.min()),
                                 int(xs.max()), int(ys.max())],
                    "cue_ink_disjoint": True,
                }
    return None


def cue_ink_disjoint(l2_image: Image.Image, l1_image: Image.Image,
                     allowed_colors: frozenset[tuple[int, int, int]]) -> bool:
    """From-disk re-check that the cue replaced only allowed colours."""
    a2 = np.asarray(l2_image, dtype=np.uint8)
    a1 = np.asarray(l1_image, dtype=np.uint8)
    changed = np.any(a1 != a2, axis=2)
    if not changed.any():
        return False  # a cued image with no cue at all is a violation, not a pass
    replaced = {tuple(int(v) for v in px) for px in a2[changed]}
    return replaced <= allowed_colors


SPLIT_BUCKETS = {"training": (0, 60), "development": (60, 80),
                 "confirmatory": (80, 100)}


def split_of(scene_program_id: str) -> str:
    """Split a scene program belongs to, hashed from its id alone."""
    bucket = int(hashlib.sha256(
        (scene_program_id + "|split-v1").encode()).hexdigest()[:8], 16) % 100
    for name, (lo, hi) in SPLIT_BUCKETS.items():
        if lo <= bucket < hi:
            return name
    raise AssertionError(bucket)


COORD_ALLOWED = frozenset({(250, 250, 248), (255, 255, 255)})
COORD_ORIGIN, COORD_SCALE = (700, 650), 68
LABEL_POOL = [f"{letter}{digit}"
              for letter in "BCDFGHJKLMNPRSTVWXYZ" for digit in "23456789"]

EXTREMUM_KINDS = {
    "largest_y": {"axis": 1, "best": max, "read": 0,
                  "phrase": "largest y-coordinate", "read_name": "x"},
    "smallest_y": {"axis": 1, "best": min, "read": 0,
                   "phrase": "smallest y-coordinate", "read_name": "x"},
    "leftmost": {"axis": 0, "best": min, "read": 1,
                 "phrase": "smallest x-coordinate", "read_name": "y"},
    "rightmost": {"axis": 0, "best": max, "read": 1,
                  "phrase": "largest x-coordinate", "read_name": "y"},
}
EXTREMUM_ROTATION = ("largest_y", "rightmost", "smallest_y", "leftmost")
COORD_MARGIN = 1  # extremum-axis gap between the top two points, both sides


def _spacing_ok(candidate: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    return all(max(abs(candidate[0] - x), abs(candidate[1] - y)) >= 2
               for x, y in others)


def base_scene(rng: random.Random,
               n_points: int = 20) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Labels and spaced integer positions for one side of a scene pair."""
    labels = rng.sample(LABEL_POOL, n_points)
    coordinates = _sample_high_entropy_points(rng, n_points)
    return labels, dict(zip(labels, coordinates))


def coord_extremum(points: dict[str, tuple[int, int]], kind: str) -> tuple[str, int]:
    spec = EXTREMUM_KINDS[kind]
    axis, best = spec["axis"], spec["best"]
    ordered = sorted(points.items(), key=lambda kv: kv[1][axis],
                     reverse=(best is max))
    gap = abs(ordered[0][1][axis] - ordered[1][1][axis])
    return ordered[0][0], gap


def coord_questions(kind: str, target: str) -> dict[str, str]:
    """Question text per cue level: discovery, find-and-bind, identification."""
    spec = EXTREMUM_KINDS[kind]
    return {
        "l3": (f"Consider the point with the {spec['phrase']}. "
               f"What is its {spec['read_name']}-coordinate?"),
        "l2": (f"Point {target} has the {spec['phrase']}. "
               f"What is the {spec['read_name']}-coordinate of point {target}?"),
        "probe": f"Which labeled point has the {spec['phrase']}?",
    }


def coord_read(points: dict[str, tuple[int, int]], label: str, kind: str) -> str:
    return str(points[label][EXTREMUM_KINDS[kind]["read"]])


def _coord_move_candidates(points: dict[str, tuple[int, int]], label: str,
                           predicate) -> list[tuple[int, int]]:
    others = {p for l, p in points.items() if l != label}
    return [
        (x, y)
        for x in range(-7, 8) if x != 0
        for y in [*range(-7, 0), *range(1, 8)]
        if (x, y) != points[label]
        and _spacing_ok((x, y), others)
        and predicate((x, y))
    ]


def build_coord_geometry(role: str, kind: str, n_points: int,
                         rng) -> dict[str, Any] | None:
    """One scene pair's geometry, or None on any constraint rejection."""
    labels, points_a = base_scene(rng, n_points)
    spec = EXTREMUM_KINDS[kind]
    axis, read = spec["axis"], spec["read"]
    target_a, gap_a = coord_extremum(points_a, kind)
    if gap_a < COORD_MARGIN:
        return None
    answer_a = coord_read(points_a, target_a, kind)

    if role == "target_switch":
        ordered = sorted(points_a.items(), key=lambda kv: kv[1][axis],
                         reverse=(spec["best"] is max))
        runner_up = ordered[1][0]

        def becomes_runner_up_extremum(new_pos: tuple[int, int]) -> bool:
            moved = dict(points_a)
            moved[target_a] = new_pos
            new_target, new_gap = coord_extremum(moved, kind)
            return (
                new_target == runner_up
                and new_gap >= COORD_MARGIN
                and _answers_distinguishable(
                    answer_a, coord_read(moved, runner_up, kind))
            )

        candidates = _coord_move_candidates(points_a, target_a,
                                            becomes_runner_up_extremum)
        if not candidates:
            return None
        points_b = dict(points_a)
        points_b[target_a] = candidates[rng.randrange(len(candidates))]
        target_b = runner_up
        moved_label = target_a
    elif role == "target_stable":
        fixed_extremum = points_a[target_a][axis]

        def keeps_extremum_moves_read(new_pos: tuple[int, int]) -> bool:
            if new_pos[axis] != fixed_extremum:
                return False
            moved = dict(points_a)
            moved[target_a] = new_pos
            new_target, new_gap = coord_extremum(moved, kind)
            return (
                new_target == target_a
                and new_gap >= COORD_MARGIN
                and abs(new_pos[read] - points_a[target_a][read]) >= 3
                and _answers_distinguishable(
                    answer_a, str(new_pos[read]))
            )

        candidates = _coord_move_candidates(points_a, target_a,
                                            keeps_extremum_moves_read)
        if not candidates:
            return None
        points_b = dict(points_a)
        points_b[target_a] = candidates[rng.randrange(len(candidates))]
        target_b = target_a
        moved_label = target_a
    elif role == "invariance":
        pool = [l for l in labels if l != target_a]
        rng.shuffle(pool)
        points_b = None
        moved_label = None
        for distractor in pool:
            def preserves_everything(new_pos: tuple[int, int],
                                     _d=distractor) -> bool:
                moved = dict(points_a)
                moved[_d] = new_pos
                new_target, new_gap = coord_extremum(moved, kind)
                return (
                    new_target == target_a
                    and new_gap >= COORD_MARGIN
                    and max(abs(new_pos[0] - points_a[_d][0]),
                            abs(new_pos[1] - points_a[_d][1])) >= 3
                )

            candidates = _coord_move_candidates(points_a, distractor,
                                                preserves_everything)
            if candidates:
                points_b = dict(points_a)
                points_b[distractor] = candidates[rng.randrange(len(candidates))]
                moved_label = distractor
                break
        if points_b is None:
            return None
        target_b = target_a
    else:
        raise AssertionError(f"unknown role {role}")

    answer_b = coord_read(points_b, target_b, kind)
    if role == "invariance":
        if answer_a != answer_b:
            return None
    return {
        "family": "hier_coord_v1",
        "role": role,
        "extremum_kind": kind,
        "n_points": n_points,
        "labels": labels,
        "points_a": points_a,
        "points_b": points_b,
        "target_a": target_a,
        "target_b": target_b,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "moved_label": moved_label,
        "questions": coord_questions(kind, target_a),
        "questions_b_target": coord_questions(kind, target_b),
    }


# Exactly one title and one cue-level-neutral footer are drawn into a scene.
# In-image text must never state the task procedure or name a target, or the
# discovery and identification levels would carry their own answer.
COORD_TITLE = "Coordinate Survey Register"
COORD_FOOTER = "Each point is identified by its printed label."
SCENE_TEXT = {"hier_coord_v1": {"title": COORD_TITLE, "footer": COORD_FOOTER}}
# Any of these substrings in drawn text marks a task-procedure instruction.
PROCEDURE_TOKENS = ("locate", "requested", "then read", "first find")


def render_coordinate_scene(points: dict[str, tuple[int, int]]) -> Image.Image:
    """Render one scene. The layout is that of
    lwl.grounding.build_suite._render_high_entropy_coordinate_register, which
    stays untouched for the grounding suite; only the footer string differs."""
    width, height = 1400, 1240
    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 38), COORD_TITLE, anchor="mm",
              font=_font(26, True), fill=(25, 25, 25))
    origin = (700, 650)
    scale = 68
    plot_left = origin[0] - 7 * scale
    plot_right = origin[0] + 7 * scale
    plot_top = origin[1] - 7 * scale
    plot_bottom = origin[1] + 7 * scale
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom),
                   fill="white", outline=(75, 75, 75), width=2)
    for value in range(-7, 8):
        x = origin[0] + value * scale
        y = origin[1] - value * scale
        draw.line((x, plot_top, x, plot_bottom), fill=(224, 228, 232), width=1)
        draw.line((plot_left, y, plot_right, y), fill=(224, 228, 232), width=1)
        if value:
            draw.text((x, origin[1] + 19), str(value), anchor="mm",
                      font=_font(13), fill=(65, 65, 65))
            draw.text((origin[0] - 20, y), str(value), anchor="mm",
                      font=_font(13), fill=(65, 65, 65))
    draw.line((plot_left, origin[1], plot_right, origin[1]), fill=(40, 40, 40), width=3)
    draw.line((origin[0], plot_top, origin[0], plot_bottom), fill=(40, 40, 40), width=3)

    for index, (label, point) in enumerate(points.items()):
        x = origin[0] + point[0] * scale
        y = origin[1] - point[1] * scale
        color = _COLORS[index % len(_COLORS)]
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color,
                     outline="white", width=2)
        label_x = x + (17 if point[0] <= 0 else -17)
        label_y = y - 16
        draw.text(
            (label_x, label_y),
            label,
            anchor="lm" if point[0] <= 0 else "rm",
            font=_font(19, True),
            fill=(18, 18, 18),
            stroke_width=2,
            stroke_fill="white",
        )
    draw.text(
        (plot_left, 1186),
        COORD_FOOTER,
        font=_font(15),
        fill=(70, 70, 70),
    )
    return image


def coord_target_px(points: dict[str, tuple[int, int]], label: str) -> tuple[int, int]:
    x, y = points[label]
    return (COORD_ORIGIN[0] + x * COORD_SCALE, COORD_ORIGIN[1] - y * COORD_SCALE)


def render_coord_layers(points: dict[str, tuple[int, int]],
                        target: str) -> tuple[Image.Image, Image.Image, dict] | None:
    """(uncued image, cued image, cue record) or None if no legal cue placement."""
    base = render_coordinate_scene(points)
    cued = add_offset_cue(base, coord_target_px(points, target), COORD_ALLOWED)
    if cued is None:
        return None
    l1, record = cued
    return base, l1, record


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _nearest_gridline(gold_x: int) -> int:
    candidate = gold_x + 1
    if candidate == 0:
        candidate = gold_x + 2
    if candidate > 7:
        candidate = gold_x - 1
        if candidate == 0:
            candidate = gold_x - 2
    return candidate


def coord_hard_negatives(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    """Plausible wrong answers for a scene pair, each with the roles it plays
    (the twin's gold, the target's other coordinate, a neighbour's coordinate,
    the most similar label's coordinate, the adjacent gridline)."""
    points_a, points_b = geometry["points_a"], geometry["points_b"]
    target = geometry["target_a"]
    gold_a, gold_b = geometry["answer_a"], geometry["answer_b"]
    values: dict[str, set[str]] = {}

    def add(value: Any, role: str) -> None:
        values.setdefault(str(value), set()).add(role)

    add(gold_a, "gold_member_a")
    add(gold_b, "gold_member_b")
    add(gold_b, "twin_member_gold_for_a")
    add(gold_a, "twin_member_gold_for_b")
    for tag, scene in (("member_a", points_a), ("member_b", points_b)):
        add(scene[target][1], f"same_point_y_{tag}")
        others = {label: p for label, p in scene.items() if label != target}
        nn_label = min(
            sorted(others),
            key=lambda l: (math.hypot(others[l][0] - scene[target][0],
                                      others[l][1] - scene[target][1]), l),
        )
        add(others[nn_label][0], f"nearest_neighbor_x_{tag}")
    sim_label = min(
        sorted(l for l in points_a if l != target),
        key=lambda l: (_levenshtein(l, target), l),
    )
    add(points_a[sim_label][0], "most_similar_label_x")
    add(_nearest_gridline(int(gold_a)), "nearest_gridline_member_a")
    add(_nearest_gridline(int(gold_b)), "nearest_gridline_member_b")
    return [
        {"answer": value, "negative_types": sorted(roles)}
        for value, roles in sorted(values.items())
    ]

"""Generators for the held-out grounding suite: a named-point coordinate task, a header-cued
table task and a cued-readout marked-plot task. Both members of a pair share every pixel except
the one that decides the answer.
"""
from __future__ import annotations

import argparse
import math
import random
import string
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lwl.evaluation.metrics import match_tier
from lwl.grounding.schema import pair_record, stable_id, write_jsonl
from lwl.paths import data_path, results_root


COLORS = [
    (38, 94, 168),
    (205, 75, 65),
    (49, 139, 87),
    (142, 84, 160),
    (218, 145, 42),
    (32, 145, 153),
    (190, 91, 132),
    (105, 114, 130),
    (113, 92, 52),
    (61, 61, 61),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _answers_distinguishable(answer_a: str, answer_b: str) -> bool:
    return match_tier(answer_a, answer_b) == 0 and match_tier(answer_b, answer_a) == 0


def _exact_change_mask(image_a: Image.Image, image_b: Image.Image) -> Image.Image:
    array_a = np.asarray(image_a.convert("RGB"))
    array_b = np.asarray(image_b.convert("RGB"))
    changed = np.any(array_a != array_b, axis=2).astype(np.uint8) * 255
    return Image.fromarray(changed, mode="L")


def _save_rendered_pair(
    *,
    out_dir: Path,
    pair_id: str,
    image_a: Image.Image,
    image_b: Image.Image,
    question: str,
    answer_a: str,
    answer_b: str,
    category: str,
    template_id: str,
    provenance: dict[str, Any],
    verifier_results: dict[str, Any],
    swap_sides: bool = False,
) -> dict[str, Any]:
    """Write both images and their change mask, and return the manifest row."""
    if swap_sides:
        image_a, image_b = image_b, image_a
        answer_a, answer_b = answer_b, answer_a
    provenance = dict(provenance)
    provenance["semantic_side_assignment_swapped"] = swap_sides
    verifier_results = dict(verifier_results)
    verifier_results["semantic_side_assignment_swapped"] = swap_sides
    if image_a.size != image_b.size:
        raise ValueError(f"pair dimensions differ for {pair_id}")
    if not _answers_distinguishable(answer_a, answer_b):
        raise ValueError(f"degenerate answers for {pair_id}: {answer_a!r}, {answer_b!r}")
    image_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_a_path = image_dir / f"{pair_id}_a.png"
    image_b_path = image_dir / f"{pair_id}_b.png"
    mask_a_path = mask_dir / f"{pair_id}_a_mask.png"
    mask_b_path = mask_dir / f"{pair_id}_b_mask.png"
    mask = _exact_change_mask(image_a, image_b)
    if not np.any(np.asarray(mask)):
        raise ValueError(f"pair has no pixel change: {pair_id}")
    image_a.save(image_a_path, format="PNG", optimize=False, compress_level=9)
    image_b.save(image_b_path, format="PNG", optimize=False, compress_level=9)
    mask.save(mask_a_path, format="PNG", optimize=False, compress_level=9)
    mask.save(mask_b_path, format="PNG", optimize=False, compress_level=9)
    return pair_record(
        pair_id=pair_id,
        image_a_path=str(image_a_path),
        image_b_path=str(image_b_path),
        changed_region_mask_a=str(mask_a_path),
        changed_region_mask_b=str(mask_b_path),
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        category=category,
        template_id=template_id,
        provenance=provenance,
        verifier_results=verifier_results,
    )


def _procedural_labels(rng: random.Random, count: int) -> list[str]:
    labels: set[str] = set()
    consonants = "BCDFGHJKLMNPRSTVWXYZ"
    vowels = "AEIOU"
    while len(labels) < count:
        label = f"{rng.choice(consonants)}{rng.choice(vowels)}{rng.choice(consonants)}-{rng.randint(10, 99)}{rng.choice(string.ascii_uppercase)}"
        labels.add(label)
    return list(labels)


def _render_chart(labels: list[str], values: list[list[int]], target_series: int, target_x: int) -> Image.Image:
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), (249, 250, 248))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 78, 820, 650
    draw.text((width // 2, 30), "Multi-Series Calibration Trace", anchor="mm", font=_font(24, True), fill=(25, 25, 25))
    draw.rectangle((left, top, right, bottom), fill=(255, 255, 255), outline=(45, 45, 45), width=2)
    for tick in range(0, 101, 10):
        y = bottom - round(tick / 100 * (bottom - top))
        draw.line((left, y, right, y), fill=(232, 232, 232), width=1)
        if tick % 20 == 0:
            draw.text((left - 12, y), str(tick), anchor="rm", font=_font(13), fill=(55, 55, 55))
    x_positions = [left + 55 + index * 128 for index in range(6)]
    for index, x in enumerate(x_positions, start=1):
        draw.line((x, top, x, bottom), fill=(242, 242, 242), width=1)
        draw.text((x, bottom + 24), str(index), anchor="mm", font=_font(14), fill=(55, 55, 55))
    draw.text(((left + right) // 2, bottom + 54), "x", anchor="mm", font=_font(17, True), fill=(40, 40, 40))

    for series_index, series_values in enumerate(values):
        points = [
            (x, bottom - round(value / 100 * (bottom - top)))
            for x, value in zip(x_positions, series_values)
        ]
        color = COLORS[series_index]
        draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(255, 255, 255), width=1)
        if series_index == target_series:
            target_point = points[target_x]
            draw.ellipse(
                (target_point[0] - 15, target_point[1] - 15, target_point[0] + 15, target_point[1] + 15),
                fill=(255, 255, 255),
                outline=(0, 0, 0),
                width=2,
            )
            draw.text(target_point, "*", anchor="mm", font=_font(26, True), fill=(0, 0, 0))

    legend_left = 855
    draw.rectangle((legend_left, 74, 1168, 650), fill=(255, 255, 255), outline=(175, 175, 175), width=2)
    draw.text((1010, 100), "Series key", anchor="mm", font=_font(19, True), fill=(25, 25, 25))
    for index, label in enumerate(labels):
        y = 142 + index * 48
        draw.line((legend_left + 28, y, legend_left + 65, y), fill=COLORS[index], width=5)
        draw.ellipse((legend_left + 43, y - 4, legend_left + 51, y + 4), fill=COLORS[index])
        draw.text((legend_left + 82, y), label, anchor="lm", font=_font(16), fill=(20, 20, 20))
        if index == target_series:
            draw.text((legend_left + 16, y), "*", anchor="mm", font=_font(25, True), fill=(0, 0, 0))
    draw.text((90, 714), f"The black star marks the queried point at x = {target_x + 1}.", font=_font(14), fill=(75, 75, 75))
    return image


def generate_nine_series_chart_pairs(out_dir: Path, n: int, seed: int) -> list[dict[str, Any]]:
    """Marked-plot pairs: nine series over six x positions, with the starred point's value changed."""
    rows = []
    for index in range(n):
        pair_seed = seed + index * 104729
        rng = random.Random(pair_seed)
        labels = _procedural_labels(rng, 9)
        values_a = [[rng.randrange(10, 91, 10) for _ in range(6)] for _ in range(9)]
        target_series = rng.randrange(9)
        target_x = rng.randrange(1, 5)
        values_b = [list(series) for series in values_a]
        current = values_a[target_series][target_x]
        candidates = [value for value in range(10, 91, 10) if abs(value - current) >= 20]
        values_b[target_series][target_x] = rng.choice(candidates)
        answer_a = str(current)
        answer_b = str(values_b[target_series][target_x])
        pair_id = "v02_chart9x6_starred_" + stable_id(
            pair_seed, labels, target_series, target_x, answer_a, answer_b
        )
        rows.append(
            _save_rendered_pair(
                out_dir=out_dir,
                pair_id=pair_id,
                image_a=_render_chart(labels, values_a, target_series, target_x),
                image_b=_render_chart(labels, values_b, target_series, target_x),
                question=f"What is the value of the starred series at x = {target_x + 1}?",
                answer_a=answer_a,
                answer_b=answer_b,
                category="chart_two_hop_read",
                template_id="starred_series_value_nine_v07",
                provenance={
                    "generator": "lwl.grounding.build_suite",
                    "pair_seed": pair_seed,
                    "visual_operation": "star_localize_then_legend_bind_then_coordinate_read",
                    "training_domain_alignment": "medium",
                    "caption_failure_targeted": "fifty_four_question_blind_series_value_bindings",
                    "render_variant": "nine_series_six_intervals_direct_star_r16",
                },
                verifier_results={
                    "exact_by_construction": True,
                    "series_count": len(labels),
                    "x_count": 6,
                    "target_series_index": target_series,
                    "target_x": target_x + 1,
                    "target_point_starred": True,
                    "target_legend_starred": True,
                    "shared_content_seed": pair_seed,
                    "only_semantic_change": "one series value",
                },
                swap_sides=rng.random() < 0.5,
            )
        )
    return rows


def _render_high_entropy_coordinate_register(points: dict[str, tuple[int, int]]) -> Image.Image:
    width, height = 1400, 1240
    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 38), "Coordinate Survey Register", anchor="mm", font=_font(26, True), fill=(25, 25, 25))
    origin = (700, 650)
    scale = 68
    plot_left = origin[0] - 7 * scale
    plot_right = origin[0] + 7 * scale
    plot_top = origin[1] - 7 * scale
    plot_bottom = origin[1] + 7 * scale
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), fill="white", outline=(75, 75, 75), width=2)
    for value in range(-7, 8):
        x = origin[0] + value * scale
        y = origin[1] - value * scale
        draw.line((x, plot_top, x, plot_bottom), fill=(224, 228, 232), width=1)
        draw.line((plot_left, y, plot_right, y), fill=(224, 228, 232), width=1)
        if value:
            draw.text((x, origin[1] + 19), str(value), anchor="mm", font=_font(13), fill=(65, 65, 65))
            draw.text((origin[0] - 20, y), str(value), anchor="mm", font=_font(13), fill=(65, 65, 65))
    draw.line((plot_left, origin[1], plot_right, origin[1]), fill=(40, 40, 40), width=3)
    draw.line((origin[0], plot_top, origin[0], plot_bottom), fill=(40, 40, 40), width=3)

    for index, (label, point) in enumerate(points.items()):
        x = origin[0] + point[0] * scale
        y = origin[1] - point[1] * scale
        color = COLORS[index % len(COLORS)]
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color, outline="white", width=2)
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
        "Locate the requested label, then read its coordinate from the numbered axes.",
        font=_font(15),
        fill=(70, 70, 70),
    )
    return image


def _sample_high_entropy_points(rng: random.Random, count: int) -> list[tuple[int, int]]:
    candidates = [(x, y) for x in range(-7, 8) for y in range(-7, 8) if x != 0 and y != 0]
    rng.shuffle(candidates)
    selected: list[tuple[int, int]] = []
    for candidate in candidates:
        if all(max(abs(candidate[0] - x), abs(candidate[1] - y)) >= 2 for x, y in selected):
            selected.append(candidate)
            if len(selected) == count:
                return selected
    raise ValueError(f"unable to place {count} high-entropy coordinate labels")


def generate_coordinate_register_high_entropy_pairs(out_dir: Path, n: int, seed: int) -> list[dict[str, Any]]:
    """Named-point pairs: twenty labelled points, with the queried point moved along x."""
    rows = []
    label_pool = [f"{letter}{digit}" for letter in "BCDFGHJKLMNPRSTVWXYZ" for digit in "23456789"]
    for index in range(n):
        pair_seed = seed + index * 104729
        rng = random.Random(pair_seed)
        labels = rng.sample(label_pool, 20)
        coordinates = _sample_high_entropy_points(rng, len(labels))
        points_a = dict(zip(labels, coordinates))
        candidates_by_label: dict[str, list[tuple[int, int]]] = {}
        for label, point in points_a.items():
            other_points = {value for key, value in points_a.items() if key != label}
            candidates = [
                (x, point[1])
                for x in range(-7, 8)
                if x != 0
                and abs(x - point[0]) >= 3
                and _answers_distinguishable(str(point[0]), str(x))
                and all(max(abs(x - ox), abs(point[1] - oy)) >= 2 for ox, oy in other_points)
            ]
            if candidates:
                candidates_by_label[label] = candidates
        if not candidates_by_label:
            raise ValueError(f"no horizontal counterfactual targets for seed {pair_seed}")
        target_label = rng.choice(list(candidates_by_label))
        target_a = points_a[target_label]
        target_candidates = candidates_by_label[target_label]
        target_b = rng.choice(target_candidates)
        points_b = dict(points_a)
        points_b[target_label] = target_b
        answer_a = str(target_a[0])
        answer_b = str(target_b[0])
        pair_id = "v02_register20x_" + stable_id(pair_seed, labels, target_label, target_a, target_b)
        rows.append(
            _save_rendered_pair(
                out_dir=out_dir,
                pair_id=pair_id,
                image_a=_render_high_entropy_coordinate_register(points_a),
                image_b=_render_high_entropy_coordinate_register(points_b),
                question=f"What is the x-coordinate of point {target_label}?",
                answer_a=answer_a,
                answer_b=answer_b,
                category="geometry_coordinate_indexing",
                template_id="coordinate_register_twenty_point_x_v02",
                provenance={
                    "generator": "lwl.grounding.build_suite",
                    "pair_seed": pair_seed,
                    "visual_operation": "random_label_localization_then_x_coordinate_read",
                    "training_domain_alignment": "high",
                    "caption_failure_targeted": "twenty_question_blind_label_coordinate_bindings",
                    "render_variant": "twenty_point_x_r10_scale68_radius10_label19",
                },
                verifier_results={
                    "exact_by_construction": True,
                    "point_count": len(labels),
                    "target_label": target_label,
                    "target_a": target_a,
                    "target_b": target_b,
                    "target_y_preserved": target_a[1] == target_b[1],
                    "all_labels_randomized": True,
                },
                swap_sides=rng.random() < 0.5,
            )
        )
    return rows


def _render_header_cued_table(
    table: list[list[str]],
    row_labels: list[str],
    col_labels: list[str],
    target_row: int,
    target_col: int,
) -> Image.Image:
    width, height = 1120, 800
    image = Image.new("RGB", (width, height), (247, 248, 246))
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 42, width - 45, height - 42), fill="white", outline=(185, 185, 185), width=2)
    draw.text((width // 2, 78), "Repeated Field Verification Form", anchor="mm", font=_font(24, True), fill=(25, 25, 25))
    left, top = 102, 132
    row_header_width, cell_width, cell_height = 160, 130, 62
    draw.rectangle((left, top, left + row_header_width, top + cell_height), fill=(231, 235, 239), outline=(125, 125, 125))
    for column, label in enumerate(col_labels):
        x0 = left + row_header_width + column * cell_width
        outline = (36, 102, 165) if column == target_col else (125, 125, 125)
        width_line = 4 if column == target_col else 1
        draw.rectangle((x0, top, x0 + cell_width, top + cell_height), fill=(231, 235, 239), outline=outline, width=width_line)
        draw.text((x0 + cell_width // 2, top + cell_height // 2), label, anchor="mm", font=_font(17, True), fill=(32, 32, 32))
    for row, label in enumerate(row_labels):
        y0 = top + (row + 1) * cell_height
        outline = (36, 102, 165) if row == target_row else (125, 125, 125)
        width_line = 4 if row == target_row else 1
        draw.rectangle((left, y0, left + row_header_width, y0 + cell_height), fill=(231, 235, 239), outline=outline, width=width_line)
        draw.text((left + row_header_width // 2, y0 + cell_height // 2), label, anchor="mm", font=_font(15, True), fill=(32, 32, 32))
        for column, value in enumerate(table[row]):
            x0 = left + row_header_width + column * cell_width
            fill = (255, 255, 255) if (row + column) % 2 == 0 else (248, 249, 250)
            draw.rectangle((x0, y0, x0 + cell_width, y0 + cell_height), fill=fill, outline=(155, 155, 155), width=1)
            draw.text((x0 + cell_width // 2, y0 + cell_height // 2), value, anchor="mm", font=_font(21), fill=(15, 15, 15))
    draw.text((102, 724), "Blue outlines cue only the requested row header and column header; the cell is not highlighted.", font=_font(14), fill=(70, 70, 70))
    return image


def generate_header_table_pairs(out_dir: Path, n: int, seed: int) -> list[dict[str, Any]]:
    """Header-cued table pairs: the row and column headers are outlined, the cell is not."""
    rows = []
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    col_labels = ["F2", "G4", "H7", "J9", "K3", "L8"]
    for index in range(n):
        pair_seed = seed + index * 104729
        rng = random.Random(pair_seed)
        row_labels = [f"Case-{value:03d}" for value in rng.sample(range(101, 999), 8)]
        table_a = [["".join(rng.choice(chars) for _ in range(2)) for _ in col_labels] for _ in row_labels]
        target_row = rng.randrange(len(row_labels))
        target_col = rng.randrange(len(col_labels))
        table_b = [list(row) for row in table_a]
        replacement = table_a[target_row][target_col]
        while replacement == table_a[target_row][target_col]:
            replacement = "".join(rng.choice(chars) for _ in range(2))
        table_b[target_row][target_col] = replacement
        answer_a = table_a[target_row][target_col]
        answer_b = table_b[target_row][target_col]
        pair_id = "v02_headertable_" + stable_id(pair_seed, row_labels, target_row, target_col, answer_a, answer_b)
        rows.append(
            _save_rendered_pair(
                out_dir=out_dir,
                pair_id=pair_id,
                image_a=_render_header_cued_table(table_a, row_labels, col_labels, target_row, target_col),
                image_b=_render_header_cued_table(table_b, row_labels, col_labels, target_row, target_col),
                question=f"What is the 2-character code at row {row_labels[target_row]} and column {col_labels[target_col]}?",
                answer_a=answer_a,
                answer_b=answer_b,
                category="document_header_indexing",
                template_id="header_cued_table_code_v02",
                provenance={
                    "generator": "lwl.grounding.build_suite",
                    "pair_seed": pair_seed,
                    "visual_operation": "header_cued_row_column_indexing",
                    "training_domain_alignment": "low",
                },
                verifier_results={
                    "exact_by_construction": True,
                    "target_row": target_row,
                    "target_column": target_col,
                    "target_cell_highlighted": False,
                },
                swap_sides=rng.random() < 0.5,
            )
        )
    return rows


# Family name to (seed offset, generator). The offsets are fixed: rebuilding a family with the
# same seed has to reproduce the pair ids of the released manifests.
GENERATORS: dict[str, tuple[int, Callable[[Path, int, int], list[dict[str, Any]]]]] = {
    "chart_nine": (10, generate_nine_series_chart_pairs),
    "coordinate_register_high_entropy": (15, generate_coordinate_register_high_entropy_pairs),
    "header_table": (16, generate_header_table_pairs),
}


def build(
    out_dir: str | Path,
    n_per_template: int,
    seed: int,
    families: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Render n_per_template pairs for each selected family and return their manifest rows."""
    out_dir = Path(out_dir)
    rows: list[dict[str, Any]] = []
    selected = set(GENERATORS) if families is None else families
    unknown = selected - set(GENERATORS)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    for name, (offset, generator) in GENERATORS.items():
        if name not in selected:
            continue
        rows.extend(generator(out_dir / name, n_per_template, seed + offset * 1009))
    return rows


def write_contact_sheets(rows: list[dict[str, Any]], output_dir: str | Path, n_per_template: int = 20) -> list[Path]:
    """One sheet per template, tiling both sides of each pair with their answers, for inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_template: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_template.setdefault(str(row["template_id"]), []).append(row)
    outputs = []
    for template_id, template_rows in sorted(by_template.items()):
        selected = template_rows[:n_per_template]
        tile_w, tile_h = 470, 250
        columns = 4
        rows_count = math.ceil(len(selected) / columns)
        sheet = Image.new("RGB", (tile_w * columns, tile_h * rows_count), (238, 238, 238))
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(selected):
            x0 = (index % columns) * tile_w
            y0 = (index // columns) * tile_h
            draw.rectangle((x0 + 2, y0 + 2, x0 + tile_w - 2, y0 + tile_h - 2), fill="white", outline=(160, 160, 160))
            for side_index, side in enumerate(("a", "b")):
                with Image.open(row[f"image_{side}_path"]) as source:
                    thumb = source.convert("RGB")
                    thumb.thumbnail((220, 178), Image.Resampling.LANCZOS)
                x = x0 + 8 + side_index * 230 + (220 - thumb.width) // 2
                y = y0 + 8 + (178 - thumb.height) // 2
                sheet.paste(thumb, (x, y))
                draw.text((x0 + 118 + side_index * 230, y0 + 191), f"{side.upper()}: {row[f'answer_{side}']}", anchor="mm", font=_font(13, True), fill=(20, 20, 20))
            question = str(row["question"])
            if len(question) > 70:
                question = question[:67] + "..."
            draw.text((x0 + 10, y0 + 214), question, font=_font(11), fill=(35, 35, 35))
            draw.text((x0 + 10, y0 + 232), str(row["pair_id"]), font=_font(9), fill=(100, 100, 100))
        output = output_dir / f"{template_id}.png"
        sheet.save(output, format="PNG", optimize=False, compress_level=9)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=data_path("grounding", "suite_source", "renderable"))
    parser.add_argument("--manifest", default=data_path("grounding", "suite_source", "manifest.jsonl"))
    parser.add_argument("--contact-sheet-dir", default=results_root() / "contact_sheets" / "grounding_suite")
    parser.add_argument("--n-per-template", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--families", help="Comma-separated family names; default is all families")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    manifest = Path(args.manifest)
    if manifest.exists() or (out_dir.exists() and any(out_dir.iterdir())):
        raise FileExistsError(f"refusing to overwrite an existing build: {manifest} / {out_dir}")
    families = {item.strip() for item in args.families.split(",") if item.strip()} if args.families else None
    rows = build(args.out_dir, args.n_per_template, args.seed, families=families)
    write_jsonl(args.manifest, rows)
    sheets = write_contact_sheets(rows, args.contact_sheet_dir)
    print(f"manifest={args.manifest} pairs={len(rows)} contact_sheets={len(sheets)}")


if __name__ == "__main__":
    main()

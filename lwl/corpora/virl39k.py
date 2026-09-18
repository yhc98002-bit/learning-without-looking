"""Read the ViRL39K parquet release: row loading, corpus statistics and a contact sheet.

Rows keep the upstream qid, category, source and pass-rate columns.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


BOXED_CONTENT = re.compile(r"^\\boxed\{(.+)\}$")
NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\d+/\d+)(?:\^\\circ)?$")


def answer_type(answer: str) -> str:
    """Classify an answer as multiple_choice, numeric or text_or_expression."""
    value = answer.strip()
    match = BOXED_CONTENT.fullmatch(value)
    content = match.group(1).strip() if match else value
    if re.fullmatch(r"[A-Z]", content):
        return "multiple_choice"
    if NUMERIC.fullmatch(content):
        return "numeric"
    return "text_or_expression"


def load_rows(parquet_path: str | Path, image_root: str | Path) -> list[dict[str, Any]]:
    parquet_path = Path(parquet_path)
    image_root = Path(image_root)
    required = {"question", "answer", "category", "source", "qid", "image"}
    table = pq.read_table(parquet_path)
    missing_columns = required - set(table.column_names)
    if missing_columns:
        raise ValueError(f"ViRL39K parquet missing columns: {sorted(missing_columns)}")
    rows = []
    seen_qids: set[str] = set()
    for raw in table.to_pylist():
        qid = str(raw["qid"])
        if qid in seen_qids:
            raise ValueError(f"duplicate ViRL39K qid: {qid}")
        seen_qids.add(qid)
        relative_images = [str(item) for item in (raw.get("image") or [])]
        rows.append(
            {
                "qid": qid,
                "question": str(raw["question"]),
                "answer": str(raw["answer"]),
                "category": str(raw["category"]),
                "source": str(raw["source"]),
                "pass_rate_32b_trained": float(raw.get("PassRate_32BTrained", -1.0)),
                "pass_rate_7b_base": float(raw.get("PassRate_7BBase", -1.0)),
                "image_paths": [str(image_root / relative) for relative in relative_images],
                "relative_image_paths": relative_images,
            }
        )
    return rows


def dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    references = [Path(path) for row in rows for path in row["image_paths"]]
    missing = [str(path) for path in references if not path.is_file()]
    dimensions: Counter[str] = Counter()
    readable = 0
    for path in references:
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                readable += 1
        except OSError:
            missing.append(str(path))
    return {
        "n_rows": len(rows),
        "n_image_references": len(references),
        "n_readable_images": readable,
        "n_missing_or_unreadable_images": len(missing),
        "missing_image_rate": len(missing) / len(references) if references else 0.0,
        "missing_examples": missing[:20],
        "answer_type_counts": dict(sorted(Counter(answer_type(row["answer"]) for row in rows).items())),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "image_dimension_counts": dict(dimensions.most_common()),
    }


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def write_contact_sheet(rows: list[dict[str, Any]], output: str | Path, seed: int = 20260710, count: int = 16) -> None:
    """Render a grid of sampled items with their qid, question and answer, for eyeballing."""
    candidates = [row for row in rows if row["image_paths"] and Path(row["image_paths"][0]).is_file()]
    selected = random.Random(seed).sample(candidates, min(count, len(candidates)))
    tile_width, tile_height, columns = 420, 300, 4
    rows_count = max(1, (len(selected) + columns - 1) // columns)
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows_count), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        x0 = (index % columns) * tile_width
        y0 = (index // columns) * tile_height
        with Image.open(row["image_paths"][0]) as opened:
            image = opened.convert("RGB")
            image.thumbnail((390, 205), Image.Resampling.LANCZOS)
        sheet.paste(image, (x0 + (tile_width - image.width) // 2, y0 + 8))
        question = row["question"].replace("<image>", "").replace("\n", " ").strip()
        if len(question) > 75:
            question = question[:72] + "..."
        draw.text((x0 + 10, y0 + 220), str(row["qid"]), font=_font(11), fill=(60, 60, 60))
        draw.text((x0 + 10, y0 + 240), question, font=_font(10), fill=(20, 20, 20))
        draw.text((x0 + 10, y0 + 274), str(row["answer"]), font=_font(11), fill=(20, 70, 130))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=False, compress_level=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--contact-sheet", required=True)
    args = parser.parse_args()
    rows = load_rows(args.parquet, args.image_root)
    stats = dataset_stats(rows)
    output = Path(args.stats_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_contact_sheet(rows, args.contact_sheet)
    print(json.dumps({key: stats[key] for key in ("n_rows", "n_image_references", "missing_image_rate")}))


if __name__ == "__main__":
    main()

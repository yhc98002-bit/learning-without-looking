"""Corpus rows turned into chat messages under one image condition: the real image, gray,
noise, no image at all, or a question-blind caption in place of the image.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from jinja2 import Template

from lwl.captions.store import merge_caption_rows
from lwl.evaluation.image_conditions import materialize_image
from lwl.paths import resolve_data_reference


CONDITIONS = ("real", "gray", "noise", "none", "caption")


def load_geometry_rows(
    manifest: str | Path,
    splits: Iterable[str] = ("train", "test"),
    *,
    train_filter_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    split_order = {name: index for index, name in enumerate(splits)}
    selected = set(split_order)
    rows = []
    matched_train_ids: set[int] = set()
    with Path(manifest).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = str(row["split"])
            if split not in selected:
                continue
            row_index = int(row["row_index"])
            if split == "train" and train_filter_ids is not None:
                if row_index not in train_filter_ids:
                    continue
                matched_train_ids.add(row_index)
            rows.append(row)
    if train_filter_ids is not None and "train" in selected:
        missing = sorted(train_filter_ids - matched_train_ids)
        if missing:
            preview = ", ".join(str(value) for value in missing[:10])
            raise ValueError(
                f"train filter contains {len(missing)} IDs absent from the selected manifest: {preview}"
            )
    return sorted(
        rows,
        key=lambda row: (split_order[str(row["split"])], int(row["row_index"])),
    )


def load_caption_map(shards: Iterable[str | Path]) -> dict[str, str]:
    rows, _ = merge_caption_rows(shards)
    return {str(row["image_sha256"]): str(row["caption"]).strip() for row in rows}


def build_conditioned_messages(
    row: dict[str, Any],
    format_prompt: str,
    condition: str,
    cache_dir: str | Path,
    captions: dict[str, str] | None = None,
    noise_seed: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported image condition: {condition}")
    rendered = Template(format_prompt.strip()).render(content=str(row["problem"]))
    images = list(row.get("images", []))
    parts = rendered.split("<image>")
    if len(parts) - 1 != len(images):
        raise ValueError(
            f"row {row['row_index']} has {len(parts) - 1} image markers but {len(images)} images"
        )

    content: list[dict[str, Any]] = []
    conditioned_paths: list[str] = []
    cache_dir = Path(cache_dir)
    for index, part in enumerate(parts):
        if index:
            image = images[index - 1]
            if condition in {"real", "gray", "noise"}:
                path = materialize_image(
                    str(resolve_data_reference(image["path"])),
                    condition,
                    cache_dir,
                    noise_seed=noise_seed,
                )
                content.append({"type": "image", "image": path})
                conditioned_paths.append(path)
            elif condition == "caption":
                if captions is None or str(image["sha256"]) not in captions:
                    raise KeyError(f"missing fixed caption for image {image['sha256']}")
                caption = captions[str(image["sha256"])]
                content.append(
                    {
                        "type": "text",
                        "text": f"\n[Question-blind image description {index}: {caption}]\n",
                    }
                )
        if part:
            content.append({"type": "text", "text": part})

    if condition == "none":
        text = "".join(str(item["text"]) for item in content if item["type"] == "text")
        text = re.sub(r"[ \t]+", " ", text)
        content = [{"type": "text", "text": text}]
    return [{"role": "user", "content": content}], conditioned_paths

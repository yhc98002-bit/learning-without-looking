#!/usr/bin/env python3
"""Re-render a constructed-scene batch from its recorded manifests with the cue-level-neutral footer.
No random number is drawn; only image and mask paths, their hashes and provenance change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.paths import data_path, results_root
from lwl.scenes.program import (
    COORD_ALLOWED,
    COORD_FOOTER,
    COORD_TITLE,
    SCENE_TEXT,
    _draw_cue,
    coord_target_px,
    cue_ink_disjoint,
    render_coordinate_scene,
)

FAMILY = "hier_coord_v1"
CELLS = ("n8", "n12", "n20")
LAYERS = ("l3", "l2", "l1", "probe")
# Recorded in the provenance of every re-rendered row, as released.
RENDER_REV = "r2-footer-neutral"
IMAGE_FIELDS = ("image_a_path", "image_b_path", "image_a_sha256",
                "image_b_sha256", "changed_region_mask_a",
                "changed_region_mask_b", "mask_sha256", "provenance")


def save_image(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    image.save(path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_mask(image_a: Image.Image, image_b: Image.Image, path: Path) -> str:
    changed = np.any(np.asarray(image_a, dtype=np.uint8)
                     != np.asarray(image_b, dtype=np.uint8), axis=2)
    if not changed.any():
        raise AssertionError(f"pair has no pixel change: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    Image.fromarray(changed.astype(np.uint8) * 255, mode="L").save(
        path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def points_of(scene: list) -> dict[str, tuple[int, int]]:
    return {label: (int(x), int(y)) for label, x, y in scene}


def draw_recorded_cue(base: Image.Image, points: dict, target: str,
                      cue: dict) -> Image.Image:
    cued = base.copy()
    _draw_cue(ImageDraw.Draw(cued), coord_target_px(points, target),
              tuple(cue["cue_direction"]), tuple(cue["cue_radii"]))
    if not cue_ink_disjoint(base, cued, COORD_ALLOWED):
        raise AssertionError("re-drawn cue is not ink-disjoint")
    delta = np.any(np.asarray(base) != np.asarray(cued), axis=2)
    if int(delta.sum()) != int(cue["cue_pixel_count"]):
        raise AssertionError(
            f"cue pixel count changed: {int(delta.sum())} vs "
            f"{cue['cue_pixel_count']}")
    return cued


def rewrite_row(row: dict, out_root: Path, images: dict, masks: dict) -> dict:
    new = json.loads(json.dumps(row))  # deep copy
    layer = row["layer"]
    img_layer = "l3" if layer == "probe" else layer
    mask_layer = "l1" if layer == "l1" else "l3"
    pair_key = row["mother_item_id"]
    cell_dir = out_root / FAMILY / row["cell"]
    new["image_a_path"] = str(cell_dir / "images" / f"{pair_key}_a_{img_layer}.png")
    new["image_b_path"] = str(cell_dir / "images" / f"{pair_key}_b_{img_layer}.png")
    new["image_a_sha256"] = images[(pair_key, img_layer, "a")]
    new["image_b_sha256"] = images[(pair_key, img_layer, "b")]
    mask_path = str(cell_dir / "masks" / f"{pair_key}_{mask_layer}_mask.png")
    new["changed_region_mask_a"] = mask_path
    new["changed_region_mask_b"] = mask_path
    new["mask_sha256"] = masks[(pair_key, mask_layer)]
    new["provenance"] = {**row["provenance"], "render_rev": RENDER_REV,
                         "rendered_text": SCENE_TEXT[FAMILY],
                         "v1_image_a_sha256": row["image_a_sha256"],
                         "v1_image_b_sha256": row["image_b_sha256"]}
    # non-image fields must be byte-equal to the input row
    for probe_row, v1_row in ((new, row),):
        a = {k: v for k, v in probe_row.items() if k not in IMAGE_FIELDS}
        b = {k: v for k, v in v1_row.items() if k not in IMAGE_FIELDS}
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            raise AssertionError(f"non-image field drift for {row['pair_id']}")
    return new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_path("scenes", "development"),
                        help="scene split to re-render")
    parser.add_argument("--out-dir", type=Path,
                        default=data_path("scenes", "development_rerendered"),
                        help="destination; it must not already hold a build")
    parser.add_argument("--report", type=Path,
                        default=results_root() / "scenes_rerender.json")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    summary = {"schema_version": "lwl.scene-rerender.v1",
               "render_rev": RENDER_REV,
               "footer": {"v1": "Locate the requested label, then read its "
                                "coordinate from the numbered axes.",
                          "r2": COORD_FOOTER},
               "title": COORD_TITLE, "cells": {}}
    for cell in CELLS:
        rows = {layer: [json.loads(l) for l in
                        (args.data_dir / f"manifest_{FAMILY}_{cell}_{layer}.jsonl")
                        .read_text().splitlines() if l.strip()]
                for layer in LAYERS}
        by_pair = {layer: {r["mother_item_id"]: r for r in rows[layer]}
                   for layer in LAYERS}
        images: dict = {}
        masks: dict = {}
        n_images = 0
        cell_dir = args.out_dir / FAMILY / cell
        for l3_row in rows["l3"]:
            pair_key = l3_row["mother_item_id"]
            base = {}
            for side in ("a", "b"):
                pts = points_of(l3_row[f"scene_{side}"])
                base[side] = render_coordinate_scene(pts)
                images[(pair_key, "l3", side)] = save_image(
                    base[side], cell_dir / "images" / f"{pair_key}_{side}_l3.png")
                n_images += 1
            masks[(pair_key, "l3")] = save_mask(
                base["a"], base["b"], cell_dir / "masks" / f"{pair_key}_l3_mask.png")
            l1_row = by_pair["l1"].get(pair_key)
            if l1_row is not None:
                for side in ("a", "b"):
                    images[(pair_key, "l2", side)] = save_image(
                        base[side], cell_dir / "images" / f"{pair_key}_{side}_l2.png")
                    n_images += 1
                    if images[(pair_key, "l2", side)] != images[(pair_key, "l3", side)]:
                        raise AssertionError(f"l2 != l3 bytes for {pair_key} {side}")
                cued = {}
                for side in ("a", "b"):
                    pts = points_of(l1_row[f"scene_{side}"])
                    target = str(l1_row["verifier_results"][f"target_label_{side}"])
                    cue = l1_row["verifier_results"]["cue"][side]
                    cued[side] = draw_recorded_cue(base[side], pts, target, cue)
                    images[(pair_key, "l1", side)] = save_image(
                        cued[side], cell_dir / "images" / f"{pair_key}_{side}_l1.png")
                    n_images += 1
                masks[(pair_key, "l1")] = save_mask(
                    cued["a"], cued["b"],
                    cell_dir / "masks" / f"{pair_key}_l1_mask.png")
        for layer in LAYERS:
            out_manifest = args.out_dir / f"manifest_{FAMILY}_{cell}_{layer}.jsonl"
            if out_manifest.exists():
                raise FileExistsError(out_manifest)
            out_rows = [rewrite_row(r, args.out_dir, images, masks)
                        for r in rows[layer]]
            out_manifest.write_text(
                "".join(json.dumps(r) + "\n" for r in out_rows), encoding="utf-8")
        summary["cells"][cell] = {"scene_pairs": len(rows["l3"]),
                                  "rows": {l: len(rows[l]) for l in LAYERS},
                                  "images_written": n_images}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary["cells"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Verify a constructed-scene batch from disk: rehash every image and mask, recompute every answer
from the stored scene, and re-check the cue, pairing, split, answer-balance and wording rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from lwl.paths import data_path, resolve_data_reference
from lwl.scenes.program import (
    COORD_ALLOWED,
    EXTREMUM_KINDS,
    PROCEDURE_TOKENS,
    SCENE_TEXT,
    coord_extremum,
    cue_ink_disjoint,
    split_of,
)

FAMILY = "hier_coord_v1"
CELLS = ("n8", "n12", "n20")
LAYERS = ("l3", "l2", "l1", "probe")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_of(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def recompute_coord(row: dict, side: str) -> tuple[str, str]:
    """(target label, answer) recomputed from the row's own scene truth."""
    points = {label: (x, y) for label, x, y in row[f"scene_{side}"]}
    kind = row["verifier_results"]["extremum_kind"]
    target, gap = coord_extremum(points, kind)
    if gap < row["verifier_results"]["extremum_margin"]:
        raise AssertionError("extremum margin violated")
    return target, str(points[target][EXTREMUM_KINDS[kind]["read"]])


def verify_cell(data_dir: Path, cell: str, problems: list[str]) -> dict:
    manifests = {
        layer: data_dir / f"manifest_{FAMILY}_{cell}_{layer}.jsonl"
        for layer in LAYERS
    }
    rows = {layer: rows_of(path) for layer, path in manifests.items()}
    images_rehashed = cue_checked = 0
    gold_counts: Counter = Counter()

    by_pair: dict[str, dict[str, dict]] = {}
    for layer in ("l3", "l2", "l1"):
        for row in rows[layer]:
            by_pair.setdefault(row["mother_item_id"], {})[layer] = row
    probes = {row["mother_item_id"]: row for row in rows["probe"]}

    # Rows built or re-rendered after the in-image text was made cue-level
    # neutral carry provenance.rendered_text, which must match exactly.
    for layer in LAYERS:
        for row in rows[layer]:
            rendered = row["provenance"].get("rendered_text")
            if rendered is not None and rendered != SCENE_TEXT[FAMILY]:
                problems.append(
                    f"{row['pair_id']}: rendered_text is not the in-image text "
                    f"for {FAMILY}")

    for pair_key, layer_rows in by_pair.items():
        l3 = layer_rows["l3"]
        pid = l3["pair_id"]
        # split rule
        declared = l3["split"]
        if declared not in ("training", "development", "confirmatory"):
            problems.append(f"{pid}: unknown split {declared!r}")
        elif split_of(l3["scene_program_id"]) != declared:
            problems.append(f"{pid}: scene program is not in its declared split")
        # answers and targets recomputed from scene truth, per side
        for side in ("a", "b"):
            try:
                target, answer = recompute_coord(l3, side)
            except AssertionError as error:
                problems.append(f"{pid}: {error}")
                continue
            if str(answer) != str(l3[f"answer_{side}"]):
                problems.append(f"{pid}: recomputed answer_{side} {answer} != {l3[f'answer_{side}']}")
            if str(target) != str(l3["verifier_results"][f"target_label_{side}"]):
                problems.append(f"{pid}: recomputed target_{side} {target} != recorded")
            if pair_key in probes and str(probes[pair_key][f"answer_{side}"]) != str(target):
                problems.append(f"{pid}: probe gold_{side} != recomputed target")
        # question-operand rule: discovery must NOT name the target; the cued
        # levels must
        target_name = str(l3["verifier_results"]["target_label_a"])
        if target_name in l3["question"]:
            problems.append(f"{pid}: L3 question names the target (oracle leak)")
        for layer in ("l2", "l1"):
            if layer in layer_rows and target_name not in layer_rows[layer]["question"]:
                problems.append(f"{pid}: {layer} question does not name the target")
        # cross-cue-level matching
        reference = (json.dumps([l3["answer_a"], l3["answer_b"], l3["hard_negatives"],
                                 l3["scene_a"], l3["scene_b"]], sort_keys=True, default=str))
        for layer in ("l2", "l1"):
            if layer not in layer_rows:
                continue
            row = layer_rows[layer]
            candidate = (json.dumps([row["answer_a"], row["answer_b"], row["hard_negatives"],
                                     row["scene_a"], row["scene_b"]], sort_keys=True, default=str))
            if candidate != reference:
                problems.append(f"{pid}: {layer} pair matching violated")
        # cue-level matrix
        if l3["role"] == "target_switch" and ("l2" in layer_rows or "l1" in layer_rows):
            problems.append(f"{pid}: switch pair carries l2/l1 rows")
        if l3["role"] != "target_switch" and ("l2" not in layer_rows or "l1" not in layer_rows):
            problems.append(f"{pid}: stable/invariance pair missing l2/l1 rows")
        # image and mask integrity
        for layer, row in layer_rows.items():
            images = {side: resolve_data_reference(row[f"image_{side}_path"], data_dir)
                      for side in ("a", "b")}
            for side, path in images.items():
                if sha256_file(path) != row[f"image_{side}_sha256"]:
                    problems.append(f"{pid}: {layer} image_{side} sha mismatch")
                images_rehashed += 1
            mask_path = resolve_data_reference(row["changed_region_mask_a"], data_dir)
            if sha256_file(mask_path) != row["mask_sha256"]:
                problems.append(f"{pid}: {layer} mask sha mismatch")
            with Image.open(images["a"]) as ia, Image.open(images["b"]) as ib:
                diff = np.any(np.asarray(ia, dtype=np.uint8) != np.asarray(ib, dtype=np.uint8), axis=2)
                with Image.open(mask_path) as mask:
                    if not np.array_equal(diff.astype(np.uint8) * 255,
                                          np.asarray(mask, dtype=np.uint8)):
                        problems.append(f"{pid}: {layer} mask is not the exact pixel diff")
        if "l2" in layer_rows:
            l2, l1 = layer_rows["l2"], layer_rows["l1"]
            for side in ("a", "b"):
                if l2[f"image_{side}_sha256"] != l3[f"image_{side}_sha256"]:
                    problems.append(f"{pid}: l2/l3 image_{side} not byte-identical")
                with Image.open(resolve_data_reference(l2[f"image_{side}_path"], data_dir)) as base, \
                        Image.open(resolve_data_reference(l1[f"image_{side}_path"], data_dir)) as cued:
                    if not cue_ink_disjoint(base, cued, COORD_ALLOWED):
                        problems.append(f"{pid}: cue ink rule violated on side {side}")
                    cue_checked += 1
        if l3["role"] != "invariance":
            gold_counts[str(l3["answer_a"])] += 1
            gold_counts[str(l3["answer_b"])] += 1

    balance = None
    if gold_counts:
        total = sum(gold_counts.values())
        top_value, top_count = gold_counts.most_common(1)[0]
        budget = max(1, int(0.10 * total))
        balance = {"max_count": top_count, "budget": budget, "total": total,
                   "max_share_value": top_value}
        if top_count > budget:
            problems.append(f"{FAMILY}/{cell}: balance budget violated ({balance})")

    return {
        "scene_pairs": len(by_pair),
        "rows": {layer: len(rows[layer]) for layer in LAYERS},
        "images_rehashed": images_rehashed,
        "cue_pairs_checked": cue_checked,
        "balance": balance,
    }


def scene_text_policy_problems() -> list[str]:
    """Static policy: the in-image strings must be cue-level neutral - no
    task-procedure tokens and no point labels."""
    problems: list[str] = []
    for family, texts in SCENE_TEXT.items():
        for kind, text in texts.items():
            low = text.lower()
            for token in PROCEDURE_TOKENS:
                if token in low:
                    problems.append(
                        f"{family} {kind}: procedure token {token!r} in "
                        f"in-image text")
            if re.search(r"\bpoint [A-Z][A-Za-z0-9]*\b", text):
                problems.append(
                    f"{family} {kind}: point label in in-image text")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_path("scenes", "development"))
    args = parser.parse_args()
    problems: list[str] = scene_text_policy_problems()
    summary: dict = {}
    for cell in CELLS:
        summary[f"{FAMILY}/{cell}"] = verify_cell(args.data_dir, cell, problems)
    print(json.dumps({"cells": summary, "n_problems": len(problems),
                      "problems": problems[:20]}, indent=2, sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

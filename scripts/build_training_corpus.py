#!/usr/bin/env python3
"""Build the constructed training corpus in the trainer's row schema.

Each scene pair is one group of four adjacent rows (discovery a/b, then identification probe a/b),
shuffled at pair level, plus the small development-bucket file EasyR1 requires as val_files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.paths import (
    data_path,
    repo_relative,
    resolve_data_reference,
    results_root,
)

FAMILY = "hier_coord_v1"
TRAIN_CELLS = ("n8", "n12")          # the n20 cell is held out of training
MEMBER_ORDER = (("l3", "a"), ("l3", "b"), ("probe", "a"), ("probe", "b"))
MEMBER_LABELS = tuple(f"{layer}_{side}" for layer, side in MEMBER_ORDER)
GROUP_MODE = "pair4"
CORPUS_SEED = 20260817
# Category values carried by the released rows.
CATEGORY = "hier_v1_st3"
VAL_CATEGORY = "hier_v1_st3_plumbing_val"


def rel(path: str) -> str:
    return repo_relative(path)


def build_rows(data_dir: Path) -> tuple[list[dict], dict]:
    units: list[list[list[dict]]] = []
    stats = {"cells": {}, "n_groups": 0, "n_rows": 0}
    for cell in TRAIN_CELLS:
        by_layer = {}
        for layer in ("l3", "probe"):
            path = data_dir / f"manifest_{FAMILY}_{cell}_{layer}.jsonl"
            by_layer[layer] = {r["mother_item_id"]: r for r in
                               (json.loads(l) for l in
                                path.read_text().splitlines() if l.strip())}
        pairs = sorted(set(by_layer["l3"]) & set(by_layer["probe"]))
        for pair in pairs:
            base: dict[tuple[str, str], dict] = {}
            ok = True
            for layer, side in MEMBER_ORDER:
                row = by_layer[layer][pair]
                if row["split"] != "training":
                    ok = False
                    break
                base[(layer, side)] = {
                    "problem": f"<image>{row['question']}",
                    "answer": str(row[f"answer_{side}"]),
                    "images": [rel(row[f"image_{side}_path"])],
                    "template_id": row["template_id"],
                    "category": CATEGORY,
                }
            if not ok or len(base) != len(MEMBER_ORDER):
                continue
            units.append([[{**base[(layer, side)],
                            "pair_group_uid": pair,
                            "pair_member": f"{layer}_{side}"}
                           for layer, side in MEMBER_ORDER]])
        stats["cells"][cell] = len(pairs)
    rng = random.Random(CORPUS_SEED)
    rng.shuffle(units)
    groups = [group for unit in units for group in unit]
    rows = [row for group in groups for row in group]
    stats["n_groups"] = len(groups)
    stats["n_rows"] = len(rows)
    stats["n_scene_pairs"] = len(units)
    return rows, stats


def build_val_rows(dev_dir: Path) -> list[dict]:
    """Inert validation rows: the first two development pairs of each training cell's L3 manifest."""
    val_rows: list[dict] = []
    for cell in TRAIN_CELLS:
        l3_path = dev_dir / f"manifest_{FAMILY}_{cell}_l3.jsonl"
        if not l3_path.exists():
            continue
        for line in l3_path.read_text().splitlines()[:2]:
            row = json.loads(line)
            if row["split"] != "development":
                continue
            for side in ("a", "b"):
                val_rows.append({
                    "problem": f"<image>{row['question']}",
                    "answer": str(row[f"answer_{side}"]),
                    "images": [rel(row[f"image_{side}_path"])],
                    "pair_group_uid": row["mother_item_id"],
                    "pair_member": f"l3_{side}",
                    "template_id": row["template_id"],
                    "category": VAL_CATEGORY,
                })
    return val_rows


def val_blob_of(val_rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in val_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_path("scenes", "training"))
    parser.add_argument("--out-dir", type=Path, default=data_path("training", "constructed"))
    parser.add_argument("--dev-dir", type=Path, default=data_path("scenes", "development"),
                        help="development bucket, used only for the inert "
                             "validation file EasyR1 requires")
    parser.add_argument("--report", type=Path, default=results_root() / "constructed_corpus.json")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.report = args.report.resolve()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.out_dir.mkdir(parents=True, exist_ok=False)

    rows, stats = build_rows(args.data_dir.resolve())
    if not rows:
        raise SystemExit("no training rows built")

    # The trainer reads each group as four adjacent rows in a fixed order.
    group_size, member_labels = len(MEMBER_ORDER), MEMBER_LABELS
    for index in range(0, len(rows), group_size):
        block = rows[index:index + group_size]
        uids = {r["pair_group_uid"] for r in block}
        if len(uids) != 1:
            raise AssertionError(f"group not adjacent at row {index}: {uids}")
        if tuple(r["pair_member"] for r in block) != member_labels:
            raise AssertionError(f"member order wrong at row {index}")
    for row in rows:
        if not resolve_data_reference(row["images"][0]).is_file():
            raise FileNotFoundError(row["images"][0])
        if row["problem"].count("<image>") != len(row["images"]):
            raise AssertionError(f"image marker/count mismatch: {row['pair_group_uid']}")

    # EasyR1 requires data.val_files even with validation disabled (val_freq 0); the rows come
    # from the development bucket so a stray validation pass never reads training items.
    val_rows = build_val_rows(args.dev_dir.resolve())
    if not val_rows:
        raise SystemExit("no plumbing val rows built")
    val_blob = val_blob_of(val_rows)
    (args.out_dir / "plumbing_val.jsonl").write_text(val_blob, encoding="utf-8")

    jsonl = args.out_dir / "train.jsonl"
    blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    jsonl.write_text(blob, encoding="utf-8")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        schema = pa.schema([("problem", pa.string()), ("answer", pa.string()),
                            ("images", pa.list_(pa.string())),
                            ("pair_group_uid", pa.string()),
                            ("pair_member", pa.string()),
                            ("template_id", pa.string()), ("category", pa.string())])
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, args.out_dir / "train.parquet")
        parquet = True
    except Exception as error:            # noqa: BLE001 - reported, not swallowed
        parquet = f"unavailable: {error}"

    definition = ("one scene pair = one intervention group of 4 members "
                  "(L3 a/b + identification probe a/b), adjacent, fixed order")
    report = {
        "schema_version": "lwl.constructed-training-corpus.v1",
        "split": "training",
        "family": FAMILY,
        "cells": TRAIN_CELLS,
        "group_mode": GROUP_MODE,
        "group_definition": definition,
        "group_size": group_size,
        "member_order": list(member_labels),
        "corpus_seed": CORPUS_SEED,
        "n_groups": stats["n_groups"],
        "n_rows": stats["n_rows"],
        "n_scene_pairs": stats["n_scene_pairs"],
        "scene_pairs_per_cell": stats["cells"],
        "train_jsonl_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "plumbing_val_rows": len(val_rows),
        "plumbing_val_sha256": hashlib.sha256(val_blob.encode()).hexdigest(),
        "plumbing_val_note": "development-bucket rows; validation is disabled "
                             "(val_freq 0) and this file exists only to satisfy "
                             "EasyR1's non-null val_files requirement; rows carry "
                             "the mode's group contract",
        "parquet": parquet,
        "out_dir": str(args.out_dir),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("group_mode", "n_groups", "n_rows", "n_scene_pairs",
                       "scene_pairs_per_cell", "parquet", "train_jsonl_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

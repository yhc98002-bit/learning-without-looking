#!/usr/bin/env python3
"""Build one batch of constructed coordinate scenes: for each density cell, counterfactual scene
pairs rendered at every cue level, with one manifest per cell and level. Nothing is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.paths import data_path, repo_relative, repo_root, results_root
from lwl.scenes.program import (
    COORD_MARGIN,
    EXTREMUM_ROTATION,
    SCENE_TEXT,
    build_coord_geometry,
    coord_hard_negatives,
    render_coord_layers,
    render_coordinate_scene,
    split_of,
)

ROOT = repo_root()
FAMILY = "hier_coord_v1"      # family tag carried by the released manifests
FAMILY_TAG = "coord"          # item-id component, likewise carried by the data
BATCH_SEED = 20260817
ROLES = ("target_switch", "target_stable", "invariance")
PER_ROLE = 50
BALANCE_CAP = 0.10            # max share of the pooled causal answers per value
CELLS = (("n8", 8), ("n12", 12), ("n20", 20))


def attempt_rng(cell: str, role: str, attempt: int) -> random.Random:
    seed = int(hashlib.sha256(
        f"{BATCH_SEED}|{FAMILY}|{cell}|{role}|{attempt}".encode()
    ).hexdigest()[:12], 16)
    return random.Random(seed)


def scene_program_id(payload: Any) -> str:
    # The prefix and the salt are part of every released scene id.
    blob = json.dumps(payload, sort_keys=True, default=str)
    return "hier1_" + hashlib.sha256(("hier1|" + blob).encode()).hexdigest()[:16]


def _save_image(image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    image.save(path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask(image_a, image_b, path: Path) -> str:
    import numpy as np
    from PIL import Image as PILImage
    changed = np.any(
        np.asarray(image_a, dtype=np.uint8) != np.asarray(image_b, dtype=np.uint8),
        axis=2,
    )
    if not changed.any():
        raise AssertionError(f"pair has no pixel change: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(changed.astype(np.uint8) * 255, mode="L").save(
        path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_cell(cell_name: str, n_points: int, out_dir: Path,
               split: str = "development",
               per_role: int | None = None) -> dict[str, Any]:
    rows_by_layer: dict[str, list[dict]] = {"l3": [], "l2": [], "l1": [], "probe": []}
    attempts_by_role: dict[str, int] = {}
    cue_rejections = 0
    split_rejections = 0
    balance_rejections = 0
    gold_counts: Counter = Counter()
    # switch+stable pooled member answers; max(1, ...) keeps smoke-scale builds
    # (tiny PER_ROLE in fixtures) from a zero budget while real cells get 20.
    per_role = PER_ROLE if per_role is None else per_role
    causal_gold_budget = max(1, int(BALANCE_CAP * 2 * 2 * per_role))

    for role_index, role in enumerate(ROLES):
        built = 0
        attempt = 0
        cap_attempts = per_role * 4000
        while built < per_role:
            attempt += 1
            if attempt > cap_attempts:
                raise RuntimeError(
                    f"{FAMILY}/{cell_name}/{role}: exhausted {cap_attempts} attempts "
                    f"at scene {built} (cue rejections {cue_rejections}, "
                    f"split {split_rejections}, balance {balance_rejections})")
            rng = attempt_rng(cell_name, role, attempt)
            kind = EXTREMUM_ROTATION[(role_index * per_role + built) % 4]
            geometry = build_coord_geometry(role, kind, n_points, rng)
            if geometry is None:
                continue
            spid = scene_program_id(sorted(geometry["points_a"].items()))
            if split_of(spid) != split:
                split_rejections += 1
                continue
            causal = role != "invariance"
            if causal:
                projected = gold_counts.copy()
                projected[str(geometry["answer_a"])] += 1
                projected[str(geometry["answer_b"])] += 1
                if max(projected.values()) > causal_gold_budget:
                    balance_rejections += 1
                    continue

            derive_l2_l1 = role != "target_switch"
            if derive_l2_l1:
                layers_a = render_coord_layers(geometry["points_a"], geometry["target_a"])
                layers_b = render_coord_layers(geometry["points_b"], geometry["target_b"])
                if layers_a is None or layers_b is None:
                    cue_rejections += 1
                    continue
                base_a, l1_a, cue_a = layers_a
                base_b, l1_b, cue_b = layers_b
            else:
                base_a = render_coordinate_scene(geometry["points_a"])
                base_b = render_coordinate_scene(geometry["points_b"])
                l1_a = l1_b = cue_a = cue_b = None
            scene_a = [[l, p[0], p[1]] for l, p in sorted(geometry["points_a"].items())]
            scene_b = [[l, p[0], p[1]] for l, p in sorted(geometry["points_b"].items())]
            negatives = coord_hard_negatives(geometry)

            swap = rng.random() < 0.5
            pair_key = (f"hier1_{FAMILY_TAG}_{cell_name}_{role}_{spid[-12:]}")

            def side(a_thing, b_thing):
                return (b_thing, a_thing) if swap else (a_thing, b_thing)

            image_a_l3, image_b_l3 = side(base_a, base_b)
            answer_a, answer_b = side(geometry["answer_a"], geometry["answer_b"])
            scene_side_a, scene_side_b = side(scene_a, scene_b)
            target_side_a, target_side_b = side(geometry["target_a"], geometry["target_b"])

            cell_dir = out_dir / FAMILY / cell_name
            sha_a_l3 = _save_image(image_a_l3, cell_dir / "images" / f"{pair_key}_a_l3.png")
            sha_b_l3 = _save_image(image_b_l3, cell_dir / "images" / f"{pair_key}_b_l3.png")
            sha_a_l2 = _save_image(image_a_l3, cell_dir / "images" / f"{pair_key}_a_l2.png")
            sha_b_l2 = _save_image(image_b_l3, cell_dir / "images" / f"{pair_key}_b_l2.png")
            mask_l3 = _mask(image_a_l3, image_b_l3, cell_dir / "masks" / f"{pair_key}_l3_mask.png")

            # Field names and fixed values as in the released manifests.
            common = {
                "schema_version": "lwl.hier-v1.pair.v1",
                "mother_item_id": pair_key,
                "family": FAMILY,
                "cell": cell_name,
                "role": role,
                "category": "hier_v1",
                "template_id": f"{FAMILY}_{cell_name}",
                "scene_program_id": spid,
                "split": split,
                "answers_equal": answer_a == answer_b,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "scene_a": scene_side_a,
                "scene_b": scene_side_b,
                "hard_negatives": negatives,
                "provenance": {
                    "generator": "scripts.build_scenes",
                    "batch_seed": BATCH_SEED,
                    "semantic_side_assignment_swapped": swap,
                    "rendered_text": SCENE_TEXT[FAMILY],
                },
            }
            vr_common = {
                "exact_by_construction": True,
                "role": role,
                "target_label_a": str(target_side_a),
                "target_label_b": str(target_side_b),
                "extremum_kind": geometry["extremum_kind"],
                "n_points": n_points,
                "extremum_margin": COORD_MARGIN,
            }

            rows_by_layer["l3"].append({
                **common,
                "pair_id": f"{pair_key}__l3",
                "layer": "l3",
                "question": geometry["questions"]["l3"],
                "image_a_path": str(cell_dir / "images" / f"{pair_key}_a_l3.png"),
                "image_b_path": str(cell_dir / "images" / f"{pair_key}_b_l3.png"),
                "image_a_sha256": sha_a_l3,
                "image_b_sha256": sha_b_l3,
                "changed_region_mask_a": str(cell_dir / "masks" / f"{pair_key}_l3_mask.png"),
                "changed_region_mask_b": str(cell_dir / "masks" / f"{pair_key}_l3_mask.png"),
                "mask_sha256": mask_l3,
                "verifier_results": {**vr_common, "layer": "l3", "oracle": "none"},
            })
            rows_by_layer["probe"].append({
                **common,
                "pair_id": f"{pair_key}__probe",
                "layer": "probe",
                "question": geometry["questions"]["probe"],
                "answer_a": str(target_side_a),
                "answer_b": str(target_side_b),
                "answers_equal": str(target_side_a) == str(target_side_b),
                "hard_negatives": None,
                "image_a_path": str(cell_dir / "images" / f"{pair_key}_a_l3.png"),
                "image_b_path": str(cell_dir / "images" / f"{pair_key}_b_l3.png"),
                "image_a_sha256": sha_a_l3,
                "image_b_sha256": sha_b_l3,
                "changed_region_mask_a": str(cell_dir / "masks" / f"{pair_key}_l3_mask.png"),
                "changed_region_mask_b": str(cell_dir / "masks" / f"{pair_key}_l3_mask.png"),
                "mask_sha256": mask_l3,
                "verifier_results": {**vr_common, "layer": "probe",
                                     "oracle": "none", "probe": "discovery"},
            })
            if derive_l2_l1:
                image_a_l1, image_b_l1 = side(l1_a, l1_b)
                cue_side_a, cue_side_b = side(cue_a, cue_b)
                sha_a_l1 = _save_image(image_a_l1, cell_dir / "images" / f"{pair_key}_a_l1.png")
                sha_b_l1 = _save_image(image_b_l1, cell_dir / "images" / f"{pair_key}_b_l1.png")
                mask_l1 = _mask(image_a_l1, image_b_l1,
                                cell_dir / "masks" / f"{pair_key}_l1_mask.png")
                for layer, q_key, ia, ib, sa, sb, mask, cue in (
                    ("l2", "l2", f"{pair_key}_a_l2.png", f"{pair_key}_b_l2.png",
                     sha_a_l2, sha_b_l2, mask_l3, None),
                    ("l1", "l2", f"{pair_key}_a_l1.png", f"{pair_key}_b_l1.png",
                     sha_a_l1, sha_b_l1, mask_l1,
                     {"a": cue_side_a, "b": cue_side_b}),
                ):
                    rows_by_layer[layer].append({
                        **common,
                        "pair_id": f"{pair_key}__{layer}",
                        "layer": layer,
                        "question": geometry["questions"][q_key],
                        "image_a_path": str(cell_dir / "images" / ia),
                        "image_b_path": str(cell_dir / "images" / ib),
                        "image_a_sha256": sa,
                        "image_b_sha256": sb,
                        "changed_region_mask_a": str(cell_dir / "masks" / f"{pair_key}_{'l3' if layer == 'l2' else 'l1'}_mask.png"),
                        "changed_region_mask_b": str(cell_dir / "masks" / f"{pair_key}_{'l3' if layer == 'l2' else 'l1'}_mask.png"),
                        "mask_sha256": mask,
                        "verifier_results": {
                            **vr_common, "layer": layer,
                            "oracle": "target_identity" if layer == "l2"
                            else "target_identity+location_cue",
                            **({"cue": cue} if cue else {}),
                        },
                    })
            if causal:
                gold_counts[str(geometry["answer_a"])] += 1
                gold_counts[str(geometry["answer_b"])] += 1
            built += 1
        attempts_by_role[role] = attempt

    total = sum(gold_counts.values())
    top_value, top_count = gold_counts.most_common(1)[0]
    balance = {
        "n_causal_member_golds": total,
        "answer_support_k": len(gold_counts),
        "max_share_value": top_value,
        "max_share": top_count / total,
        "cap": BALANCE_CAP,
        "enforced_budget_per_value": causal_gold_budget,
        # The enforced invariant is the integer budget; at full scale
        # (PER_ROLE=50, budget 20/200) it equals the 0.10 share cap. At
        # smoke scale the integer floor makes the share test meaningless.
        "pass": top_count <= causal_gold_budget,
    }
    if not balance["pass"]:
        raise AssertionError(f"balance cap violated: {balance}")

    return {
        "rows_by_layer": rows_by_layer,
        "attempts_by_role": attempts_by_role,
        "cue_rejections": cue_rejections,
        "split_rejections": split_rejections,
        "balance_rejections": balance_rejections,
        "balance": balance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="destination (default: the data directory's scenes/<split>)")
    parser.add_argument("--split", choices=("training", "development", "confirmatory"),
                        default="development",
                        help="scene-program bucket to generate")
    parser.add_argument("--per-role", type=int, default=None,
                        help="scene pairs per role per cell (default: PER_ROLE=50)")
    parser.add_argument("--report", type=Path, default=None,
                        help="build report (default: results/scenes_<split>_build.json)")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = data_path("scenes", args.split)
    if args.report is None:
        args.report = results_root() / f"scenes_{args.split}_build.json"
    # --out-dir/--report may be given relative; the recorded manifest paths need
    # them absolute.
    args.out_dir = args.out_dir.resolve()
    args.report = args.report.resolve()
    if args.report.exists():
        raise FileExistsError(f"{args.report} already exists")

    report: dict[str, Any] = {
        "schema_version": "lwl.scene-build.v1",
        "batch_seed": BATCH_SEED,
        "per_role": args.per_role or PER_ROLE,
        "split": args.split,
        "roles": ROLES,
        "layer_role_matrix": "l3 all roles; l2/l1 stable+invariance; probe all",
        "cells": {},
        "file_sha256": {},
    }
    for cell_name, n_points in CELLS:
        result = build_cell(cell_name, n_points, args.out_dir,
                            split=args.split, per_role=args.per_role)
        manifests = {}
        for layer, rows in result["rows_by_layer"].items():
            manifest = args.out_dir / f"manifest_{FAMILY}_{cell_name}_{layer}.jsonl"
            if manifest.exists():
                raise FileExistsError(manifest)
            blob = "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows)
            manifest.write_text(blob, encoding="utf-8")
            manifests[layer] = {
                "path": repo_relative(manifest),
                "rows": len(rows),
                "sha256": hashlib.sha256(blob.encode()).hexdigest(),
            }
        report["cells"][f"{FAMILY}/{cell_name}"] = {
            "cell_args": {"n_points": n_points},
            "manifests": manifests,
            "attempts_by_role": result["attempts_by_role"],
            "cue_rejections": result["cue_rejections"],
            "split_rejections": result["split_rejections"],
            "balance_rejections": result["balance_rejections"],
            "balance": result["balance"],
        }
        print(json.dumps({f"{FAMILY}/{cell_name}":
                          report["cells"][f"{FAMILY}/{cell_name}"]["attempts_by_role"]}))

    report["git_hash"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    report["command"] = " ".join(sys.argv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"cells": list(report["cells"])}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

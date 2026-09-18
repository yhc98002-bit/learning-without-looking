"""Build the grounding twin: the same three templates as the suite, regenerated from their own
seeds and checked for pair ids that overlap the suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from lwl.grounding.build_suite import (
    generate_coordinate_register_high_entropy_pairs,
    generate_header_table_pairs,
    generate_nine_series_chart_pairs,
    write_contact_sheets,
)
from lwl.grounding.schema import write_jsonl
from lwl.paths import data_path, results_root


TWIN_SEEDS = {
    "document": 20261001,
    "geometry": 20261002,
    "chart": 20261003,
}
TWIN_TEMPLATE_COUNTS = {
    "header_cued_table_code_v02": 300,
    "coordinate_register_twenty_point_x_v02": 600,
    "starred_series_value_nine_v07": 300,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pair_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {str(json.loads(line)["pair_id"]) for line in handle if line.strip()}


def build_twin(
    *,
    out_dir: Path,
    manifest: Path,
    contact_sheet_dir: Path,
    metadata_output: Path,
    suite_manifest: Path,
    template_counts: dict[str, int] | None = None,
    seeds: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Render the twin, write its manifest and contact sheets, and return its build record."""
    counts = dict(template_counts or TWIN_TEMPLATE_COUNTS)
    seeds = dict(seeds or TWIN_SEEDS)
    required_templates = set(TWIN_TEMPLATE_COUNTS)
    if set(counts) != required_templates or any(value <= 0 for value in counts.values()):
        raise ValueError("the twin needs positive counts for exactly the three suite templates")
    if set(seeds) != {"document", "geometry", "chart"} or len(set(seeds.values())) != 3:
        raise ValueError("the twin needs three distinct family seeds")
    if manifest.exists() or metadata_output.exists() or contact_sheet_dir.exists():
        raise FileExistsError("refusing to overwrite an existing twin build")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to write into a nonempty source directory: {out_dir}")

    rows = []
    rows.extend(
        generate_header_table_pairs(
            out_dir / "document",
            counts["header_cued_table_code_v02"],
            seeds["document"],
        )
    )
    rows.extend(
        generate_coordinate_register_high_entropy_pairs(
            out_dir / "geometry",
            counts["coordinate_register_twenty_point_x_v02"],
            seeds["geometry"],
        )
    )
    rows.extend(
        generate_nine_series_chart_pairs(
            out_dir / "chart",
            counts["starred_series_value_nine_v07"],
            seeds["chart"],
        )
    )
    observed_counts = dict(sorted(Counter(str(row["template_id"]) for row in rows).items()))
    if observed_counts != counts:
        raise RuntimeError(f"twin template count mismatch: expected {counts}, found {observed_counts}")
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("the twin generator produced duplicate pair IDs")
    overlap = sorted(set(pair_ids) & _read_pair_ids(suite_manifest))
    if overlap:
        raise RuntimeError(f"twin pair IDs overlap the suite: {overlap[:5]}")

    write_jsonl(manifest, rows)
    sheets = write_contact_sheets(rows, contact_sheet_dir)
    payload: dict[str, Any] = {
        "schema_version": "lwl.grounding-twin-build.v1",
        "status": "pass",
        "selection_applied": False,
        "regeneration_applied": False,
        "threshold_change_applied": False,
        "n_pairs": len(rows),
        "template_counts": observed_counts,
        "seeds": seeds,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "suite_pair_id_overlap": 0,
        "contact_sheets": [str(path) for path in sheets],
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path,
                        default=data_path("grounding", "twin_source", "renderable"))
    parser.add_argument("--manifest", type=Path,
                        default=data_path("grounding", "twin_source", "manifest.jsonl"))
    parser.add_argument(
        "--contact-sheet-dir", type=Path,
        default=results_root() / "contact_sheets" / "grounding_twin",
    )
    parser.add_argument(
        "--metadata-output", type=Path,
        default=data_path("grounding", "twin_source", "generation.json"),
    )
    parser.add_argument(
        "--suite-manifest",
        type=Path,
        default=data_path("grounding", "suite_source", "manifest.jsonl"),
        metavar="SUITE_MANIFEST",
        help="source manifest of the grounding suite; the twin may not share a pair id with it",
    )
    args = parser.parse_args()
    payload = build_twin(
        out_dir=args.out_dir,
        manifest=args.manifest,
        contact_sheet_dir=args.contact_sheet_dir,
        metadata_output=args.metadata_output,
        suite_manifest=args.suite_manifest,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

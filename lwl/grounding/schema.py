"""Record format for the grounding pair manifests, with the hashing and id helpers the
builders share.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "lwl.grounding-pair.v1"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: Any, length: int = 16) -> str:
    """Hex id derived from the given parts; identical parts always give the same id."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def pair_record(
    *,
    pair_id: str,
    image_a_path: str,
    image_b_path: str,
    question: str,
    answer_a: str,
    answer_b: str,
    category: str,
    template_id: str,
    provenance: dict[str, Any],
    changed_region_mask_a: str | None = None,
    changed_region_mask_b: str | None = None,
    verifier_results: dict[str, Any] | None = None,
    artifact_gate_score: float | None = None,
    catch_twin_id: str | None = None,
) -> dict[str, Any]:
    """One manifest row: two images differing in a single answer-relevant fact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "image_a_path": image_a_path,
        "image_b_path": image_b_path,
        "image_a_sha256": sha256_file(image_a_path),
        "image_b_sha256": sha256_file(image_b_path),
        "changed_region_mask_a": changed_region_mask_a,
        "changed_region_mask_b": changed_region_mask_b,
        "question": question,
        "answer_a": str(answer_a),
        "answer_b": str(answer_b),
        "category": category,
        "template_id": template_id,
        "provenance": provenance,
        "verifier_results": verifier_results or {},
        "artifact_gate_score": artifact_gate_score,
        "catch_twin_id": catch_twin_id,
    }

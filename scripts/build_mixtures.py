#!/usr/bin/env python3
"""Build the dose mixtures: every 240-row training step holds a share f of constructed rows, in
corpus order, and 1 - f of decontaminated ViRL39K rows drawn from a seeded permutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.paths import data_path, resolve_data_reference

CORPUS_COLUMNS = ("problem", "answer", "images", "pair_group_uid", "pair_member", "template_id", "category")
DEFAULT_CONSTRUCTED = data_path("training", "constructed", "train.jsonl")
DEFAULT_VIRL = data_path("training", "virl39k", "train.jsonl")
DEFAULT_VAL = data_path("training", "constructed", "plumbing_val.jsonl")
DEFAULT_SEED = 20260822
SCHEMA = "lwl.mixture-corpus.v1"
# Category of every ViRL row, as released.
VIRL_CATEGORY = "virl39k_m7"


class MixtureError(ValueError):
    pass


def mixture_plan(fraction: "float | str | Fraction", batch: int, steps: int) -> tuple[int, int, int]:
    """(constructed rows per window, virl rows per window, windows). f*batch must be an integer."""
    # Mixtures are exact rationals - 1/3 of a 240-row window is 80 rows, which no float literal
    # can express. A Fraction passes through untouched; a string is parsed exactly.
    fr = fraction if isinstance(fraction, Fraction) else Fraction(str(fraction))
    if not (0 <= fr <= 1):
        raise MixtureError(f"fraction must be in [0,1], got {fraction}")
    per = fr * batch
    if per.denominator != 1:
        raise MixtureError(f"fraction {fraction} x batch {batch} is not an integer ({per}); choose f with f*{batch} integral")
    h = int(per)
    return h, batch - h, steps


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def project_virl(row: dict) -> dict:
    qid = row.get("qid")
    if qid is None:
        raise MixtureError("ViRL row without qid")
    return {"problem": row["problem"], "answer": str(row["answer"]), "images": list(row["images"]),
            "pair_group_uid": f"virl:{qid}", "pair_member": "virl", "template_id": "virl39k", "category": VIRL_CATEGORY}


def project_constructed(row: dict) -> dict:
    return {k: row[k] for k in CORPUS_COLUMNS}


def build_rows(constructed_rows: list[dict], virl_rows: list[dict], *, fraction: "float | str | Fraction", batch: int, steps: int,
               seed: int) -> tuple[list[dict], dict]:
    """Deterministic per-window mixture; the constructed side cycles in corpus order
    from the head, ViRL is drawn without replacement from a seeded permutation."""
    h_per, v_per, n_win = mixture_plan(fraction, batch, steps)
    need_v = v_per * n_win
    if need_v > len(virl_rows):
        raise MixtureError(f"need {need_v} ViRL rows, pool has {len(virl_rows)}")
    if h_per and not constructed_rows:
        raise MixtureError("constructed-scene pool empty")
    rng = random.Random(seed)
    order = list(range(len(virl_rows)))
    rng.shuffle(order)
    v_iter = iter(order)
    out: list[dict] = []
    comp = []
    hi = 0
    for w in range(n_win):
        hs = []
        for _ in range(h_per):
            hs.append(project_constructed(constructed_rows[hi % len(constructed_rows)])); hi += 1
        vs = [project_virl(virl_rows[next(v_iter)]) for _ in range(v_per)]
        win = hs + vs                      # constructed rows first inside the window (group adjacency kept), then ViRL
        out.extend(win)
        comp.append({"window": w, "constructed_rows": len(hs), "virl_rows": len(vs),
                     "constructed_row_offset_start": (w * h_per) % len(constructed_rows) if constructed_rows else None})
    _fr = fraction if isinstance(fraction, Fraction) else Fraction(str(fraction))
    stats = {"fraction": str(_fr), "fraction_float": float(_fr), "batch": batch, "steps": n_win, "constructed_rows_per_window": h_per,
             "virl_rows_per_window": v_per, "n_rows": len(out), "constructed_rows_total": h_per * n_win,
             "virl_rows_total": need_v, "constructed_pool": len(constructed_rows), "virl_pool": len(virl_rows),
             "constructed_cycles": (h_per * n_win) / len(constructed_rows) if constructed_rows else 0.0, "seed": seed,
             "windows": comp}
    return out, stats


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fraction", type=str, required=True,
                    help="resolvable (constructed-scene) fraction per step as an exact rational or "
                         "decimal, e.g. 0.75, 1/3, 80/240 (it must make f x batch integral)")
    ap.add_argument("--batch", type=int, default=240)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--constructed", type=Path, default=DEFAULT_CONSTRUCTED, metavar="CONSTRUCTED_ROWS",
                    help="training rows of the constructed scenes")
    ap.add_argument("--virl", type=Path, default=DEFAULT_VIRL)
    ap.add_argument("--plumbing-val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--skip-image-check", action="store_true")
    a = ap.parse_args()
    if a.out_dir.exists() or a.report.exists():
        raise SystemExit(f"{a.out_dir} / {a.report} already exists")
    constructed = read_jsonl(a.constructed); virl = read_jsonl(a.virl)
    try:
        frac = Fraction(a.fraction)
    except (ValueError, ZeroDivisionError) as exc:
        raise MixtureError(f"--fraction {a.fraction!r} is not a number or exact rational: {exc}") from exc
    rows, stats = build_rows(constructed, virl, fraction=frac, batch=a.batch, steps=a.steps, seed=a.seed)
    if not a.skip_image_check:
        missing = [r["images"][0] for r in rows if not resolve_data_reference(r["images"][0]).is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} image files missing, e.g. {missing[:3]}")
    for r in rows:
        if r["problem"].count("<image>") != len(r["images"]):
            raise SystemExit(f"image marker/count mismatch: {r['pair_group_uid']}")
    a.out_dir.mkdir(parents=True)
    blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (a.out_dir / "train.jsonl").write_text(blob, encoding="utf-8")
    parquet = False
    try:
        import pyarrow as pa, pyarrow.parquet as pq  # noqa: E401
        schema = pa.schema([("problem", pa.string()), ("answer", pa.string()), ("images", pa.list_(pa.string())),
                            ("pair_group_uid", pa.string()), ("pair_member", pa.string()),
                            ("template_id", pa.string()), ("category", pa.string())])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), a.out_dir / "train.parquet"); parquet = True
    except Exception:  # noqa: BLE001
        parquet = False
    val_blob = a.plumbing_val.read_text(encoding="utf-8")
    (a.out_dir / "plumbing_val.jsonl").write_text(val_blob, encoding="utf-8")
    report = {"schema_version": SCHEMA,
              "out_dir": str(a.out_dir), "constructed_source": str(a.constructed), "virl_source": str(a.virl),
              "constructed_source_sha256": hashlib.sha256(a.constructed.read_bytes()).hexdigest(),
              "virl_source_sha256": hashlib.sha256(a.virl.read_bytes()).hexdigest(),
              "train_jsonl_sha256": sha256_text(blob), "plumbing_val_sha256": sha256_text(val_blob),
              "plumbing_val_source": str(a.plumbing_val), "parquet": parquet, "columns": list(CORPUS_COLUMNS),
              "virl_qids": sorted({r["pair_group_uid"][5:] for r in rows if r["pair_member"] == "virl"}),
              **stats}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {a.out_dir} ({len(rows)} rows; f={stats['fraction']}; scenes/win={stats['constructed_rows_per_window']} virl/win={stats['virl_rows_per_window']}) and {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

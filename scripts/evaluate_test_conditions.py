#!/usr/bin/env python3
"""Evaluate one trained checkpoint under several test-time image conditions.

One call to scripts/evaluate_geometry3k.py per condition, at the decoding the paper
reports. A condition is skipped only when its predictions hold all 601 rows; an interrupted
one resumes from its complete rows, which evaluate_geometry3k.py checks before keeping.
The GPU is chosen with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lwl.evaluation.conditioned_inputs import CONDITIONS

DEFAULT_CONDITIONS = ("real", "gray", "none")
EXPECTED_ROWS = 601
SEED = 20260710
MAX_TOKENS = 2048
BATCH_SIZE = 4
MAX_MODEL_LEN = 8192
GLOBAL_STEP = 100


def cell_command(args: argparse.Namespace, condition: str, run_dir: Path) -> list[str]:
    """The evaluation command for one model x condition cell."""
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_geometry3k.py"),
        "--arm", args.arm,
        "--condition", condition,
        "--model-path", str(args.model_path),
        "--manifest", str(args.manifest),
        "--format-prompt", str(args.format_prompt),
        "--output", str(run_dir / "predictions.jsonl"),
        "--cache-dir", str(run_dir / "cache"),
        "--run-manifest", str(args.run_manifest),
        "--source-training-manifest", str(args.source_training_manifest),
        "--checkpoint-index-sha256", args.checkpoint_index_sha256,
        "--batch-size", str(args.batch_size),
        "--max-model-len", str(args.max_model_len),
        "--max-tokens", str(MAX_TOKENS),
        "--seed", str(SEED),
        "--global-step", str(args.global_step),
    ]
    if condition == "caption":
        command.append("--caption-shards")
        command.extend(str(shard) for shard in args.caption_shards)
    return command


def complete_rows(path: Path) -> list[str]:
    """Rows of a predictions file, without a last line that an interruption cut short."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [line for line in handle if line.endswith("\n") and line.strip()]


def count_rows(path: Path) -> int:
    return len(complete_rows(path))


def set_aside(predictions: Path) -> Path | None:
    """Move an interrupted predictions file to a resume file and return it, or None if empty."""
    resume = predictions.with_name("predictions.resume.jsonl")
    if predictions.is_file():
        resume.write_text("".join(complete_rows(predictions)), encoding="utf-8")
        predictions.unlink()
    return resume if count_rows(resume) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--format-prompt", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--source-training-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-index-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--caption-shards", type=Path, nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--global-step", type=int, default=GLOBAL_STEP)
    args = parser.parse_args()

    unknown = [condition for condition in args.conditions if condition not in CONDITIONS]
    if unknown:
        raise ValueError(f"unsupported image conditions: {unknown}")
    if "caption" in args.conditions and not args.caption_shards:
        raise ValueError("the caption condition requires the question-blind caption store")
    if "caption" not in args.conditions and args.caption_shards:
        raise ValueError("caption shards are only valid for the caption condition")

    status = 0
    for condition in args.conditions:
        run_dir = args.out / condition
        predictions = run_dir / "predictions.jsonl"
        if count_rows(predictions) == EXPECTED_ROWS:
            print(f"skip {condition}: {predictions} is complete")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        command = cell_command(args, condition, run_dir)
        resume = set_aside(predictions)
        if resume is not None:
            command.extend(["--resume-from", str(resume)])
            print(f"resume {condition} from {count_rows(resume)} rows", flush=True)
        else:
            print(f"run {condition}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(f"{condition}: evaluation exited {completed.returncode}", file=sys.stderr)
            if resume is not None:
                print(f"{condition}: the rows it resumed from are kept in {resume}", file=sys.stderr)
            status = 1
            continue
        rows = count_rows(predictions)
        if rows < EXPECTED_ROWS:
            print(f"{condition}: {rows} rows written, expected {EXPECTED_ROWS}", file=sys.stderr)
            status = 1
        else:
            if resume is not None:
                resume.unlink()
            print(f"{condition}: {rows} rows in {predictions}")
    sys.exit(status)


if __name__ == "__main__":
    main()

"""The standard training reward: answer correctness and response format combined at
``format_weight``, one response at a time. The Geometry3K and ViRL39K access runs, the
constructed-corpus runs and the dose mixtures use it; the long-horizon runs use EasyR1's own r1v
reward. The other returned fields are logged for comparison and never enter training.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, TypeVar

from mathruler.grader import grade_answer

from lwl.evaluation.prompt_contract import response_satisfies_contract
from lwl.paths import repo_root
from lwl.rewards.answer import (
    PARSER_VERSION,
    answers_match,
    extract_answer_span,
)


REWARD_NAME = "lwl_pilot_v1"
REWARD_TYPE = "sequential"
PILOT_REWARD_VERSION = "pilot-reward-v1"
SYMBOLIC_GRADER_GUARD_VERSION = "posix-itimer-v1"
DEFAULT_SYMBOLIC_GRADER_TIMEOUT_SECONDS = 5.0
NATIVE_R1V_PATH = (
    repo_root() / "artifacts" / "repos" / "EasyR1" / "examples" / "reward_function" / "r1v.py"
)
REASON_CODES = {
    "none": 0.0,
    "canonical_correct_mathruler_incorrect": 1.0,
    "mathruler_correct_canonical_incorrect": 2.0,
    "mathruler_error_canonical_incorrect": 3.0,
    "mathruler_error_canonical_correct": 4.0,
}

_T = TypeVar("_T")


class SymbolicGraderTimeout(TimeoutError):
    pass


@contextmanager
def _symbolic_grader_deadline(seconds: float) -> Iterator[None]:
    if seconds <= 0:
        raise ValueError("symbolic grader timeout must be positive")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("symbolic grader timeout requires the process main thread")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise SymbolicGraderTimeout(
            f"symbolic grader exceeded {seconds:.3f} seconds"
        )

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0.0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(previous_delay - elapsed, 1e-6),
                previous_interval,
            )


def _bounded_call(call: Callable[[], _T], timeout_seconds: float) -> _T:
    with _symbolic_grader_deadline(timeout_seconds):
        return call()


@lru_cache(maxsize=1)
def load_native_r1v() -> ModuleType:
    if not NATIVE_R1V_PATH.is_file():
        raise FileNotFoundError(f"native EasyR1 r1v reward is absent: {NATIVE_R1V_PATH}")
    spec = importlib.util.spec_from_file_location("lwl_native_r1v_shadow", NATIVE_R1V_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load native EasyR1 r1v reward: {NATIVE_R1V_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mathruler_grade(
    answer: str, ground_truth: str, timeout_seconds: float
) -> tuple[bool, str | None]:
    try:
        return bool(
            _bounded_call(
                lambda: grade_answer(answer, ground_truth),
                timeout_seconds,
            )
        ), None
    except Exception as error:  # pragma: no cover - depends on symbolic parser internals
        return False, type(error).__name__


def _native_r1v_shadow(
    response: str,
    ground_truth: str,
    format_weight: float,
    timeout_seconds: float,
) -> tuple[float, str | None]:
    try:
        score = _bounded_call(
            lambda: load_native_r1v().compute_score(
                {"response": response, "ground_truth": ground_truth},
                format_weight=format_weight,
            ),
            timeout_seconds,
        )
        return float(score["overall"]), None
    except Exception as error:  # shadow failures must not change the optimized reward
        return 0.0, type(error).__name__


def _disagreement_reason(
    *, mathruler_correct: bool, canonical_correct: bool, mathruler_error: str | None
) -> str:
    if mathruler_error is not None:
        return (
            "mathruler_error_canonical_correct"
            if canonical_correct
            else "mathruler_error_canonical_incorrect"
        )
    if mathruler_correct == canonical_correct:
        return "none"
    return (
        "mathruler_correct_canonical_incorrect"
        if mathruler_correct
        else "canonical_correct_mathruler_incorrect"
    )


def grade_response_accuracy(
    response: str,
    ground_truth: str,
    *,
    symbolic_grader_timeout_seconds: float = DEFAULT_SYMBOLIC_GRADER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Extract the answer span, grade it both ways, and name any disagreement."""

    extracted = extract_answer_span(response)
    mathruler_correct, mathruler_error = _mathruler_grade(
        extracted.span,
        ground_truth,
        symbolic_grader_timeout_seconds,
    )
    canonical_correct = bool(answers_match(extracted.span, ground_truth))
    reason = _disagreement_reason(
        mathruler_correct=mathruler_correct,
        canonical_correct=canonical_correct,
        mathruler_error=mathruler_error,
    )
    return {
        "extracted": extracted,
        "mathruler_correct": mathruler_correct,
        "mathruler_error": mathruler_error,
        "canonical_correct": canonical_correct,
        "reward_disagreement_reason": reason,
    }


def _append_shadow(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def compute_score(
    reward_input: dict[str, Any],
    format_weight: float = 0.5,
    shadow_log_path: str | None = None,
    require_shadow_log: bool = False,
    symbolic_grader_timeout_seconds: float = DEFAULT_SYMBOLIC_GRADER_TIMEOUT_SECONDS,
) -> dict[str, float]:
    """Score one response as (1 - format_weight) * accuracy + format_weight * format.

    The canonical parser extracts the answer span and mathruler grades that span. Where
    mathruler and canonical numeric equivalence disagree, mathruler's verdict is the
    accuracy term and the disagreement is recorded.
    """

    if not 0.0 <= format_weight <= 1.0:
        raise ValueError(f"format_weight must be in [0, 1], found {format_weight}")
    response = str(reward_input["response"])
    ground_truth = str(reward_input["ground_truth"]).strip()
    grade = grade_response_accuracy(
        response,
        ground_truth,
        symbolic_grader_timeout_seconds=symbolic_grader_timeout_seconds,
    )
    extracted = grade["extracted"]
    mathruler_correct = bool(grade["mathruler_correct"])
    mathruler_error = grade["mathruler_error"]
    canonical_correct = bool(grade["canonical_correct"])
    contract_valid = bool(response_satisfies_contract(response))
    reason = str(grade["reward_disagreement_reason"])
    accuracy_reward = float(mathruler_correct)
    format_reward = float(contract_valid)
    training_reward = (1.0 - format_weight) * accuracy_reward + format_weight * format_reward
    native_shadow, native_shadow_error = _native_r1v_shadow(
        response,
        ground_truth,
        format_weight,
        symbolic_grader_timeout_seconds,
    )
    canonical_shadow = float(canonical_correct)

    resolved_shadow_path = shadow_log_path or os.environ.get("LWL_REWARD_SHADOW_LOG")
    if require_shadow_log and not resolved_shadow_path:
        raise RuntimeError(
            "the reward requires LWL_REWARD_SHADOW_LOG or shadow_log_path"
        )
    if resolved_shadow_path:
        _append_shadow(
            Path(resolved_shadow_path),
            {
                "schema_version": "lwl.pilot-reward-shadow.v1",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pid": os.getpid(),
                "pilot_reward_version": PILOT_REWARD_VERSION,
                "symbolic_grader_guard_version": SYMBOLIC_GRADER_GUARD_VERSION,
                "symbolic_grader_timeout_seconds": symbolic_grader_timeout_seconds,
                "parser_version": PARSER_VERSION,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "ground_truth": ground_truth,
                "extracted_answer": extracted.span,
                "extraction_level": extracted.extraction_level,
                "extractor_valid": extracted.extractor_valid,
                "contract_valid": contract_valid,
                "mathruler_accuracy_reward": accuracy_reward,
                "mathruler_error": mathruler_error,
                "training_reward": training_reward,
                "native_r1v_shadow_reward": native_shadow,
                "native_r1v_shadow_error": native_shadow_error,
                "native_r1v_shadow_valid": native_shadow_error is None,
                "canonical_eval_reward": canonical_shadow,
                "reward_disagreement_reason": reason,
            },
        )

    return {
        "overall": training_reward,
        "format": format_reward,
        "accuracy": accuracy_reward,
        "training_reward": training_reward,
        "native_r1v_shadow_reward": native_shadow,
        "native_r1v_shadow_valid": float(native_shadow_error is None),
        "canonical_eval_reward": canonical_shadow,
        "reward_disagreement": float(reason != "none"),
        "reward_disagreement_reason_code": REASON_CODES[reason],
    }

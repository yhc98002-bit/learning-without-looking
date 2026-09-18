"""Answer extraction and comparison.

Pulls the final answer out of a response - answer tags first, then a boxed span, then
a trailing "answer:" line - and decides whether it matches the reference, allowing for
presentation, units and numeric equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from fractions import Fraction
from typing import Any


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:：]\s*(.*)$", re.IGNORECASE)
PARSER_VERSION = "canonical-v2"

_UNIT_PATTERNS = (
    (re.compile(r"square\s+centimeters?|cm\^?2", re.IGNORECASE), "cm2"),
    (re.compile(r"square\s+meters?|m\^?2", re.IGNORECASE), "m2"),
    (re.compile(r"square\s+inches?|in\^?2", re.IGNORECASE), "in2"),
    (re.compile(r"square\s+feet|ft\^?2", re.IGNORECASE), "ft2"),
    (re.compile(r"millimeters?|mm", re.IGNORECASE), "mm"),
    (re.compile(r"centimeters?|cm", re.IGNORECASE), "cm"),
    (re.compile(r"kilometers?|km", re.IGNORECASE), "km"),
    (re.compile(r"meters?|metres?|m", re.IGNORECASE), "m"),
    (re.compile(r"inches?|in", re.IGNORECASE), "in"),
    (re.compile(r"feet|foot|ft", re.IGNORECASE), "ft"),
    (re.compile(r"degrees?|deg", re.IGNORECASE), "degree"),
)


@dataclass(frozen=True)
class ExtractedAnswer:
    span: str
    extraction_level: str
    extraction_fallback_used: bool
    extractor_valid: bool
    parser_version: str = PARSER_VERSION


def _balanced_boxed_spans(text: str) -> list[str]:
    spans: list[str] = []
    needle = r"\boxed{"
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            return spans
        pos = idx + len(needle)
        depth = 1
        out: list[str] = []
        while pos < len(text):
            char = text[pos]
            if char == "{":
                depth += 1
                out.append(char)
            elif char == "}":
                depth -= 1
                if depth == 0:
                    spans.append("".join(out).strip())
                    break
                out.append(char)
            else:
                out.append(char)
            pos += 1
        start = idx + len(needle)


def extract_answer_span(text: Any) -> ExtractedAnswer:
    text = str(text).strip()
    tagged = ANSWER_TAG_RE.findall(text)
    if tagged:
        return ExtractedAnswer(tagged[-1].strip(), "tag", False, True)
    boxed = _balanced_boxed_spans(text)
    if boxed:
        return ExtractedAnswer(boxed[-1].strip(), "boxed", False, True)
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        match = FINAL_RE.search(line)
        if match:
            return ExtractedAnswer(match.group(1).strip(), "line", False, True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return ExtractedAnswer(lines[-1], "lastline", True, False)
    return ExtractedAnswer(text, "fulltext", True, False)


def extract_final_answer(text: Any) -> str:
    return extract_answer_span(text).span


def numeric_value(text: str) -> float | None:
    cleaned = text.strip().lower().replace(",", "")
    cleaned = cleaned.replace("$", "")
    percent = cleaned.endswith("%")
    cleaned = cleaned[:-1] if percent else cleaned
    frac_match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", cleaned)
    if frac_match:
        try:
            numerator = float(Fraction(frac_match.group(1)))
            denominator = float(Fraction(frac_match.group(2)))
            if denominator == 0:
                return None
            value = numerator / denominator
            return value / 100.0 if percent else value
        except Exception:
            return None
    try:
        value = float(Fraction(cleaned))
        return value / 100.0 if percent else value
    except Exception:
        return None


def _unwrap_presentation(text: str) -> str:
    previous = None
    while text != previous:
        previous = text
        stripped = text.strip()
        wrappers = (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
        for opening, closing in wrappers:
            if stripped.startswith(opening) and stripped.endswith(closing):
                inner = stripped[len(opening) : len(stripped) - len(closing)].strip()
                if inner:
                    text = inner
                    break
        else:
            text = stripped
        text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    return text


def normalize_presentation(value: Any) -> str:
    text = _unwrap_presentation(str(value))
    text = re.sub(r"\\(?:left|right)\s*", "", text)
    text = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", " degrees", text)
    text = text.replace("°", " degrees")
    text = re.sub(r"\\(?:,|;|!|quad\b|qquad\b)", " ", text)
    text = re.sub(r"\\([A-Za-z]+)\s+\{", r"\\\1{", text)
    text = re.sub(r"\\sqrt\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))\b", r"\\sqrt{\1}", text)
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+\}", "}", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Whitespace around TeX structure is presentation-only, including implicit multiplication.
    text = re.sub(r"\s*(\\[A-Za-z]+|[{}^_])\s*", r"\1", text)
    return text


def _split_unit_suffix(text: str) -> tuple[str, str | None]:
    for pattern, canonical in _UNIT_PATTERNS:
        match = re.fullmatch(rf"(.+?)\s*(?:{pattern.pattern})", text, re.IGNORECASE)
        if not match:
            continue
        core = match.group(1).strip()
        # Unit stripping is only valid for an answer-like numeric or mathematical span.
        if re.search(r"\d|\\(?:frac|sqrt)", core):
            return core, canonical
    return text, None


def normalize_text(value: Any) -> str:
    text = normalize_presentation(value)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"[.,;:!?]+$", "", text)
    return text


def normalize_answer(value: Any) -> str:
    return normalize_text(extract_final_answer(value))


def answers_match(prediction: Any, gold: Any, numeric_tol: float = 1e-4) -> bool:
    pred = normalize_answer(prediction)
    target = normalize_text(gold)
    if pred == target:
        return True
    pred_core, pred_unit = _split_unit_suffix(pred)
    target_core, target_unit = _split_unit_suffix(target)
    if pred_unit is not None and target_unit is not None and pred_unit != target_unit:
        return False
    if pred_unit is not None or target_unit is not None:
        pred = pred_core
        target = target_core
        if pred == target:
            return True
    pred_num = numeric_value(pred)
    target_num = numeric_value(target)
    if pred_num is not None and target_num is not None:
        return math.isclose(pred_num, target_num, rel_tol=numeric_tol, abs_tol=numeric_tol)
    return False


def answer_reward(prediction: Any, gold: Any) -> float:
    return 1.0 if answers_match(prediction, gold) else 0.0

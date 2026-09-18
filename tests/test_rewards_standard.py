"""The single-response training reward: the accuracy and format split, and its shadow log."""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture(scope="module")
def standard():
    """The reward module; a missing dependency fails these tests without stopping the others."""
    try:
        return importlib.import_module("lwl.rewards.standard")
    except ImportError as error:
        pytest.fail(f"lwl.rewards.standard cannot be imported: {error}", pytrace=False)


@pytest.fixture(autouse=True)
def no_shadow_log_from_the_environment(monkeypatch):
    monkeypatch.delenv("LWL_REWARD_SHADOW_LOG", raising=False)


def score(module, response, ground_truth="7", **kwargs):
    return module.compute_score({"response": response, "ground_truth": ground_truth}, **kwargs)


# (response, accuracy, format): the default reward is 0.5 * accuracy + 0.5 * format.
SPLIT_CASES = [
    ("<answer>7</answer>", 1.0, 1.0, "a right answer in one tag"),
    (r"work, so \boxed{7}", 1.0, 0.0, "a right answer without tags"),
    ("<answer>8</answer>", 0.0, 1.0, "a wrong answer in one tag"),
    ("the answer is seven", 0.0, 0.0, "a wrong answer without tags"),
    ("<answer>7</answer><answer>7</answer>", 1.0, 0.0, "a right answer in two tags"),
    ("<answer></answer>", 0.0, 0.0, "an empty tag"),
]


@pytest.mark.parametrize("response, accuracy, format_score, description", SPLIT_CASES)
def test_accuracy_and_format_are_weighted_evenly_by_default(
    standard, response, accuracy, format_score, description
):
    scored = score(standard, response)
    assert (scored["accuracy"], scored["format"]) == (accuracy, format_score), (
        f"{description}: accuracy and format came out {scored['accuracy']} and "
        f"{scored['format']}, expected {accuracy} and {format_score}"
    )
    assert scored["overall"] == pytest.approx(0.5 * accuracy + 0.5 * format_score), (
        f"{description}: the reward came out {scored['overall']}, expected "
        f"{0.5 * accuracy + 0.5 * format_score}"
    )


@pytest.mark.parametrize("weight", [0.0, 0.25, 1.0])
def test_the_format_weight_moves_the_whole_reward(standard, weight):
    scored = score(standard, r"\boxed{7}", format_weight=weight)
    assert (scored["accuracy"], scored["format"]) == (1.0, 0.0), (
        "the response is right and untagged, so accuracy 1.0 and format 0.0 are expected, got "
        f"{scored['accuracy']} and {scored['format']}"
    )
    assert scored["overall"] == pytest.approx(1.0 - weight), (
        f"at weight {weight} the reward should be {1.0 - weight}, got {scored['overall']}"
    )


@pytest.mark.parametrize("weight", [-0.1, 1.5])
def test_a_weight_outside_the_unit_interval_is_rejected(standard, weight):
    with pytest.raises(ValueError, match="format_weight"):
        score(standard, "<answer>7</answer>", format_weight=weight)


def test_the_symbolic_grader_decides_and_a_disagreement_is_recorded(standard):
    scored = score(standard, "<answer>50%</answer>", ground_truth="0.5")
    assert scored["accuracy"] == 0.0, (
        f"the symbolic grader rejects 50% for 0.5 and its verdict is the reward, got {scored['accuracy']}"
    )
    assert scored["overall"] == pytest.approx(0.5), (
        f"a tagged answer the symbolic grader rejects earns the format half only, got {scored['overall']}"
    )
    assert scored["canonical_eval_reward"] == 1.0, (
        f"the evaluation matcher accepts 50% for 0.5, got {scored['canonical_eval_reward']}"
    )
    assert scored["reward_disagreement"] == 1.0, "the two verdicts differ and should be flagged"
    assert scored["reward_disagreement_reason_code"] == standard.REASON_CODES[
        "canonical_correct_mathruler_incorrect"
    ], f"unexpected disagreement code {scored['reward_disagreement_reason_code']}"
    agreed = score(standard, "<answer>7</answer>")
    assert (agreed["reward_disagreement"], agreed["reward_disagreement_reason_code"]) == (0.0, 0.0), (
        "a right answer both graders accept should carry no disagreement"
    )


def test_a_missing_framework_reward_does_not_change_the_training_reward(standard):
    scored = score(standard, "<answer>7</answer>")
    if standard.NATIVE_R1V_PATH.is_file():
        pytest.skip("the training framework is installed next to the repository")
    assert scored["native_r1v_shadow_valid"] == 0.0, (
        "without the training framework the shadow reward should be marked invalid"
    )
    assert scored["overall"] == scored["training_reward"] == 1.0, (
        f"the training reward should be unaffected, got {scored['overall']}"
    )


def test_the_returned_fields_are_the_ones_the_trainer_reads(standard):
    scored = score(standard, "<answer>7</answer>")
    expected = {
        "overall",
        "format",
        "accuracy",
        "training_reward",
        "native_r1v_shadow_reward",
        "native_r1v_shadow_valid",
        "canonical_eval_reward",
        "reward_disagreement",
        "reward_disagreement_reason_code",
    }
    assert set(scored) == expected, f"unexpected reward fields: {sorted(set(scored) ^ expected)}"


def test_the_shadow_log_is_required_when_the_caller_asks_for_it(standard):
    with pytest.raises(RuntimeError, match="LWL_REWARD_SHADOW_LOG"):
        score(standard, "<answer>7</answer>", require_shadow_log=True)


def test_the_shadow_log_records_one_line_per_response(standard, tmp_path):
    path = tmp_path / "shadow.jsonl"
    returned = score(standard, "<answer>7</answer>", shadow_log_path=str(path), require_shadow_log=True)
    score(standard, "<answer>8</answer>", shadow_log_path=str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, f"two responses should leave two lines, found {len(lines)}"
    first = json.loads(lines[0])
    for field in (
        "extracted_answer",
        "ground_truth",
        "contract_valid",
        "mathruler_accuracy_reward",
        "canonical_eval_reward",
        "training_reward",
        "parser_version",
        "response_sha256",
    ):
        assert field in first, f"the shadow record is missing {field}: {sorted(first)}"
    assert first["extracted_answer"] == "7", (
        f"the record should hold the extracted answer, got {first['extracted_answer']!r}"
    )
    assert first["training_reward"] == returned["overall"], (
        f"the record holds {first['training_reward']}, the caller received {returned['overall']}"
    )
    assert json.loads(lines[1])["mathruler_accuracy_reward"] == 0.0, (
        "the second response was wrong and its record should say so"
    )


def test_the_environment_variable_names_the_shadow_log(standard, tmp_path, monkeypatch):
    path = tmp_path / "from_env.jsonl"
    monkeypatch.setenv("LWL_REWARD_SHADOW_LOG", str(path))
    score(standard, "<answer>7</answer>", require_shadow_log=True)
    assert path.exists(), f"the reward should have written to {path}"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, f"one response should leave one line, found {len(lines)}"

"""The tiered answer matcher, pair scoring, the collapse flag and the tests that judge a run."""
from __future__ import annotations

import math

import pytest

from lwl.evaluation.metrics import (
    aggregate_pair_metrics,
    bootstrap_ci,
    golds_equivalent,
    is_correct,
    match_tier,
    mcnemar_exact,
    pair_score,
    permutation_null_pair_accuracy,
)


# (span, reference, tier): 2 is an exact or equivalent answer, 1 a sentence that carries the
# reference, 0 no match.
MATCH_CASES = [
    ("5", "5", 2, "identical text"),
    ("b2", "B2", 2, "a label in another case"),
    ("5.0", "5", 2, "the same number written differently"),
    ("+5", "5", 2, "an explicit plus sign"),
    ("50%", "0.5", 2, "a percentage against its decimal"),
    (r"\frac{1}{2}", "0.5", 2, "a fraction against its decimal"),
    ("30\u00b0", "30", 2, "a degree sign"),
    ("5", "5 cm", 2, "a bare number against a reference with a unit"),
    ("0.50004", "0.5", 2, "a difference inside the numeric tolerance"),
    ("0.5003", "0.5", 0, "a difference outside the numeric tolerance"),
    ("1000.05", "1000", 2, "a difference inside the relative tolerance of a large number"),
    ("1000.2", "1000", 0, "a difference outside the relative tolerance of a large number"),
    ("x = 5 m", "5", 1, "the number carried by a longer sentence"),
    ("the answer is B2", "B2", 1, "a label carried by a longer sentence"),
    ("-1", "1", 0, "a sign flip on the prediction"),
    ("1", "-1", 0, "a sign flip on the reference"),
    ("x = -5", "5", 0, "a sign flip inside a sentence"),
    ("-5 m", "5", 0, "a sign flip in front of a unit"),
    ("15", "5", 0, "a number that only ends with the reference"),
    ("B25", "B2", 0, "a label that only starts with the reference"),
    ("5 cm", "5 m", 0, "the same number with a different unit"),
    ("two", "2", 0, "a spelled-out number"),
    ("", "5", 0, "an empty span"),
    ("   ", "5", 0, "a whitespace-only span"),
    ("5", "", 0, "an empty reference"),
]


@pytest.mark.parametrize("span, gold, expected, description", MATCH_CASES)
def test_match_tier_table(span, gold, expected, description):
    tier = match_tier(span, gold)
    assert tier == expected, (
        f"{description}: match_tier({span!r}, {gold!r}) gave tier {tier}, expected {expected}"
    )


# Whole responses: the answer span is extracted first, then matched.
RESPONSE_CASES = [
    ("<answer>5</answer>", "5", True, "one well-formed tag"),
    ("<ANSWER>5</ANSWER>", "5", True, "tags in upper case"),
    ("<answer>3</answer> <answer>5</answer>", "5", True, "two tags, the last of which is right"),
    ("<answer>3</answer> <answer>5</answer>", "3", False, "two tags, only the first of which is right"),
    ("<answer>5</answer", "5", True, "a truncated closing tag around a visible number"),
    ("<answer>15</answer", "5", False, "a truncated closing tag around a longer number"),
    ("<answer></answer>", "5", False, "an empty tagged span"),
    ("<answer>   </answer>", "5", False, "a whitespace-only tagged span"),
    ("<answer></answer>\n5", "5", False, "an empty tag followed by the number outside it"),
    ("", "5", False, "an empty response"),
    ("<answer>-5</answer>", "5", False, "the reference with its sign flipped"),
]


@pytest.mark.parametrize("response, gold, expected, description", RESPONSE_CASES)
def test_is_correct_on_whole_responses(response, gold, expected, description):
    verdict = is_correct(response, gold)
    assert verdict is expected, (
        f"{description}: is_correct({response!r}, {gold!r}) gave {verdict}, expected {expected}"
    )


def test_golds_equivalent_marks_the_pairs_that_share_one_answer():
    assert golds_equivalent("3", "3.0"), "3 and 3.0 are the same answer"
    assert golds_equivalent("B2", "b2"), "B2 and b2 are the same label"
    assert not golds_equivalent("3", "-3"), "3 and -3 are different answers"
    assert not golds_equivalent("", "3"), "an empty answer is never equivalent to another"


def pair(answer_a, answer_b, prediction_a, prediction_b, pair_id="p1"):
    """One pair in the shape `pair_score` reads."""
    return {
        "pair_id": pair_id,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "prediction_a": prediction_a,
        "prediction_b": prediction_b,
    }


def test_a_pair_is_right_only_when_both_sides_are():
    scored = pair_score(pair("5", "-7", "<answer>5</answer>", "<answer>-7</answer>"))
    assert scored["pair_correct"] is True, "both sides answered their own question"
    assert scored["strict_pair_correct"] is True, "both responses carry exactly one answer tag"
    assert scored["collapsed"] is False, "the two answers differ, so nothing collapsed"
    assert scored["ambiguous"] is False, "neither response names both answers"


def test_the_twins_answer_does_not_count():
    scored = pair_score(pair("5", "-7", "<answer>-7</answer>", "<answer>-7</answer>"))
    assert scored["correct_a"] is False, "side a gave the other member's answer"
    assert scored["correct_b"] is True, "side b gave its own answer"
    assert scored["pair_correct"] is False, "one wrong side loses the pair"
    assert (scored["match_tier_a"], scored["other_match_tier_a"]) == (0, 2), (
        "side a should match the other member's answer and not its own, got tiers "
        f"{scored['match_tier_a']} (own) and {scored['other_match_tier_a']} (other)"
    )


def test_the_twins_answer_with_the_sign_flipped_does_not_count():
    scored = pair_score(pair("7", "-7", "<answer>-7</answer>", "<answer>7</answer>"))
    assert (scored["correct_a"], scored["correct_b"]) == (False, False), (
        "each side gave the other member's answer, which differs only in sign, got "
        f"{scored['correct_a']} and {scored['correct_b']}"
    )
    assert scored["ambiguous"] is False, (
        "a sign-aware matcher sees one answer per response, not both"
    )


def test_naming_both_answers_is_ambiguous_rather_than_correct():
    scored = pair_score(pair("5", "-7", "<answer>5 or -7</answer>", "<answer>-7</answer>"))
    assert scored["ambiguous"] is True, "a response naming both answers is ambiguous"
    assert scored["correct_a"] is False, "an ambiguous response must not be scored correct"


def test_the_collapse_flag_marks_one_answer_given_twice():
    collapsed = pair_score(pair("5", "-7", "<answer>5</answer>", "<answer>5</answer>"))
    assert collapsed["collapsed"] is True, "the same answer was given for both members"
    assert collapsed["pair_correct"] is False, "one of the two sides must be wrong"
    presentation = pair_score(pair("5", "-7", "<answer>5</answer>", "<answer>5.</answer>"))
    assert presentation["collapsed"] is True, (
        "answers that differ only in trailing punctuation are the same answer"
    )
    shared_gold = pair_score(pair("3", "3", "<answer>3</answer>", "<answer>3</answer>"))
    assert shared_gold["collapsed"] is False, (
        "the two members share one answer, so repeating it is not a collapse"
    )
    assert shared_gold["pair_correct"] is True, (
        "on a pair with one shared answer, giving that answer twice is right"
    )


def test_a_response_without_tags_fails_the_strict_score_only():
    scored = pair_score(pair("5", "-7", "<answer>5</answer>", r"\boxed{-7}"))
    assert scored["pair_correct"] is True, "both answers are right"
    assert scored["strict_pair_correct"] is False, "side b carries no answer tag"
    assert scored["format_valid_a"] is True, "side a should pass the format check"
    assert scored["format_valid_b"] is False, "side b should fail the format check"


FOUR_PAIRS = [
    pair("5", "-7", "<answer>5</answer>", "<answer>-7</answer>", "p1"),
    pair("5", "-7", "<answer>-7</answer>", "<answer>-7</answer>", "p2"),
    pair("5", "-7", "<answer>5</answer>", "<answer>5</answer>", "p3"),
    pair("5", "-7", "<answer>5</answer>", r"\boxed{-7}", "p4"),
]


def test_aggregate_metrics_match_the_hand_count():
    metrics = aggregate_pair_metrics(FOUR_PAIRS)
    # p1 and p4 are right; p2 and p3 each give one answer twice; p4 side b is untagged.
    expected = {
        "n_pairs": 4.0,
        "pair_accuracy": 2 / 4,
        "member_accuracy": 6 / 8,
        "strict_pair_accuracy": 1 / 4,
        "strict_member_accuracy": 5 / 8,
        "collapse_rate": 2 / 4,
        "format_valid_rate": 7 / 8,
    }
    for field, want in expected.items():
        assert metrics[field] == pytest.approx(want), (
            f"{field} came out {metrics[field]}, expected {want}"
        )


def test_aggregate_metrics_of_no_pairs_are_undefined_not_zero():
    metrics = aggregate_pair_metrics([])
    assert metrics["n_pairs"] == 0, f"n_pairs was {metrics['n_pairs']}"
    assert math.isnan(metrics["pair_accuracy"]), (
        f"pair accuracy over no pairs should be undefined, got {metrics['pair_accuracy']}"
    )


def test_bootstrap_interval_is_reproducible_and_brackets_the_mean():
    values = [1.0] * 30 + [0.0] * 20
    first = bootstrap_ci(values, n_boot=200, seed=11)
    assert first == bootstrap_ci(values, n_boot=200, seed=11), "the same seed gave two intervals"
    assert first[0] < 0.6 < first[1], f"the interval {first} does not contain the mean 0.6"
    spread = [index / 49 for index in range(50)]
    assert bootstrap_ci(spread, n_boot=200, seed=11) != bootstrap_ci(spread, n_boot=200, seed=12), (
        "two seeds gave the identical interval on fifty distinct values, so the seed is unused"
    )
    empty = bootstrap_ci([], n_boot=10)
    assert all(math.isnan(bound) for bound in empty), (
        f"an interval over no values should be undefined, got {empty}"
    )


def test_the_permutation_null_separates_answers_that_depend_on_the_side():
    side_dependent = [
        pair("5", "-7", "<answer>5</answer>", "<answer>-7</answer>", f"p{index}")
        for index in range(20)
    ]
    result = permutation_null_pair_accuracy(side_dependent, n_perm=200, seed=0)
    assert result["observed"] == 1.0, f"every pair is right, got {result['observed']}"
    assert result["p_ge"] == pytest.approx(1 / 201), (
        f"swapping the two responses should destroy the score, p was {result['p_ge']}"
    )
    assert 0.3 < result["null_mean"] < 0.7, (
        f"about half the pairs should keep their order under the null, got {result['null_mean']}"
    )


def test_the_permutation_null_is_fixed_by_its_seed():
    mixed = [
        pair("5", "-7", "<answer>5</answer>", "<answer>-7</answer>" if index % 2 else "<answer>5</answer>",
             f"p{index}")
        for index in range(10)
    ]
    first = permutation_null_pair_accuracy(mixed, n_perm=100, seed=4)
    again = permutation_null_pair_accuracy(mixed, n_perm=100, seed=4)
    assert first == again, f"the same seed gave two results: {first} and {again}"
    assert first["observed"] == 0.5, f"five of the ten pairs are right, got {first['observed']}"


def test_the_permutation_null_cannot_separate_a_pair_with_one_shared_answer():
    shared_gold = [
        pair("3", "3", "<answer>3</answer>", "<answer>3</answer>", f"q{index}")
        for index in range(5)
    ]
    result = permutation_null_pair_accuracy(shared_gold, n_perm=200, seed=0)
    assert result["observed"] == 1.0, f"every pair is right, got {result['observed']}"
    assert result["p_ge"] == 1.0, (
        f"swapping two identical responses changes nothing, so p should be 1.0, got {result['p_ge']}"
    )


def run_with(correct_ids, n=4):
    """A run over the same `n` pairs, right on the pairs in `correct_ids`."""
    return [
        pair(
            "5",
            "-7",
            "<answer>5</answer>" if index in correct_ids else "<answer>0</answer>",
            "<answer>-7</answer>",
            f"p{index}",
        )
        for index in range(n)
    ]


def test_mcnemar_counts_the_pairs_that_moved():
    result = mcnemar_exact(run_with({0, 1, 2}), run_with({3}))
    assert result["n_common"] == 4, f"the two runs share four ids, got {result['n_common']}"
    assert result["b10"] == 3, f"three pairs were lost, got {result['b10']}"
    assert result["b01"] == 1, f"one pair was gained, got {result['b01']}"
    assert result["p_value"] == pytest.approx(0.625), (
        f"the exact p for three against one discordant pairs is 10/16, got {result['p_value']}"
    )


def test_mcnemar_uses_only_the_shared_ids():
    result = mcnemar_exact(run_with(set(range(10)), n=10), run_with(set(), n=6))
    assert result["n_common"] == 6, f"six ids are shared, got {result['n_common']}"
    assert (result["b10"], result["b01"]) == (6, 0), (
        f"six shared pairs were lost and none gained, got b10={result['b10']} b01={result['b01']}"
    )
    assert result["p_value"] == pytest.approx(2 / 64), (
        f"the exact p for six against zero discordant pairs is 2/64, got {result['p_value']}"
    )


def test_mcnemar_reports_no_difference_when_nothing_moved():
    rows = run_with({0, 1})
    result = mcnemar_exact(rows, rows)
    assert result["p_value"] == 1.0, (
        f"a run compared with itself has no discordant pairs, p was {result['p_value']}"
    )

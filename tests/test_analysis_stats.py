"""The analysis statistics on small inputs with hand-computed answers."""
from __future__ import annotations

import math

import pytest

from lwl.analysis import stats
from lwl.evaluation.metrics import mcnemar_exact, pair_score


def pair_rows(flags, prefix="p", seed=1):
    """Released-shape pair rows whose recorded correctness follows `flags`."""
    return [
        {
            "pair_id": f"{prefix}{index}",
            "gold_a": "1",
            "gold_b": "2",
            "prediction_a": "1" if correct else "2",
            "prediction_b": "2" if correct else "1",
            "pair_correct": bool(correct),
            "seed": seed,
        }
        for index, correct in enumerate(flags)
    ]


def test_pair_and_item_accuracy_match_the_hand_count():
    pairs = stats.pair_accuracy(pair_rows([1, 1, 0, 0, 1]))
    assert pairs == pytest.approx(3 / 5), f"three of five pairs are right, got {pairs}"
    items = [{"correct": True}, {"correct": False}, {"correct": True}, {"correct": True}]
    item_accuracy = stats.item_accuracy(items)
    assert item_accuracy == pytest.approx(3 / 4), f"three of four items are right, got {item_accuracy}"


def test_pair_accuracy_agrees_with_rescoring_the_released_rows():
    rows = pair_rows([1, 0, 1, 1, 0, 0, 1])
    recorded = stats.pair_accuracy(rows)
    rescored = sum(pair_score(row)["pair_correct"] for row in stats.to_scorable(rows)) / len(rows)
    assert recorded == pytest.approx(rescored), (
        f"the recorded flags give {recorded}, rescoring the predictions gives {rescored}"
    )


def test_an_unscored_row_is_an_error_not_a_wrong_answer():
    rows = pair_rows([1, 1])
    rows[1]["pair_correct"] = None
    with pytest.raises(ValueError, match="no pair_correct"):
        stats.pair_accuracy(rows)
    with pytest.raises(ValueError, match="at least one value"):
        stats.accuracy([])


def test_seed_means_and_mean_over_seeds():
    rows = (
        pair_rows([1, 1, 1, 1], prefix="s1-", seed=1)
        + pair_rows([1, 1, 0, 0], prefix="s2-", seed=2)
        + pair_rows([0, 0, 0, 0], prefix="s3-", seed=3)
    )
    per_seed = stats.seed_means(rows)
    assert per_seed == {1: 1.0, 2: 0.5, 3: 0.0}, f"per-seed accuracies came out {per_seed}"
    summary = stats.mean_over_seeds(rows)
    assert summary["estimate"] == pytest.approx(0.5), f"the mean over seeds is 0.5, got {summary['estimate']}"
    assert summary["n_seeds"] == 3, f"three seeds were given, got {summary['n_seeds']}"
    assert summary["sd"] == pytest.approx(0.5), f"the sample sd of 1, 0.5, 0 is 0.5, got {summary['sd']}"
    single = stats.mean_over_seeds(pair_rows([1, 0], seed=7))
    assert single["sd"] == 0.0, f"one seed has no spread, got {single['sd']}"


def test_the_paired_difference_is_right_minus_left_over_shared_ids():
    before = pair_rows([1, 1, 0, 0, 0])
    after = pair_rows([1, 0, 1, 1])
    summary = stats.difference_with_ci(before, after, draws=200, seed=1)
    assert summary["n_shared"] == 4, f"four ids are shared, got {summary['n_shared']}"
    assert summary["estimate"] == pytest.approx(1 / 4), (
        f"over the shared ids one pair was lost and two gained, so +1/4, got {summary['estimate']}"
    )
    unpaired = stats.difference(before, after)
    assert unpaired == pytest.approx(3 / 4 - 2 / 5), f"the unpaired difference is 3/4 - 2/5, got {unpaired}"


def test_the_paired_difference_is_exact_when_every_item_moves():
    summary = stats.difference_with_ci(pair_rows([0] * 4), pair_rows([1] * 4), draws=200, seed=1)
    assert summary["estimate"] == 1.0, f"every item moved from wrong to right, got {summary['estimate']}"
    assert summary["ci95"] == [1.0, 1.0], f"no resample can differ, got {summary['ci95']}"


def test_bootstrap_intervals_are_reproducible_for_a_fixed_seed():
    left = pair_rows([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    right = pair_rows([1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
    first = stats.difference_with_ci(left, right, draws=300, seed=9)
    again = stats.difference_with_ci(left, right, draws=300, seed=9)
    assert first == again, f"the same seed gave {first['ci95']} and {again['ci95']}"
    single = stats.accuracy_with_ci(right, "pair_correct", draws=300, seed=9)
    assert single == stats.accuracy_with_ci(right, "pair_correct", draws=300, seed=9), (
        "the same seed gave two accuracy intervals"
    )
    low, high = single["ci95"]
    assert low <= single["estimate"] <= high, f"the interval {single['ci95']} misses {single['estimate']}"


def test_accuracy_with_ci_uses_the_exact_interval_for_a_constant_cell():
    zero = stats.accuracy_with_ci(pair_rows([0] * 30), "pair_correct", draws=200, seed=1)
    assert (zero["estimate"], zero["ci_method"]) == (0.0, "clopper_pearson"), (
        f"an all-wrong cell should get the exact interval, got {zero['ci_method']} at {zero['estimate']}"
    )
    assert zero["ci95"] == stats.clopper_pearson_interval(0, 30), (
        f"the interval should be the exact one for 0 of 30, got {zero['ci95']}"
    )
    mixed = stats.accuracy_with_ci(pair_rows([1, 0, 1, 1]), "pair_correct", draws=200, seed=1)
    assert (mixed["estimate"], mixed["ci_method"]) == (0.75, "percentile_bootstrap"), (
        f"a mixed cell should get the bootstrap, got {mixed['ci_method']} at {mixed['estimate']}"
    )


@pytest.mark.parametrize(
    "successes, n, low, high",
    [(5, 10, 0.2366, 0.7634), (0, 30, 0.0, 0.1135), (30, 30, 0.8865, 1.0)],
)
def test_wilson_interval_known_values(successes, n, low, high):
    interval = stats.wilson_interval(successes, n)
    assert interval == [pytest.approx(low, abs=5e-4), pytest.approx(high, abs=5e-4)], (
        f"the Wilson interval for {successes}/{n} is [{low}, {high}], got {interval}"
    )


@pytest.mark.parametrize(
    "successes, n, low, high",
    [(0, 100, 0.0, 0.0362), (5, 10, 0.1871, 0.8129), (10, 10, 0.6915, 1.0)],
)
def test_clopper_pearson_known_values(successes, n, low, high):
    interval = stats.clopper_pearson_interval(successes, n)
    assert interval == [pytest.approx(low, abs=5e-4), pytest.approx(high, abs=5e-4)], (
        f"the exact interval for {successes}/{n} is [{low}, {high}], got {interval}"
    )


def test_binomial_intervals_reject_impossible_counts():
    with pytest.raises(ValueError, match="positive"):
        stats.wilson_interval(0, 0)
    with pytest.raises(ValueError, match="lie in"):
        stats.clopper_pearson_interval(11, 10)


def test_mcnemar_on_flags_matches_mcnemar_on_pair_rows():
    left = pair_rows([1] * 10 + [0] * 10)
    right = pair_rows([0, 0] + [1] * 8 + [1] * 6 + [0] * 4)
    from_flags = stats.mcnemar_rows(left, right)
    from_rows = mcnemar_exact(stats.to_scorable(left), stats.to_scorable(right))
    assert (from_flags["b01"], from_flags["b10"]) == (6.0, 2.0), (
        f"six pairs were gained and two lost, got b01={from_flags['b01']} b10={from_flags['b10']}"
    )
    assert (from_rows["b01"], from_rows["b10"]) == (6.0, 2.0), (
        f"rescoring should find the same moves, got b01={from_rows['b01']} b10={from_rows['b10']}"
    )
    assert from_flags["p_value"] == pytest.approx(74 / 256), (
        f"the exact p for (6, 2) is 2 * (1 + 8 + 28) / 256, got {from_flags['p_value']}"
    )
    assert from_rows["p_value"] == pytest.approx(from_flags["p_value"]), (
        f"the two McNemar versions disagree: {from_rows['p_value']} and {from_flags['p_value']}"
    )


def test_mcnemar_flags_at_the_extremes():
    all_lost = stats.mcnemar_flags([1] * 10, [0] * 10)["p_value"]
    assert all_lost == pytest.approx(2 / 1024), f"ten of ten discordant one way gives 2/1024, got {all_lost}"
    unchanged = stats.mcnemar_flags([1, 0], [1, 0])["p_value"]
    assert unchanged == 1.0, f"no discordant pair gives p = 1, got {unchanged}"
    with pytest.raises(ValueError, match="differ in length"):
        stats.mcnemar_flags([1, 0], [1])


def test_to_scorable_puts_the_golds_where_the_scorer_reads_them():
    row = {"pair_id": "x", "gold_a": "5", "gold_b": "7", "prediction_a": "5", "prediction_b": "7"}
    assert pair_score(stats.to_scorable([row])[0])["pair_correct"] is True, (
        "each side gave its own gold and should be scored right"
    )
    swapped = {**row, "prediction_a": "7", "prediction_b": "5"}
    assert pair_score(stats.to_scorable([swapped])[0])["pair_correct"] is False, (
        "each side gave the other side's gold and should be scored wrong"
    )


def test_failure_sets_and_their_overlap():
    rows = pair_rows([1, 0, 1, 0])
    assert stats.correct_ids(rows) == {"p0", "p2"}, f"right ids came out {stats.correct_ids(rows)}"
    assert stats.failed_ids(rows) == {"p1", "p3"}, f"failed ids came out {stats.failed_ids(rows)}"
    cases = [
        (({1, 2, 3}, {2, 3, 4}), 2 / 4),
        (({1, 2}, {1, 2}, {1, 2}), 1.0),
        (({1}, {2}), 0.0),
        ((set(), set()), 0.0),
    ]
    for sets, expected in cases:
        overlap = stats.jaccard_overlap(*sets)
        assert overlap == pytest.approx(expected), f"the overlap of {sets} is {expected}, got {overlap}"
    with pytest.raises(ValueError, match="at least two"):
        stats.jaccard_overlap({1})


def test_the_overlap_null_separates_a_shared_failure_set():
    universe = list(range(100))
    shared = [set(range(10)), set(range(10))]
    result = stats.jaccard_permutation_null(shared, universe, draws=200, seed=0)
    assert result["jaccard"] == 1.0, f"two identical sets overlap fully, got {result['jaccard']}"
    assert result["null_mean"] < 0.1, (
        f"random sets of ten in a hundred barely overlap, got {result['null_mean']}"
    )
    assert result["p_value"] == pytest.approx(1 / 201), (
        f"no random draw should reach full overlap, p was {result['p_value']}"
    )
    assert result == stats.jaccard_permutation_null(shared, universe, draws=200, seed=0), (
        "the same seed gave two results"
    )


def test_chance_level_is_the_mean_of_one_over_the_options():
    assert stats.chance_level([4, 4, 2, None]) == pytest.approx((0.25 + 0.25 + 0.5 + 0.0) / 4), (
        "chance should average 1/k over the items, with a free-form item at zero"
    )
    assert stats.chance_level([None, 0]) == 0.0, "free-form items have no chance level"
    with pytest.raises(ValueError, match="at least one item"):
        stats.chance_level([])


@pytest.mark.parametrize(
    "removed, present, chance, expected, meaning",
    [
        (0.6, 0.6, 0.25, 1.0, "an unchanged accuracy retains everything above chance"),
        (0.25, 0.6, 0.25, 0.0, "an accuracy at chance retains nothing"),
        (0.1, 0.6, 0.25, -0.15 / 0.35, "an accuracy below chance retains less than nothing"),
        (0.4, 0.6, 0.25, 0.15 / 0.35, "an accuracy between the two retains a share"),
        (0.4, 0.6, 0.0, 2 / 3, "with no chance level the correction is the plain ratio"),
    ],
)
def test_corrected_retention_values(removed, present, chance, expected, meaning):
    value = stats.chance_corrected_retention(removed, present, chance)
    assert value == pytest.approx(expected), f"{meaning}: expected {expected}, got {value}"


def test_corrected_retention_is_undefined_when_the_image_does_not_beat_chance():
    with pytest.raises(ValueError, match="chance level"):
        stats.chance_corrected_retention(0.2, 0.25, 0.25)


def test_retention_summary_reports_both_ratios_reproducibly():
    with_image = [1.0] * 6 + [0.0] * 4
    image_removed = [1.0] * 3 + [0.0] * 7
    summary = stats.retention_summary(with_image, image_removed, 0.25, draws=200, seed=3)
    assert (summary["with_image"], summary["image_removed"]) == (pytest.approx(0.6), pytest.approx(0.3)), (
        f"the two accuracies are 0.6 and 0.3, got {summary['with_image']} and {summary['image_removed']}"
    )
    assert summary["naive_retention"] == pytest.approx(0.5), (
        f"0.3 / 0.6 is 0.5, got {summary['naive_retention']}"
    )
    assert summary["corrected_retention"] == pytest.approx(0.05 / 0.35), (
        f"(0.3 - 0.25) / (0.6 - 0.25) is 1/7, got {summary['corrected_retention']}"
    )
    low, high = summary["corrected_ci95"]
    assert low <= summary["corrected_retention"] <= high, (
        f"the interval {summary['corrected_ci95']} misses the estimate"
    )
    assert summary == stats.retention_summary(with_image, image_removed, 0.25, draws=200, seed=3), (
        "the same seed gave two summaries"
    )


def test_retention_summary_drops_the_draws_where_the_image_sits_at_chance():
    with_image = [1.0, 0.0, 0.0, 0.0]
    image_removed = [1.0, 0.0, 0.0, 0.0]
    summary = stats.retention_summary(with_image, image_removed, 0.0, draws=400, seed=0)
    assert summary["corrected_retention"] == pytest.approx(1.0), (
        f"identical outcomes retain everything, got {summary['corrected_retention']}"
    )
    assert 0 < summary["retained_bootstrap_draws"] < 400, (
        "resamples without the one right item have accuracy at chance and must be dropped, "
        f"kept {summary['retained_bootstrap_draws']} of 400"
    )
    with pytest.raises(ValueError, match="aligned"):
        stats.retention_summary([1.0, 0.0], [1.0], 0.0, draws=10, seed=0)


def test_accuracy_of_booleans_is_a_share_not_a_count():
    value = stats.accuracy([True, False, True])
    assert math.isclose(value, 2 / 3), f"two of three is 2/3, got {value}"

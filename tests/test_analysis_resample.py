"""Resampling helpers: seeds, percentile intervals, paired summaries and rank correlation."""
from __future__ import annotations

import pytest

from lwl.analysis import resample


def test_a_label_and_a_seed_fix_the_stream():
    first = resample.deterministic_seed(7, "pair accuracy")
    assert first == resample.deterministic_seed(7, "pair accuracy"), (
        "the same seed and label gave two different streams"
    )
    assert first != resample.deterministic_seed(7, "item accuracy"), (
        "two labels shared one stream"
    )
    assert first != resample.deterministic_seed(8, "pair accuracy"), (
        "two seeds shared one stream"
    )
    assert 0 <= first < 2**32, f"the seed {first} is out of range"


def test_the_percentile_interval_cuts_where_it_says():
    low, high = resample.percentile_interval(list(range(101)))
    assert low == pytest.approx(2.5), f"the lower cut should be 2.5, got {low}"
    assert high == pytest.approx(97.5), f"the upper cut should be 97.5, got {high}"
    low, high = resample.percentile_interval(list(range(101)), alpha=0.5)
    assert (low, high) == (pytest.approx(25.0), pytest.approx(75.0)), (
        f"a 50 percent interval should run from 25 to 75, got {low} and {high}"
    )
    with pytest.raises(ValueError, match="nonempty and finite"):
        resample.percentile_interval([])


def test_the_bootstrap_is_reproducible_and_centred_on_the_mean():
    values = [1.0] * 30 + [0.0] * 20
    first = resample.mean_with_paired_bootstrap(values, draws=500, seed=3)
    second = resample.mean_with_paired_bootstrap(values, draws=500, seed=3)
    assert first == second, "the same seed gave two different summaries"
    assert first["estimate"] == pytest.approx(0.6), (
        f"the mean of the values is 0.6, the summary says {first['estimate']}"
    )
    assert first["n"] == 50, f"the summary should count 50 values, it counts {first['n']}"
    assert first["ci95"][0] < 0.6 < first["ci95"][1], (
        f"the interval {first['ci95']} does not contain the mean"
    )
    other = resample.mean_with_paired_bootstrap(values, draws=500, seed=4)
    assert other["estimate"] == first["estimate"], "the seed moved the estimate itself"
    spread = [index / 49 for index in range(50)]
    assert (
        resample.mean_with_paired_bootstrap(spread, draws=500, seed=3)["ci95"]
        != resample.mean_with_paired_bootstrap(spread, draws=500, seed=4)["ci95"]
    ), "two seeds gave the identical interval on fifty distinct values, so the seed is unused"


def test_the_paired_se_is_the_standard_error_of_the_mean():
    summary = resample.mean_with_paired_bootstrap([0.0, 1.0, 2.0, 3.0], draws=100, seed=0)
    expected = (5 / 3) ** 0.5 / 2
    assert summary["paired_se"] == pytest.approx(expected), (
        f"the sample sd of 0..3 is sqrt(5/3), over sqrt(4) gives {expected}, got {summary['paired_se']}"
    )


def test_a_constant_vector_has_an_interval_of_zero_width():
    summary = resample.mean_with_paired_bootstrap([1.0] * 12, draws=200, seed=0)
    assert summary["ci95"] == [1.0, 1.0], f"the interval should be a point, got {summary['ci95']}"
    assert summary["paired_se"] == 0.0, f"the standard error should be zero, got {summary['paired_se']}"


def test_the_bootstrap_rejects_what_it_cannot_resample():
    with pytest.raises(ValueError, match="nonempty finite vector"):
        resample.mean_with_paired_bootstrap([], draws=200, seed=0)
    with pytest.raises(ValueError, match="nonempty finite vector"):
        resample.mean_with_paired_bootstrap([1.0, float("nan")], draws=200, seed=0)
    with pytest.raises(ValueError, match="at least 100 draws"):
        resample.mean_with_paired_bootstrap([1.0, 0.0], draws=99, seed=0)


def test_the_paired_difference_is_the_mean_of_the_changes():
    before = [True, True, False, False]
    after = [True, False, True, True]
    summary = resample.paired_difference(before, after, draws=200, seed=1)
    assert summary["estimate"] == pytest.approx(0.25), (
        f"one item was lost and two were gained over four, got {summary['estimate']}"
    )
    assert summary["n"] == 4, f"the summary should count four items, it counts {summary['n']}"
    with pytest.raises(ValueError, match="differ in length"):
        resample.paired_difference([True], [True, False], draws=200, seed=1)


def test_the_paired_ratio_drops_the_draws_it_cannot_divide():
    numerator = [1.0, 0.0, 1.0, 0.0]
    denominator = [1.0, 1.0, 0.0, 0.0]
    summary = resample.paired_ratio(numerator, denominator, draws=200, seed=2)
    assert summary["estimate"] == pytest.approx(1.0), (
        f"both means are 0.5, so the ratio is 1.0, got {summary['estimate']}"
    )
    assert summary["denominator_estimate"] == pytest.approx(0.5), (
        f"the denominator mean should be 0.5, got {summary['denominator_estimate']}"
    )
    assert 0 < summary["retained_bootstrap_draws"] <= 200, (
        f"the retained draws should be a subset of the 200 taken, got "
        f"{summary['retained_bootstrap_draws']}"
    )


def test_a_vanishing_denominator_leaves_no_ratio():
    summary = resample.paired_ratio([1.0, 1.0], [0.0, 0.0], draws=200, seed=0)
    assert summary["estimate"] is None, (
        f"a zero denominator should leave no estimate, got {summary['estimate']}"
    )
    assert summary["ci95"] is None, f"a zero denominator should leave no interval, got {summary['ci95']}"


def test_rank_correlation_handles_ties_and_constants():
    assert resample.tied_spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0), (
        "two vectors in the same order should correlate at 1.0"
    )
    assert resample.tied_spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0), (
        "two vectors in opposite order should correlate at -1.0"
    )
    assert resample.tied_spearman([1, 2, 3, 4], [5, 5, 5, 5]) is None, (
        "a constant vector has no rank correlation"
    )
    assert resample.tied_spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0), (
        "tied ranks should still correlate at 1.0"
    )


def test_the_floor_contrast_compares_the_two_groups():
    counts = [0, 0, 0, 0, 1, 2, 3, 4]
    gains = [0.0, 0.0, 0.2, 0.2, 0.5, 0.5, 0.7, 0.7]
    summary = resample.hurdle_summary(counts, gains, draws=200, seed=5)
    assert summary["floor_mean_gain"] == pytest.approx(0.1), (
        f"the four floor rows average 0.1, got {summary['floor_mean_gain']}"
    )
    assert summary["above_floor_mean_gain"] == pytest.approx(0.6), (
        f"the four remaining rows average 0.6, got {summary['above_floor_mean_gain']}"
    )
    assert summary["estimate"] == pytest.approx(0.5), (
        f"the contrast should be 0.5, got {summary['estimate']}"
    )
    assert summary["floor_n"] == 4 and summary["above_floor_n"] == 4, (
        f"the groups should hold four rows each, got {summary['floor_n']} and "
        f"{summary['above_floor_n']}"
    )
    with pytest.raises(ValueError, match="both floor and above-floor"):
        resample.hurdle_summary([1, 2], [0.5, 0.5], draws=200, seed=5)


def test_the_decile_table_has_a_floor_and_ten_groups():
    rows = [{"sample_correct_count": 0, "q_i": 0.0, "gain": 0.1, "row_index": 0}]
    rows += [
        {"sample_correct_count": 1, "q_i": index / 20, "gain": index / 100, "row_index": index}
        for index in range(1, 21)
    ]
    table = resample.floor_and_tail_deciles(rows)
    assert len(table) == 11, f"a floor and ten deciles make eleven groups, got {len(table)}"
    assert table[0]["group"] == "floor_c0", f"the first group should be the floor, got {table[0]['group']}"
    assert [group["n"] for group in table[1:]] == [2] * 10, (
        f"twenty rows should split into ten groups of two, got {[g['n'] for g in table[1:]]}"
    )
    assert table[1]["q_max"] <= table[2]["q_min"], "the deciles are not ordered by q"

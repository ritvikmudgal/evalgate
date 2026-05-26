from __future__ import annotations

import math

import pytest

from evalgate.stats import (
    mcnemar_test,
    normal_cdf,
    two_proportion_test,
    wilson_interval,
)


def test_normal_cdf_known_values():
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.959963985) == pytest.approx(0.975, abs=1e-4)
    assert normal_cdf(-1.959963985) == pytest.approx(0.025, abs=1e-4)


def test_two_proportion_no_difference():
    result = two_proportion_test(900, 1000, 900, 1000)
    assert result.difference == 0.0
    assert result.p_value == pytest.approx(1.0)


def test_two_proportion_large_clear_difference():
    # 0.90 vs 0.80 on 1000 each is highly significant.
    result = two_proportion_test(900, 1000, 800, 1000)
    assert result.difference == pytest.approx(-0.10)
    assert result.p_value < 0.001


def test_two_proportion_small_difference_not_significant():
    # 0.900 vs 0.895 on 1000 each is well within noise.
    result = two_proportion_test(900, 1000, 895, 1000)
    assert result.p_value > 0.05


def test_two_proportion_matches_manual_z():
    result = two_proportion_test(900, 1000, 850, 1000)
    pooled = (900 + 850) / 2000
    se = math.sqrt(pooled * (1 - pooled) * (1 / 1000 + 1 / 1000))
    expected_z = (0.85 - 0.90) / se
    assert result.statistic == pytest.approx(expected_z)


def test_two_proportion_rejects_zero_n():
    with pytest.raises(ValueError):
        two_proportion_test(0, 0, 1, 10)


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(90, 100)
    assert lo < 0.90 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_mcnemar_symmetric_is_not_significant():
    result = mcnemar_test(10, 10)
    assert result.difference == 0.0
    assert result.p_value == pytest.approx(1.0)


def test_mcnemar_one_sided_discordance_is_significant():
    # candidate wrong on 20 the baseline got right, right on only 2 it missed.
    result = mcnemar_test(20, 2)
    assert result.difference == -18.0
    assert result.p_value < 0.001


def test_mcnemar_no_discordant_pairs():
    result = mcnemar_test(0, 0)
    assert result.p_value == 1.0


def test_mcnemar_large_sample_uses_chi2():
    result = mcnemar_test(800, 600)  # n > 1000
    assert result.p_value < 0.001

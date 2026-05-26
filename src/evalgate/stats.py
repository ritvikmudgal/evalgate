"""Significance tests, implemented from the standard formulas.

Kept dependency-free (math only) so the numbers are auditable and the tests can
check them against textbook values. Two tests are provided: a two-proportion
z-test for aggregate accuracies, and McNemar's test for paired per-example
correctness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chi2_1df_sf(x: float) -> float:
    """Survival function (1 - CDF) of a chi-squared distribution with 1 dof."""

    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


@dataclass(frozen=True, slots=True)
class StatResult:
    statistic: float
    p_value: float
    difference: float


def two_proportion_test(
    baseline_successes: int,
    baseline_n: int,
    candidate_successes: int,
    candidate_n: int,
) -> StatResult:
    """Two-sided z-test for the difference between two proportions."""

    if baseline_n <= 0 or candidate_n <= 0:
        raise ValueError("sample sizes must be positive")
    p1 = baseline_successes / baseline_n
    p2 = candidate_successes / candidate_n
    diff = p2 - p1
    pooled = (baseline_successes + candidate_successes) / (baseline_n + candidate_n)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / baseline_n + 1.0 / candidate_n))
    if se == 0.0:
        return StatResult(statistic=0.0, p_value=1.0, difference=diff)
    z = diff / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return StatResult(statistic=z, p_value=p_value, difference=diff)


def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion (default 95%)."""

    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_test(discordant_baseline_only: int, discordant_candidate_only: int) -> StatResult:
    """McNemar's test on the two discordant counts.

    ``discordant_baseline_only`` (b) is the number of examples the baseline got
    right and the candidate got wrong; ``discordant_candidate_only`` (c) is the
    reverse. An exact two-sided binomial test is used for small samples, and the
    continuity-corrected chi-squared approximation for large ones.
    """

    b = discordant_baseline_only
    c = discordant_candidate_only
    n = b + c
    # difference in accuracy contributed by the discordant pairs is (c - b)/n,
    # but here difference is reported on the discordant scale as a signed count.
    diff = float(c - b)
    if n == 0:
        return StatResult(statistic=0.0, p_value=1.0, difference=0.0)
    if n <= 1000:
        p_value = _exact_binomial_two_sided(min(b, c), n)
        statistic = float(min(b, c))
    else:
        statistic = (abs(b - c) - 1.0) ** 2 / n
        p_value = chi2_1df_sf(statistic)
    return StatResult(statistic=statistic, p_value=min(1.0, p_value), difference=diff)


def _exact_binomial_two_sided(k: int, n: int) -> float:
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)

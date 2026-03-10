"""Deflated Sharpe Ratio and Probabilistic Sharpe Ratio.

Implements the framework from Bailey & Lopez de Prado (2014) for
adjusting Sharpe ratios when multiple strategies are tested.

The key insight: if you test N strategies and pick the best Sharpe,
the expected maximum Sharpe under the null hypothesis (no skill) is
much higher than zero. The Deflated Sharpe Ratio adjusts for this
multiple testing bias.

Three functions:
    - expected_max_sharpe: E[max(SR)] given N independent trials
    - probabilistic_sharpe: P(true SR > benchmark | observed data)
    - deflated_sharpe_ratio: full DSR combining both

Reference:
    Bailey, D. H. & Lopez de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting, and Non-Normality."
    Journal of Portfolio Management, 40(5), 94-107.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Euler-Mascheroni constant
_EULER_MASCHERONI = 0.5772156649015329


@dataclass
class DSRResult:
    """Result of Deflated Sharpe Ratio analysis.

    Attributes:
        observed_sharpe: The observed (raw) Sharpe ratio.
        expected_max_sharpe: E[max(SR)] under the null, given n_trials.
        deflated_sharpe: observed_sharpe - expected_max_sharpe.
        p_value: Probability that the true SR exceeds the null benchmark.
            High p_value (e.g., > 0.95) means the strategy is significant.
        is_significant: Whether p_value exceeds (1 - significance_level).
        n_trials: Number of strategies/parameter combinations tested.
    """

    observed_sharpe: float
    expected_max_sharpe: float
    deflated_sharpe: float
    p_value: float
    is_significant: bool
    n_trials: int


def expected_max_sharpe(
    n_trials: int,
    sharpe_std: float = 1.0,
) -> float:
    """Expected maximum Sharpe ratio under the null hypothesis.

    Given N independent trials with no true skill (SR ~ N(0, sigma)),
    the expected value of the maximum observed SR is:

        E[max(SR)] = sigma * [(1 - gamma) * Phi_inv(1 - 1/N)
                              + gamma * Phi_inv(1 - 1/(N*e))]

    where gamma is the Euler-Mascheroni constant (~0.5772) and
    Phi_inv is the standard normal quantile function.

    Args:
        n_trials: Number of independent strategies tested (N).
        sharpe_std: Standard deviation of Sharpe ratios across trials.
            Default 1.0. Lower values assume trials are correlated.

    Returns:
        Expected maximum Sharpe ratio.

    Raises:
        ValueError: If n_trials < 1.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    if n_trials == 1:
        return 0.0

    n = float(n_trials)
    gamma = _EULER_MASCHERONI

    # Quantile arguments
    q1 = 1.0 - 1.0 / n
    q2 = 1.0 - 1.0 / (n * np.e)

    # Clamp to avoid ppf(0) or ppf(1) edge cases
    q1 = np.clip(q1, 1e-10, 1.0 - 1e-10)
    q2 = np.clip(q2, 1e-10, 1.0 - 1e-10)

    z1 = float(stats.norm.ppf(q1))
    z2 = float(stats.norm.ppf(q2))

    return sharpe_std * ((1.0 - gamma) * z1 + gamma * z2)


def probabilistic_sharpe(
    observed_sr: float,
    benchmark_sr: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR).

    Computes the probability that the true Sharpe ratio exceeds
    a given benchmark, accounting for the estimation error in
    the observed Sharpe ratio.

    The standard error of the Sharpe ratio incorporates
    non-normality corrections (skewness and kurtosis):

        SE(SR) = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1))

    where T is the number of observations, skew is the skewness
    of returns, and kurt is the kurtosis (not excess kurtosis;
    normal distribution has kurtosis = 3).

    Args:
        observed_sr: Observed (annualized) Sharpe ratio.
        benchmark_sr: Benchmark Sharpe ratio to test against.
        n_observations: Number of return observations (T).
        skewness: Skewness of returns. Default 0.0 (normal).
        kurtosis: Kurtosis of returns (NOT excess kurtosis).
            Default 3.0 (normal). If using pandas .kurtosis(),
            add 3: kurtosis = returns.kurtosis() + 3.

    Returns:
        Probability in [0, 1] that the true SR exceeds the benchmark.
        Returns 0.5 when insufficient data.
    """
    if n_observations < 2:
        return 0.5

    sr = observed_sr
    t = float(n_observations)

    # Non-normality corrected standard error
    se_squared = (1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (t - 1.0)

    if se_squared <= 0.0:
        return 0.5

    se = np.sqrt(se_squared)
    z = (sr - benchmark_sr) / se

    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_observations: int,
    sharpe_std: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    significance_level: float = 0.05,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio.

    Adjusts the observed Sharpe ratio for the number of trials
    (strategies, parameter combinations) tested. Uses the expected
    maximum Sharpe under the null as the benchmark for the
    Probabilistic Sharpe Ratio.

    A strategy is significant if the probability of the true SR
    exceeding the multiple-testing-adjusted benchmark is above
    (1 - significance_level).

    Args:
        observed_sr: Observed (annualized) Sharpe ratio.
        n_trials: Number of independent strategies tested.
        n_observations: Number of return observations.
        sharpe_std: Standard deviation of Sharpe ratios across trials.
        skewness: Skewness of returns (default 0.0).
        kurtosis: Kurtosis of returns (default 3.0, normal).
        significance_level: Significance threshold (default 0.05).

    Returns:
        DSRResult with deflated metrics and significance flag.

    Raises:
        ValueError: If inputs are invalid.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_observations < 1:
        raise ValueError(f"n_observations must be >= 1, got {n_observations}")
    if not 0.0 < significance_level < 1.0:
        raise ValueError(
            f"significance_level must be in (0, 1), got {significance_level}"
        )

    e_max_sr = expected_max_sharpe(n_trials, sharpe_std)

    p_value = probabilistic_sharpe(
        observed_sr=observed_sr,
        benchmark_sr=e_max_sr,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    deflated_sr = observed_sr - e_max_sr

    return DSRResult(
        observed_sharpe=observed_sr,
        expected_max_sharpe=e_max_sr,
        deflated_sharpe=deflated_sr,
        p_value=p_value,
        is_significant=p_value > (1.0 - significance_level),
        n_trials=n_trials,
    )

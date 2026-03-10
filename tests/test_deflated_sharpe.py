"""Unit tests for Deflated Sharpe Ratio and Probabilistic Sharpe Ratio."""

import numpy as np
import pytest
from scipy import stats

from src.validation.deflated_sharpe import (
    DSRResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe,
)

# ---------------------------------------------------------------------------
# DSRResult dataclass
# ---------------------------------------------------------------------------


class TestDSRResult:
    def test_creation(self) -> None:
        result = DSRResult(
            observed_sharpe=1.5,
            expected_max_sharpe=1.0,
            deflated_sharpe=0.5,
            p_value=0.85,
            is_significant=True,
            n_trials=10,
        )
        assert result.observed_sharpe == 1.5
        assert result.expected_max_sharpe == 1.0
        assert result.deflated_sharpe == 0.5
        assert result.p_value == 0.85
        assert result.is_significant is True
        assert result.n_trials == 10

    def test_is_significant_true(self) -> None:
        result = DSRResult(
            observed_sharpe=2.0,
            expected_max_sharpe=0.5,
            deflated_sharpe=1.5,
            p_value=0.98,
            is_significant=True,
            n_trials=5,
        )
        assert result.is_significant is True

    def test_is_significant_false(self) -> None:
        result = DSRResult(
            observed_sharpe=0.5,
            expected_max_sharpe=2.0,
            deflated_sharpe=-1.5,
            p_value=0.3,
            is_significant=False,
            n_trials=100,
        )
        assert result.is_significant is False


# ---------------------------------------------------------------------------
# expected_max_sharpe
# ---------------------------------------------------------------------------


class TestExpectedMaxSharpe:
    def test_single_trial_returns_zero(self) -> None:
        """With one trial, no selection bias exists."""
        assert expected_max_sharpe(1) == 0.0

    def test_two_trials(self) -> None:
        """With two trials, E[max SR] should be small but positive."""
        result = expected_max_sharpe(2)
        assert result > 0.0
        assert result < 1.0

    def test_increases_with_trials(self) -> None:
        """E[max SR] should increase monotonically with N."""
        values = [expected_max_sharpe(n) for n in [2, 10, 50, 100, 500, 1000]]
        for i in range(1, len(values)):
            assert values[i] > values[i - 1]

    def test_1000_trials_significant_inflation(self) -> None:
        """At N=1000, E[max SR] should be approximately 3.26."""
        result = expected_max_sharpe(1000)
        assert 3.0 < result < 3.5

    def test_sharpe_std_scaling(self) -> None:
        """E[max SR] scales linearly with sharpe_std."""
        base = expected_max_sharpe(100, sharpe_std=1.0)
        double = expected_max_sharpe(100, sharpe_std=2.0)
        assert double == pytest.approx(2.0 * base, rel=1e-10)

    def test_sharpe_std_half(self) -> None:
        """Half sharpe_std gives half E[max SR]."""
        base = expected_max_sharpe(50, sharpe_std=1.0)
        half = expected_max_sharpe(50, sharpe_std=0.5)
        assert half == pytest.approx(0.5 * base, rel=1e-10)

    def test_known_value_n10(self) -> None:
        """For N=10, verify against manual calculation."""
        # Phi_inv(1 - 1/10) = Phi_inv(0.9) ~ 1.2816
        # Phi_inv(1 - 1/(10*e)) = Phi_inv(1 - 0.03679) ~ Phi_inv(0.96321) ~ 1.7927
        gamma = 0.5772156649015329
        z1 = float(stats.norm.ppf(0.9))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (10.0 * np.e)))
        expected = (1.0 - gamma) * z1 + gamma * z2
        result = expected_max_sharpe(10)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_n_trials_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be >= 1"):
            expected_max_sharpe(0)

    def test_n_trials_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be >= 1"):
            expected_max_sharpe(-5)


# ---------------------------------------------------------------------------
# probabilistic_sharpe
# ---------------------------------------------------------------------------


class TestProbabilisticSharpe:
    def test_sr_equals_benchmark_returns_half(self) -> None:
        """When observed == benchmark, PSR = 0.5 by symmetry."""
        result = probabilistic_sharpe(
            observed_sr=1.0,
            benchmark_sr=1.0,
            n_observations=252,
        )
        assert result == pytest.approx(0.5, abs=0.01)

    def test_sr_above_benchmark(self) -> None:
        """Large positive SR with zero benchmark -> PSR near 1.0."""
        result = probabilistic_sharpe(
            observed_sr=2.0,
            benchmark_sr=0.0,
            n_observations=1000,
        )
        assert result > 0.95

    def test_sr_below_benchmark(self) -> None:
        """SR below benchmark -> PSR near 0.0."""
        result = probabilistic_sharpe(
            observed_sr=0.0,
            benchmark_sr=2.0,
            n_observations=1000,
        )
        assert result < 0.05

    def test_few_observations_returns_half(self) -> None:
        """With only 1 observation, return 0.5 (insufficient data)."""
        result = probabilistic_sharpe(
            observed_sr=5.0,
            benchmark_sr=0.0,
            n_observations=1,
        )
        assert result == 0.5

    def test_normal_returns_symmetry(self) -> None:
        """With normal returns (skew=0, kurt=3), PSR is symmetric."""
        psr_above = probabilistic_sharpe(
            observed_sr=1.0,
            benchmark_sr=0.0,
            n_observations=252,
            skewness=0.0,
            kurtosis=3.0,
        )
        psr_below = probabilistic_sharpe(
            observed_sr=-1.0,
            benchmark_sr=0.0,
            n_observations=252,
            skewness=0.0,
            kurtosis=3.0,
        )
        # PSR(SR=1, bench=0) + PSR(SR=-1, bench=0) ~ 1.0
        assert (psr_above + psr_below) == pytest.approx(1.0, abs=0.05)

    def test_negative_skewness_decreases_psr(self) -> None:
        """Negative skewness increases SE, lowering PSR for positive SR."""
        # Use smaller SR and fewer observations so PSR doesn't saturate to 1.0
        psr_normal = probabilistic_sharpe(
            observed_sr=0.3,
            benchmark_sr=0.0,
            n_observations=50,
            skewness=0.0,
            kurtosis=3.0,
        )
        psr_neg_skew = probabilistic_sharpe(
            observed_sr=0.3,
            benchmark_sr=0.0,
            n_observations=50,
            skewness=-1.0,
            kurtosis=3.0,
        )
        assert psr_neg_skew < psr_normal

    def test_high_kurtosis_decreases_psr(self) -> None:
        """Excess kurtosis increases SE, lowering PSR for positive SR."""
        # Use smaller SR and fewer observations so PSR doesn't saturate to 1.0
        psr_normal = probabilistic_sharpe(
            observed_sr=0.3,
            benchmark_sr=0.0,
            n_observations=50,
            skewness=0.0,
            kurtosis=3.0,
        )
        psr_fat_tails = probabilistic_sharpe(
            observed_sr=0.3,
            benchmark_sr=0.0,
            n_observations=50,
            skewness=0.0,
            kurtosis=6.0,
        )
        assert psr_fat_tails < psr_normal

    def test_large_n_narrows_se(self) -> None:
        """With many observations, PSR converges to 0 or 1."""
        psr = probabilistic_sharpe(
            observed_sr=0.5,
            benchmark_sr=0.0,
            n_observations=10000,
        )
        assert psr > 0.99

    def test_returns_between_0_and_1(self) -> None:
        """PSR is always a probability in [0, 1]."""
        for sr in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0]:
            result = probabilistic_sharpe(
                observed_sr=sr,
                benchmark_sr=0.0,
                n_observations=252,
            )
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# deflated_sharpe_ratio
# ---------------------------------------------------------------------------


class TestDeflatedSharpeRatio:
    def test_single_trial_no_deflation(self) -> None:
        """With one trial, E[max SR] = 0, so no deflation."""
        result = deflated_sharpe_ratio(
            observed_sr=1.5,
            n_trials=1,
            n_observations=252,
        )
        assert result.expected_max_sharpe == 0.0
        assert result.deflated_sharpe == pytest.approx(1.5)
        assert result.n_trials == 1

    def test_many_trials_significant_deflation(self) -> None:
        """With 1000 trials, a modest SR of 1.5 is not significant."""
        result = deflated_sharpe_ratio(
            observed_sr=1.5,
            n_trials=1000,
            n_observations=252,
        )
        assert result.expected_max_sharpe > 3.0
        assert result.deflated_sharpe < 0.0
        assert result.is_significant is False

    def test_strong_sharpe_still_significant(self) -> None:
        """Very strong SR can survive even many trials."""
        result = deflated_sharpe_ratio(
            observed_sr=5.0,
            n_trials=100,
            n_observations=1000,
        )
        assert result.is_significant is True

    def test_is_significant_flag_matches_threshold(self) -> None:
        """is_significant = p_value > (1 - significance_level)."""
        result = deflated_sharpe_ratio(
            observed_sr=2.0,
            n_trials=10,
            n_observations=252,
            significance_level=0.05,
        )
        assert result.is_significant == (result.p_value > 0.95)

    def test_custom_significance_level(self) -> None:
        """Stricter significance level makes it harder to pass."""
        result_005 = deflated_sharpe_ratio(
            observed_sr=2.0,
            n_trials=20,
            n_observations=252,
            significance_level=0.05,
        )
        result_001 = deflated_sharpe_ratio(
            observed_sr=2.0,
            n_trials=20,
            n_observations=252,
            significance_level=0.01,
        )
        # Same p_value, but stricter threshold
        assert result_005.p_value == pytest.approx(result_001.p_value)
        # If significant at 0.05, might not be at 0.01
        if result_005.is_significant:
            # 0.01 threshold is stricter, so if not significant at 0.01,
            # that's expected
            assert result_001.is_significant == (result_001.p_value > 0.99)

    def test_result_fields_populated(self) -> None:
        result = deflated_sharpe_ratio(
            observed_sr=1.0,
            n_trials=10,
            n_observations=252,
        )
        assert isinstance(result.observed_sharpe, float)
        assert isinstance(result.expected_max_sharpe, float)
        assert isinstance(result.deflated_sharpe, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.is_significant, bool)
        assert isinstance(result.n_trials, int)

    def test_n_trials_stored(self) -> None:
        result = deflated_sharpe_ratio(
            observed_sr=1.0,
            n_trials=42,
            n_observations=252,
        )
        assert result.n_trials == 42

    def test_zero_sharpe_with_many_trials(self) -> None:
        """Zero SR with many trials is never significant."""
        result = deflated_sharpe_ratio(
            observed_sr=0.0,
            n_trials=100,
            n_observations=252,
        )
        assert result.is_significant is False
        assert result.deflated_sharpe < 0.0

    def test_deflated_sharpe_equals_difference(self) -> None:
        """deflated_sharpe = observed_sharpe - expected_max_sharpe."""
        result = deflated_sharpe_ratio(
            observed_sr=2.5,
            n_trials=50,
            n_observations=500,
        )
        assert result.deflated_sharpe == pytest.approx(
            result.observed_sharpe - result.expected_max_sharpe
        )

    def test_skewness_affects_significance(self) -> None:
        """Negative skewness makes it harder to be significant."""
        result_normal = deflated_sharpe_ratio(
            observed_sr=2.0,
            n_trials=10,
            n_observations=252,
            skewness=0.0,
            kurtosis=3.0,
        )
        result_skewed = deflated_sharpe_ratio(
            observed_sr=2.0,
            n_trials=10,
            n_observations=252,
            skewness=-1.5,
            kurtosis=3.0,
        )
        assert result_skewed.p_value < result_normal.p_value

    def test_kurtosis_affects_significance(self) -> None:
        """Fat tails make it harder to be significant."""
        result_normal = deflated_sharpe_ratio(
            observed_sr=2.0,
            n_trials=10,
            n_observations=252,
            skewness=0.0,
            kurtosis=3.0,
        )
        result_fat = deflated_sharpe_ratio(
            observed_sr=2.0,
            n_trials=10,
            n_observations=252,
            skewness=0.0,
            kurtosis=8.0,
        )
        assert result_fat.p_value < result_normal.p_value


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_n_trials_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be >= 1"):
            deflated_sharpe_ratio(observed_sr=1.0, n_trials=0, n_observations=252)

    def test_n_trials_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be >= 1"):
            deflated_sharpe_ratio(observed_sr=1.0, n_trials=-1, n_observations=252)

    def test_n_observations_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_observations must be >= 1"):
            deflated_sharpe_ratio(observed_sr=1.0, n_trials=10, n_observations=0)

    def test_significance_level_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="significance_level must be in"):
            deflated_sharpe_ratio(
                observed_sr=1.0,
                n_trials=10,
                n_observations=252,
                significance_level=0.0,
            )

    def test_significance_level_one_raises(self) -> None:
        with pytest.raises(ValueError, match="significance_level must be in"):
            deflated_sharpe_ratio(
                observed_sr=1.0,
                n_trials=10,
                n_observations=252,
                significance_level=1.0,
            )


# ---------------------------------------------------------------------------
# Integration with PerformanceTracker and CPCV
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_dsr_from_tracker_metrics(self) -> None:
        """Compute DSR from a PerformanceTracker's equity curve."""
        from datetime import datetime, timedelta

        from src.performance.metrics import PerformanceTracker

        tracker = PerformanceTracker(initial_capital=100000.0)

        # Simulate 252 days of equity
        rng = np.random.default_rng(42)
        equity = 100000.0
        base_date = datetime(2020, 1, 1)
        for i in range(252):
            equity *= 1.0 + rng.normal(0.0003, 0.01)
            tracker.update(base_date + timedelta(days=i), equity)

        metrics = tracker.calculate_metrics()
        observed_sr = metrics["sharpe_ratio"]

        # Compute return statistics for DSR
        eq_df = tracker.get_equity_curve()
        returns = eq_df["equity"].pct_change().dropna()
        skew = float(returns.skew())
        kurt = float(returns.kurtosis()) + 3.0  # pandas returns excess kurtosis

        result = deflated_sharpe_ratio(
            observed_sr=observed_sr,
            n_trials=50,
            n_observations=len(returns),
            skewness=skew,
            kurtosis=kurt,
        )

        assert isinstance(result, DSRResult)
        assert result.observed_sharpe == observed_sr
        assert result.n_trials == 50
        assert 0.0 <= result.p_value <= 1.0

    def test_dsr_with_cpcv_n_paths(self) -> None:
        """Use CPCV's n_paths as the n_trials for DSR."""
        from src.validation.cpcv import CPCVValidator

        validator = CPCVValidator(n_splits=6, n_test_splits=2)
        n_paths = validator.n_paths  # 15

        result = deflated_sharpe_ratio(
            observed_sr=1.5,
            n_trials=n_paths,
            n_observations=500,
        )

        assert result.n_trials == 15
        assert result.expected_max_sharpe > 0
        assert isinstance(result.is_significant, bool)

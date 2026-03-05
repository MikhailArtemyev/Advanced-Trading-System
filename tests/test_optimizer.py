"""Unit tests for portfolio optimizers."""

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    AllocationResult,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
)


def _make_returns(n_obs: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate correlated daily returns for AAPL, MSFT, GOOGL."""
    rng = np.random.RandomState(seed)
    # Factor model: common factor + idiosyncratic noise
    factor = rng.randn(n_obs) * 0.01
    returns = pd.DataFrame(
        {
            "AAPL": factor * 1.2 + rng.randn(n_obs) * 0.005 + 0.0003,
            "MSFT": factor * 0.8 + rng.randn(n_obs) * 0.005 + 0.0002,
            "GOOGL": factor * 1.0 + rng.randn(n_obs) * 0.008 + 0.0001,
        }
    )
    return returns


# ──────────────────────────────────────────────────────────────
# AllocationResult
# ──────────────────────────────────────────────────────────────


class TestAllocationResult:
    def test_creation(self):
        result = AllocationResult(
            weights={"AAPL": 0.6, "MSFT": 0.4},
            method="test",
        )
        assert result.weights["AAPL"] == 0.6
        assert result.method == "test"

    def test_defaults(self):
        result = AllocationResult()
        assert result.weights == {}
        assert result.expected_return == 0.0
        assert result.expected_volatility == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.notes == ""

    def test_weights_sum(self):
        result = AllocationResult(weights={"A": 0.5, "B": 0.3, "C": 0.2})
        assert pytest.approx(sum(result.weights.values()), abs=1e-10) == 1.0


# ──────────────────────────────────────────────────────────────
# MeanVarianceOptimizer — init
# ──────────────────────────────────────────────────────────────


class TestMeanVarianceInit:
    def test_default_params(self):
        opt = MeanVarianceOptimizer()
        assert opt.risk_free_rate == 0.02
        assert opt.target == "max_sharpe"
        assert opt.max_weight == 0.30
        assert opt.min_weight == 0.0
        assert opt.shrinkage is True

    def test_custom_params(self):
        opt = MeanVarianceOptimizer(
            risk_free_rate=0.05,
            target="min_variance",
            max_weight=0.50,
            min_weight=0.05,
            shrinkage=False,
        )
        assert opt.risk_free_rate == 0.05
        assert opt.target == "min_variance"
        assert opt.max_weight == 0.50
        assert opt.shrinkage is False


# ──────────────────────────────────────────────────────────────
# MeanVarianceOptimizer — optimize
# ──────────────────────────────────────────────────────────────


class TestMeanVarianceOptimize:
    def test_equal_weight_fallback_insufficient_data(self):
        opt = MeanVarianceOptimizer()
        returns = _make_returns(n_obs=10)  # < 30
        result = opt.optimize(["AAPL", "MSFT"], returns, {})
        assert "equal_weight" in result.method
        assert pytest.approx(result.weights["AAPL"]) == 0.5
        assert pytest.approx(result.weights["MSFT"]) == 0.5

    def test_max_sharpe(self):
        opt = MeanVarianceOptimizer(target="max_sharpe", max_weight=0.80)
        returns = _make_returns(n_obs=200)
        result = opt.optimize(["AAPL", "MSFT", "GOOGL"], returns, {})
        assert "max_sharpe" in result.method
        assert result.expected_return != 0.0
        assert result.expected_volatility > 0.0

    def test_min_variance(self):
        opt = MeanVarianceOptimizer(target="min_variance", max_weight=0.80)
        returns = _make_returns(n_obs=200)
        result = opt.optimize(["AAPL", "MSFT", "GOOGL"], returns, {})
        assert "min_variance" in result.method
        assert result.expected_volatility > 0.0

    def test_weights_sum_to_one(self):
        opt = MeanVarianceOptimizer(max_weight=0.80)
        returns = _make_returns(n_obs=200)
        result = opt.optimize(["AAPL", "MSFT", "GOOGL"], returns, {})
        total = sum(result.weights.values())
        assert pytest.approx(total, abs=1e-6) == 1.0

    def test_weight_bounds_respected(self):
        opt = MeanVarianceOptimizer(max_weight=0.50, min_weight=0.10)
        returns = _make_returns(n_obs=200)
        result = opt.optimize(["AAPL", "MSFT", "GOOGL"], returns, {})
        for w in result.weights.values():
            assert w >= 0.10 - 1e-6
            assert w <= 0.50 + 1e-6

    def test_single_asset(self):
        opt = MeanVarianceOptimizer()
        returns = _make_returns(n_obs=200)
        result = opt.optimize(["AAPL"], returns, {})
        assert result.weights["AAPL"] == pytest.approx(1.0)

    def test_no_return_data(self):
        opt = MeanVarianceOptimizer()
        returns = pd.DataFrame({"XXX": [0.01] * 100})
        result = opt.optimize(["AAPL", "MSFT"], returns, {})
        assert "equal_weight" in result.method


# ──────────────────────────────────────────────────────────────
# MeanVarianceOptimizer — shrinkage
# ──────────────────────────────────────────────────────────────


class TestMeanVarianceShrinkage:
    def test_shrinkage_changes_covariance(self):
        """Ledoit-Wolf shrinkage produces a different covariance matrix."""
        from src.optimization.mean_variance import _ledoit_wolf_shrinkage

        rng = np.random.RandomState(42)
        returns = rng.randn(40, 3) * 0.01
        sample_cov = np.cov(returns, rowvar=False)
        shrunk_cov = _ledoit_wolf_shrinkage(returns, sample_cov)
        # Shrunk covariance should differ from sample
        assert not np.allclose(sample_cov, shrunk_cov)

    def test_shrinkage_produces_valid_result(self):
        returns = _make_returns(n_obs=50)
        opt = MeanVarianceOptimizer(shrinkage=True, max_weight=0.80)
        result = opt.optimize(["AAPL", "MSFT", "GOOGL"], returns, {})
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert result.expected_volatility >= 0.0


# ──────────────────────────────────────────────────────────────
# RiskParityOptimizer — init
# ──────────────────────────────────────────────────────────────


class TestRiskParityInit:
    def test_default_params(self):
        opt = RiskParityOptimizer()
        assert opt.max_weight == 0.40
        assert opt.min_weight == 0.01

    def test_custom_params(self):
        opt = RiskParityOptimizer(max_weight=0.60, min_weight=0.05)
        assert opt.max_weight == 0.60
        assert opt.min_weight == 0.05


# ──────────────────────────────────────────────────────────────
# RiskParityOptimizer — optimize
# ──────────────────────────────────────────────────────────────


class TestRiskParityOptimize:
    def test_equal_weight_fallback(self):
        opt = RiskParityOptimizer()
        returns = _make_returns(n_obs=10)  # < 30
        result = opt.optimize(["AAPL", "MSFT"], returns, {})
        assert "equal_weight" in result.method

    def test_equal_vol_assets_get_equal_weights(self):
        """Assets with identical volatility should get roughly equal weights."""
        rng = np.random.RandomState(123)
        n = 200
        returns = pd.DataFrame(
            {
                "A": rng.randn(n) * 0.01,
                "B": rng.randn(n) * 0.01,
                "C": rng.randn(n) * 0.01,
            }
        )
        opt = RiskParityOptimizer(max_weight=0.80, min_weight=0.01)
        result = opt.optimize(["A", "B", "C"], returns, {})
        weights = list(result.weights.values())
        # All weights should be close to 1/3
        for w in weights:
            assert pytest.approx(w, abs=0.05) == 1.0 / 3.0

    def test_high_vol_asset_gets_lower_weight(self):
        """Asset with higher volatility should get lower weight."""
        rng = np.random.RandomState(42)
        n = 200
        returns = pd.DataFrame(
            {
                "LOW_VOL": rng.randn(n) * 0.005,
                "HIGH_VOL": rng.randn(n) * 0.020,
            }
        )
        opt = RiskParityOptimizer(max_weight=0.90, min_weight=0.01)
        result = opt.optimize(["LOW_VOL", "HIGH_VOL"], returns, {})
        assert result.weights["LOW_VOL"] > result.weights["HIGH_VOL"]

    def test_weights_sum_to_one(self):
        opt = RiskParityOptimizer()
        returns = _make_returns(n_obs=200)
        result = opt.optimize(["AAPL", "MSFT", "GOOGL"], returns, {})
        total = sum(result.weights.values())
        assert pytest.approx(total, abs=1e-6) == 1.0

    def test_weight_bounds_respected(self):
        opt = RiskParityOptimizer(max_weight=0.50, min_weight=0.10)
        returns = _make_returns(n_obs=200)
        result = opt.optimize(["AAPL", "MSFT", "GOOGL"], returns, {})
        for w in result.weights.values():
            assert w >= 0.10 - 1e-6
            assert w <= 0.50 + 1e-6

    def test_single_asset(self):
        opt = RiskParityOptimizer()
        returns = _make_returns(n_obs=200)
        result = opt.optimize(["AAPL"], returns, {})
        assert result.weights["AAPL"] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────
# Optimizer comparison
# ──────────────────────────────────────────────────────────────


class TestOptimizerComparison:
    def test_both_produce_valid_weights(self):
        returns = _make_returns(n_obs=200)
        symbols = ["AAPL", "MSFT", "GOOGL"]

        mv = MeanVarianceOptimizer(max_weight=0.80)
        rp = RiskParityOptimizer(max_weight=0.80, min_weight=0.0)

        r_mv = mv.optimize(symbols, returns, {})
        r_rp = rp.optimize(symbols, returns, {})

        assert pytest.approx(sum(r_mv.weights.values()), abs=1e-6) == 1.0
        assert pytest.approx(sum(r_rp.weights.values()), abs=1e-6) == 1.0

    def test_results_differ(self):
        """Mean-variance and risk-parity produce different allocations."""
        returns = _make_returns(n_obs=200)
        symbols = ["AAPL", "MSFT", "GOOGL"]

        mv = MeanVarianceOptimizer(max_weight=0.80)
        rp = RiskParityOptimizer(max_weight=0.80, min_weight=0.0)

        r_mv = mv.optimize(symbols, returns, {})
        r_rp = rp.optimize(symbols, returns, {})

        # At least one weight should differ
        diffs = [abs(r_mv.weights[s] - r_rp.weights[s]) for s in symbols]
        assert max(diffs) > 0.01

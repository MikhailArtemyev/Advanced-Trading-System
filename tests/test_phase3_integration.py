"""Phase 3 integration tests.

End-to-end tests that verify features, validation, regime detection,
and DSR analysis work together.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import BacktestConfig, load_config
from src.features import (
    ATRFeature,
    BollingerBandFeature,
    FeaturePipeline,
    MACDFeature,
    ReturnFeature,
    RSIFeature,
    SMAFeature,
    VolatilityFeature,
    ZScoreFeature,
)
from src.regime import RegimeDetector
from src.validation import deflated_sharpe_ratio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_ohlcv(n_bars=500, seed=42):
    """Create synthetic OHLCV data with a weak trend signal."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.0005, 0.015, n_bars)
    close = 100.0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="B")

    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n_bars)),
            "high": close * (1 + abs(rng.normal(0, 0.005, n_bars))),
            "low": close * (1 - abs(rng.normal(0, 0.005, n_bars))),
            "close": close,
            "volume": rng.randint(500000, 2000000, n_bars),
        },
        index=dates,
    )


def _build_default_pipeline():
    """Build a standard feature pipeline."""
    return FeaturePipeline(
        [
            SMAFeature(windows=[5, 10, 20]),
            RSIFeature(period=14),
            MACDFeature(),
            BollingerBandFeature(window=20),
            ATRFeature(period=14),
            ReturnFeature(horizons=[1, 5, 10]),
            ZScoreFeature(window=20),
            VolatilityFeature(windows=[5, 20]),
        ]
    )


# ---------------------------------------------------------------------------
# TestFeaturePipeline
# ---------------------------------------------------------------------------


class TestFeaturePipeline:
    """Test feature pipeline produces clean features and targets."""

    def test_direction_target(self):
        ohlcv = _make_synthetic_ohlcv()
        pipeline = _build_default_pipeline()

        result = pipeline.run(ohlcv, target_horizon=5, target_type="direction")
        assert len(result.features) > 100
        assert not result.features.isna().any().any()
        assert set(result.target.unique()).issubset({0, 1})

    def test_return_target(self):
        ohlcv = _make_synthetic_ohlcv()
        pipeline = _build_default_pipeline()

        result = pipeline.run(ohlcv, target_horizon=5, target_type="return")
        assert len(result.features) > 100
        assert not result.features.isna().any().any()


# ---------------------------------------------------------------------------
# TestDSRWithBacktestMetrics
# ---------------------------------------------------------------------------


class TestDSRWithBacktestMetrics:
    """Test DSR analysis on actual backtest results."""

    def test_dsr_on_ml_sharpe(self):
        """DSR should work with Sharpe ratios from a real backtest."""
        # Simulate a moderate Sharpe with some skew
        dsr_result = deflated_sharpe_ratio(
            observed_sr=0.8,
            n_trials=10,
            n_observations=500,
            skewness=-0.3,
            kurtosis=4.0,
        )
        assert dsr_result.expected_max_sharpe > 0
        assert dsr_result.deflated_sharpe < dsr_result.observed_sharpe

    def test_single_trial_preserves_sharpe(self):
        dsr_result = deflated_sharpe_ratio(
            observed_sr=1.5, n_trials=1, n_observations=500
        )
        assert dsr_result.expected_max_sharpe == 0.0
        assert dsr_result.deflated_sharpe == 1.5


# ---------------------------------------------------------------------------
# TestRegimeWithRealData
# ---------------------------------------------------------------------------


class TestRegimeWithRealData:
    """Test regime detection on synthetic data."""

    def test_regime_on_synthetic_ohlcv(self):
        ohlcv = _make_synthetic_ohlcv(n_bars=500)
        detector = RegimeDetector(n_regimes=2, vol_window=20)
        result = detector.fit_predict(ohlcv)

        assert len(result.regime_names) == 2
        assert "bull" in result.regime_names.values()
        assert "bear" in result.regime_names.values()
        assert result.current_regime in range(2)

    def test_regime_proba_on_recent(self):
        ohlcv = _make_synthetic_ohlcv(n_bars=500)
        detector = RegimeDetector(n_regimes=3, vol_window=20)
        detector.fit_predict(ohlcv)

        proba = detector.predict_proba(ohlcv.tail(100))
        assert proba.shape == (3,)
        assert abs(proba.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TestConfigBackwardsCompatibility
# ---------------------------------------------------------------------------


class TestConfigBackwardsCompatibility:
    """Verify Phase 2 configs still load and work with new config fields."""

    def test_phase2_config_loads(self):
        """Phase 2 configs should load without error — new fields use defaults."""
        config = load_config("configs/backtest_config.yaml")
        assert config.strategy.name == "sma_crossover"
        # New fields should have defaults
        assert config.features.target == {"horizon": 5, "type": "direction"}
        assert config.regime.enabled is False
        assert config.tracking.enabled is False

    def test_baseline_config_loads(self):
        config = load_config("configs/backtest_baseline.yaml")
        assert config.sizing.method == "fixed_fraction"

    def test_vol_config_loads(self):
        config = load_config("configs/backtest_phase2_vol.yaml")
        assert config.sizing.method == "volatility"

    def test_kelly_config_loads(self):
        config = load_config("configs/kelly_sizing.yaml")
        assert config.sizing.method == "kelly"


# ---------------------------------------------------------------------------
# TestConfigValidation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_regime_n_regimes_min(self):
        with pytest.raises(ValueError):
            BacktestConfig(
                data={
                    "symbols": ["AAPL"],
                    "start_date": "2020-01-01",
                    "end_date": "2024-01-01",
                },
                execution={},
                strategy={"name": "sma_crossover"},
                regime={"n_regimes": 1},
            )

    def test_validation_n_splits_min(self):
        with pytest.raises(ValueError):
            BacktestConfig(
                data={
                    "symbols": ["AAPL"],
                    "start_date": "2020-01-01",
                    "end_date": "2024-01-01",
                },
                execution={},
                strategy={"name": "sma_crossover"},
                validation={"n_splits": 1},
            )


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full end-to-end: features → regime → DSR."""

    def test_features_regime_dsr(self):
        """Features + regime detection + DSR analysis."""
        ohlcv = _make_synthetic_ohlcv(n_bars=600, seed=42)
        pipeline = _build_default_pipeline()

        # Features + target
        pipeline_result = pipeline.run(ohlcv, target_horizon=5, target_type="direction")
        assert len(pipeline_result.features) > 200

        # Regime detection
        detector = RegimeDetector(n_regimes=2, vol_window=20)
        regime_result = detector.fit_predict(ohlcv)
        assert "bull" in regime_result.regime_names.values()

        # DSR analysis
        dsr_result = deflated_sharpe_ratio(
            observed_sr=0.5,
            n_trials=10,
            n_observations=len(pipeline_result.features),
        )
        assert dsr_result.n_trials == 10
        assert dsr_result.expected_max_sharpe > 0

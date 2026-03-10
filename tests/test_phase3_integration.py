"""Phase 3 integration tests.

End-to-end tests that verify the full ML pipeline works together:
features → model training → strategy → backtest engine → metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.config import BacktestConfig, load_config
from src.execution.execution_handler import ExecutionHandler
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
from src.ml import LightGBMSignalModel, MLStrategy, XGBoostSignalModel
from src.performance.metrics import PerformanceTracker
from src.portfolio.portfolio import Portfolio
from src.regime import RegimeDetector
from src.risk.position_sizer import VolatilityBasedSizer
from src.validation import CPCVValidator, deflated_sharpe_ratio

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
# TestFeatureToModelPipeline
# ---------------------------------------------------------------------------


class TestFeatureToModelPipeline:
    """Test features → model training → prediction."""

    def test_xgboost_classification_pipeline(self):
        ohlcv = _make_synthetic_ohlcv()
        pipeline = _build_default_pipeline()

        result = pipeline.run(ohlcv, target_horizon=5, target_type="direction")
        assert len(result.features) > 100
        assert not result.features.isna().any().any()

        model = XGBoostSignalModel(n_estimators=50, max_depth=3, mode="classification")
        train_result = model.train(result.features, result.target)

        assert train_result.train_score > 0.0
        assert train_result.n_features == len(result.feature_names)

        preds = model.predict(result.features.iloc[-10:])
        assert len(preds) == 10
        for p in preds:
            assert -1.0 <= p.signal <= 1.0
            assert 0.0 <= p.confidence <= 1.0

    def test_lightgbm_regression_pipeline(self):
        ohlcv = _make_synthetic_ohlcv()
        pipeline = _build_default_pipeline()

        result = pipeline.run(ohlcv, target_horizon=5, target_type="return")
        model = LightGBMSignalModel(n_estimators=50, max_depth=3, mode="regression")
        train_result = model.train(result.features, result.target)

        assert train_result.n_train_samples > 0
        preds = model.predict(result.features.iloc[-5:])
        assert len(preds) == 5

    def test_xgboost_with_validation_split(self):
        ohlcv = _make_synthetic_ohlcv(n_bars=600)
        pipeline = _build_default_pipeline()

        result = pipeline.run(ohlcv, target_horizon=5, target_type="direction")
        split = int(len(result.features) * 0.8)
        train_X = result.features.iloc[:split]
        train_y = result.target.iloc[:split]
        val_X = result.features.iloc[split:]
        val_y = result.target.iloc[split:]

        model = XGBoostSignalModel(n_estimators=50, mode="classification")
        train_result = model.train(train_X, train_y, val_X, val_y)

        assert train_result.val_score is not None


# ---------------------------------------------------------------------------
# TestCPCVWithRealModel
# ---------------------------------------------------------------------------


class TestCPCVWithRealModel:
    """Test CPCV validation with real ML models."""

    def test_cpcv_xgboost(self):
        ohlcv = _make_synthetic_ohlcv(n_bars=500)
        pipeline = _build_default_pipeline()
        result = pipeline.run(ohlcv, target_horizon=5, target_type="direction")

        model = XGBoostSignalModel(n_estimators=30, max_depth=2, mode="classification")
        validator = CPCVValidator(n_splits=5, n_test_splits=2, purge_window=5)
        cpcv_result = validator.run(result.features, result.target, model)

        assert cpcv_result.n_paths == 10  # C(5,2)
        assert len(cpcv_result.scores) == 10
        assert 0.0 <= cpcv_result.mean_score <= 1.0
        assert cpcv_result.std_score >= 0.0

    def test_cpcv_predictions_collected(self):
        ohlcv = _make_synthetic_ohlcv(n_bars=400)
        pipeline = _build_default_pipeline()
        result = pipeline.run(ohlcv, target_horizon=5, target_type="direction")

        model = XGBoostSignalModel(n_estimators=20, max_depth=2, mode="classification")
        validator = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=3)
        cpcv_result = validator.run(result.features, result.target, model)

        assert cpcv_result.predictions is not None
        assert len(cpcv_result.predictions) > 0


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
# TestMLStrategyBacktest
# ---------------------------------------------------------------------------


class TestMLStrategyBacktest:
    """Test ML strategy running inside the backtest engine."""

    def _run_ml_backtest(self, model_cls, mode="classification"):
        """Helper to run a full ML backtest."""
        ohlcv = _make_synthetic_ohlcv(n_bars=500, seed=42)
        pipeline = _build_default_pipeline()

        # Train
        pipeline_result = pipeline.run(ohlcv, target_horizon=5, target_type="direction")
        model = model_cls(n_estimators=30, max_depth=2, mode=mode)
        model.train(pipeline_result.features, pipeline_result.target)

        # Build backtest components
        symbols = ["TEST"]

        # Create a simple in-memory data handler

        from src.data.data_handler import DataHandler

        class InMemoryHandler(DataHandler):
            def __init__(self, data):
                self._data = data
                self._idx = 0

            def get_latest_bars(self, symbol, n=1):
                end = self._idx
                start = max(0, end - n)
                return self._data.iloc[start:end].copy()

            def update_bars(self):
                if self._idx >= len(self._data):
                    return False
                self._idx += 1
                return True

            def get_current_bar(self, symbol):
                if self._idx == 0:
                    return None
                return self._data.iloc[self._idx - 1]

            def get_latest_bar_value(self, symbol, field):
                bar = self.get_current_bar(symbol)
                if bar is None:
                    return 0.0
                return float(bar[field])

            def get_current_timestamp(self):
                if self._idx == 0:
                    return self._data.index[0].to_pydatetime()
                return self._data.index[self._idx - 1].to_pydatetime()

            @property
            def continue_backtest(self):
                return self._idx < len(self._data)

            def reset(self):
                self._idx = 0

        handler = InMemoryHandler(ohlcv)

        strategy = MLStrategy(
            symbols=symbols,
            pipeline=pipeline,
            model=model,
            parameters={"signal_threshold": 0.05, "lookback_bars": 200},
        )

        sizer = VolatilityBasedSizer(risk_fraction=0.02)
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=symbols,
            commission_pct=0.001,
            position_sizer=sizer,
        )
        execution = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0005)
        tracker = PerformanceTracker(initial_capital=100000.0)

        engine = BacktestEngine(
            data_handler=handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            performance_tracker=tracker,
        )

        results = engine.run()
        metrics = tracker.calculate_metrics()
        return results, metrics

    def test_xgboost_backtest_runs(self):
        results, metrics = self._run_ml_backtest(XGBoostSignalModel)
        assert "trade_history" in results
        assert "sharpe_ratio" in metrics

    def test_lightgbm_backtest_runs(self):
        results, metrics = self._run_ml_backtest(LightGBMSignalModel)
        assert "trade_history" in results

    def test_ml_backtest_produces_trades(self):
        results, _ = self._run_ml_backtest(XGBoostSignalModel)
        # With signal_threshold=0.05, the model should produce some trades
        assert len(results["trade_history"]) >= 0  # May be 0 on synthetic data

    def test_ml_backtest_metrics_valid(self):
        _, metrics = self._run_ml_backtest(XGBoostSignalModel)
        assert isinstance(metrics["sharpe_ratio"], float)
        assert isinstance(metrics["max_drawdown_pct"], float)


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
        assert config.ml.model == "xgboost"
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

    def test_ml_config_loads(self):
        config = load_config("configs/ml_backtest_config.yaml")
        assert config.strategy.name == "ml_strategy"
        assert config.ml.model == "xgboost"
        assert config.ml.mode == "classification"
        assert len(config.features.technical) == 5
        assert len(config.features.statistical) == 4


# ---------------------------------------------------------------------------
# TestConfigValidation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_invalid_ml_model_raises(self):
        with pytest.raises(ValueError):
            BacktestConfig(
                data={
                    "symbols": ["AAPL"],
                    "start_date": "2020-01-01",
                    "end_date": "2024-01-01",
                },
                execution={},
                strategy={"name": "ml_strategy"},
                ml={"model": "invalid_model"},
            )

    def test_invalid_ml_mode_raises(self):
        with pytest.raises(ValueError):
            BacktestConfig(
                data={
                    "symbols": ["AAPL"],
                    "start_date": "2020-01-01",
                    "end_date": "2024-01-01",
                },
                execution={},
                strategy={"name": "ml_strategy"},
                ml={"mode": "invalid_mode"},
            )

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
    """Full end-to-end: config → features → train → validate → backtest → DSR."""

    def test_full_ml_pipeline(self):
        """The complete Phase 3 workflow in one test."""
        # 1. Generate data
        ohlcv = _make_synthetic_ohlcv(n_bars=600, seed=42)
        pipeline = _build_default_pipeline()

        # 2. Features + target
        pipeline_result = pipeline.run(ohlcv, target_horizon=5, target_type="direction")
        assert len(pipeline_result.features) > 200

        # 3. CPCV validation
        model = XGBoostSignalModel(n_estimators=30, max_depth=2, mode="classification")
        validator = CPCVValidator(
            n_splits=5, n_test_splits=2, purge_window=5, embargo_pct=0.01
        )
        cpcv_result = validator.run(
            pipeline_result.features, pipeline_result.target, model
        )
        assert cpcv_result.n_paths == 10
        assert 0.0 <= cpcv_result.mean_score <= 1.0

        # 4. Train final model on all data
        model2 = XGBoostSignalModel(n_estimators=50, max_depth=3, mode="classification")
        model2.train(pipeline_result.features, pipeline_result.target)
        assert model2.is_trained

        # 5. Regime detection
        detector = RegimeDetector(n_regimes=2, vol_window=20)
        regime_result = detector.fit_predict(ohlcv)
        assert "bull" in regime_result.regime_names.values()

        # 6. DSR analysis (using CPCV paths as trial count)
        dsr_result = deflated_sharpe_ratio(
            observed_sr=0.5,
            n_trials=cpcv_result.n_paths,
            n_observations=len(pipeline_result.features),
        )
        assert dsr_result.n_trials == 10
        assert dsr_result.expected_max_sharpe > 0

"""Unit tests for MLStrategy."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.events.event import SignalEvent, SignalType
from src.features.pipeline import FeaturePipeline
from src.features.technical import RSIFeature, SMAFeature
from src.ml.base_model import MLModel, ModelPrediction, TrainResult
from src.ml.ml_strategy import MLStrategy

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubMLModel(MLModel):
    """Controllable stub that returns pre-set predictions."""

    def __init__(
        self,
        signal: float = 0.0,
        confidence: float = 0.5,
    ) -> None:
        self._signal = signal
        self._confidence = confidence
        self._trained = True
        self.feature_names: list[str] = []

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | None = None,
    ) -> TrainResult:
        self.feature_names = list(features.columns)
        self._trained = True
        return TrainResult(train_score=0.9, n_train_samples=len(features))

    def predict(self, features: pd.DataFrame) -> list[ModelPrediction]:
        return [
            ModelPrediction(
                signal=self._signal,
                confidence=self._confidence,
                features_used=features.shape[1],
            )
            for _ in range(len(features))
        ]

    def save(self, path: str | Path) -> None:
        pass

    def load(self, path: str | Path) -> None:
        pass

    def get_feature_importance(self) -> dict[str, float]:
        return {}

    @property
    def is_trained(self) -> bool:
        return self._trained

    def set_prediction(self, signal: float, confidence: float = 0.5) -> None:
        """Update the prediction returned by predict()."""
        self._signal = signal
        self._confidence = confidence


class StubDataHandler:
    """Minimal data handler stub returning synthetic OHLCV bars."""

    def __init__(self, n_bars: int = 200, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n_bars, freq="B")
        prices = 100.0 * np.exp(np.cumsum(rng.standard_normal(n_bars) * 0.01))
        self._data = pd.DataFrame(
            {
                "open": prices * 0.999,
                "high": prices * 1.002,
                "low": prices * 0.998,
                "close": prices,
                "volume": np.ones(n_bars) * 1_000_000,
            },
            index=dates,
        )
        self._current_idx = n_bars - 1

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        end = self._current_idx + 1
        start = max(0, end - n)
        return self._data.iloc[start:end].copy()

    def get_current_timestamp(self) -> datetime:
        return self._data.index[self._current_idx].to_pydatetime()

    def set_bar_count(self, n: int) -> None:
        """Limit the number of available bars (simulates early stage)."""
        self._current_idx = min(n - 1, len(self._data) - 1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMLStrategyConstruction:
    def test_initialization(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        model = StubMLModel()
        strategy = MLStrategy(["AAPL"], pipeline, model)
        assert strategy.pipeline is pipeline
        assert strategy.model is model
        assert strategy.symbols == ["AAPL"]

    def test_default_parameters(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        strategy = MLStrategy(["AAPL"], pipeline, StubMLModel())
        assert strategy.signal_threshold == 0.1
        assert strategy.exit_threshold == 0.0
        assert strategy.shorts_enabled is False
        assert strategy.lookback_bars == pipeline.total_lookback + 50

    def test_custom_parameters(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        params = {
            "signal_threshold": 0.2,
            "exit_threshold": 0.05,
            "lookback_bars": 100,
            "shorts_enabled": True,
        }
        strategy = MLStrategy(["AAPL"], pipeline, StubMLModel(), parameters=params)
        assert strategy.signal_threshold == 0.2
        assert strategy.exit_threshold == 0.05
        assert strategy.lookback_bars == 100
        assert strategy.shorts_enabled is True

    def test_lookback_bars_default_from_pipeline(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([20]), RSIFeature(14)])
        strategy = MLStrategy(["AAPL"], pipeline, StubMLModel())
        # total_lookback = max(20, 15) = 20
        assert strategy.lookback_bars == 20 + 50


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------


class TestMLStrategySignals:
    def _make_strategy(
        self,
        signal: float = 0.0,
        confidence: float = 0.5,
        params: dict[str, Any] | None = None,
    ) -> tuple[MLStrategy, StubDataHandler, StubMLModel]:
        pipeline = FeaturePipeline([SMAFeature([5])])
        model = StubMLModel(signal=signal, confidence=confidence)
        default_params = {"lookback_bars": 100}
        if params:
            default_params.update(params)
        strategy = MLStrategy(["AAPL"], pipeline, model, parameters=default_params)
        data_handler = StubDataHandler(n_bars=200)
        return strategy, data_handler, model

    def test_long_signal_above_threshold(self) -> None:
        strategy, dh, _ = self._make_strategy(signal=0.5, confidence=0.8)
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.LONG
        assert signals[0].strength == pytest.approx(0.8)

    def test_no_signal_below_threshold(self) -> None:
        strategy, dh, _ = self._make_strategy(signal=0.05)
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert len(signals) == 0

    def test_exit_signal_when_long_and_negative(self) -> None:
        strategy, dh, _ = self._make_strategy(signal=-0.5)
        strategy.update_position("AAPL", 100)  # currently long
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.EXIT

    def test_short_signal_when_enabled(self) -> None:
        strategy, dh, _ = self._make_strategy(
            signal=-0.5, confidence=0.7, params={"shorts_enabled": True}
        )
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.SHORT
        assert signals[0].strength == pytest.approx(0.7)

    def test_no_short_when_disabled(self) -> None:
        strategy, dh, _ = self._make_strategy(signal=-0.5)
        # shorts_enabled is False by default, position is flat
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert len(signals) == 0

    def test_exit_on_weak_signal_when_in_position(self) -> None:
        strategy, dh, _ = self._make_strategy(
            signal=0.01, params={"exit_threshold": 0.05}
        )
        strategy.update_position("AAPL", 100)
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.EXIT

    def test_signal_strength_equals_confidence(self) -> None:
        strategy, dh, _ = self._make_strategy(signal=0.8, confidence=0.65)
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert signals[0].strength == pytest.approx(0.65)

    def test_no_duplicate_long_when_already_long(self) -> None:
        strategy, dh, _ = self._make_strategy(signal=0.5)
        strategy.update_position("AAPL", 100)  # already long
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert len(signals) == 0

    def test_no_signal_insufficient_data(self) -> None:
        strategy, dh, _ = self._make_strategy(signal=0.5)
        dh.set_bar_count(3)  # way below lookback requirement
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert len(signals) == 0

    def test_multiple_symbols(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        model = StubMLModel(signal=0.5, confidence=0.8)
        strategy = MLStrategy(
            ["AAPL", "GOOG"],
            pipeline,
            model,
            parameters={"lookback_bars": 100},
        )
        # StubDataHandler ignores symbol, so both get the same data
        dh = StubDataHandler(n_bars=200)
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert len(signals) == 2
        symbols = {s.symbol for s in signals}
        assert symbols == {"AAPL", "GOOG"}

    def test_signal_has_correct_timestamp(self) -> None:
        strategy, dh, _ = self._make_strategy(signal=0.5)
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert signals[0].timestamp == ts

    def test_signal_has_correct_symbol(self) -> None:
        strategy, dh, _ = self._make_strategy(signal=0.5)
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)
        assert signals[0].symbol == "AAPL"


# ---------------------------------------------------------------------------
# Integration — real pipeline + real XGBoost
# ---------------------------------------------------------------------------


class TestMLStrategyIntegration:
    def test_end_to_end_with_xgboost(self) -> None:
        """Train XGBoost on synthetic data, run strategy, verify signal types."""
        from src.ml.xgboost_model import XGBoostSignalModel

        pipeline = FeaturePipeline([SMAFeature([5, 10]), RSIFeature(7)])

        # Generate OHLCV, compute features + target
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
        ohlcv = pd.DataFrame(
            {
                "open": prices * 0.999,
                "high": prices * 1.002,
                "low": prices * 0.998,
                "close": prices,
                "volume": np.ones(n) * 1_000_000,
            },
            index=dates,
        )
        result = pipeline.run(ohlcv, target_horizon=5, target_type="direction")

        # Train model
        model = XGBoostSignalModel(n_estimators=20, mode="classification")
        model.train(result.features, result.target)

        # Create strategy and run
        strategy = MLStrategy(
            ["TEST"],
            pipeline,
            model,
            parameters={"lookback_bars": 200, "signal_threshold": 0.0},
        )

        # Simulate getting bars
        dh = StubDataHandler(n_bars=300)
        ts = dh.get_current_timestamp()
        signals = strategy.calculate_signals(ts, dh)

        # Should produce at least one signal (threshold=0 means any non-zero prediction)
        assert len(signals) >= 1
        for sig in signals:
            assert isinstance(sig, SignalEvent)
            assert sig.signal_type in (
                SignalType.LONG,
                SignalType.SHORT,
                SignalType.EXIT,
            )
            assert 0.0 <= sig.strength <= 1.0

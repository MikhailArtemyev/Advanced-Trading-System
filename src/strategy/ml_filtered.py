"""ML-filtered trading strategy.

Wraps a base strategy and gates entry signals through an XGBoost
classifier. EXIT signals always pass through. Entry signals are
evaluated by the model; if the predicted probability of profitability
is below the threshold, the signal is suppressed.
"""

import logging
from datetime import datetime
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np

from ..data.data_handler import DataHandler
from ..events.event import SignalEvent, SignalType
from ..ml.features import FEATURE_NAMES, MIN_BARS_REQUIRED, extract_features
from .base_strategy import Strategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .sma_crossover import SMACrossoverStrategy

logger = logging.getLogger(__name__)

_BASE_STRATEGY_MAP: dict[str, type[Strategy]] = {
    "mean_reversion": MeanReversionStrategy,
    "sma_crossover": SMACrossoverStrategy,
    "momentum": MomentumStrategy,
}


class MLFilteredStrategy(Strategy):
    """Wraps a base strategy and filters entry signals through an ML model.

    EXIT signals are always passed through (never block exits).
    LONG/SHORT signals are evaluated by the model; if the model
    predicts the trade is unprofitable (probability < threshold),
    the signal is suppressed.

    Parameters (via ``parameters`` dict):
        base_strategy: Name of the inner strategy (default "mean_reversion").
        model_path: Path to the trained model artifact (.joblib).
        threshold: Minimum probability to accept a trade (default 0.5).

        All other parameters are forwarded to the base strategy.
    """

    def __init__(
        self,
        symbols: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(symbols, parameters)

        # Extract ML-specific parameters
        base_name = self.parameters.get("base_strategy", "mean_reversion")
        model_path = self.parameters.get("model_path", "")
        self._threshold = float(self.parameters.get("threshold", 0.5))

        # Build inner strategy with remaining parameters
        base_params = {
            k: v
            for k, v in self.parameters.items()
            if k not in ("base_strategy", "model_path", "threshold")
        }
        base_cls = _BASE_STRATEGY_MAP.get(base_name)
        if base_cls is None:
            raise ValueError(
                f"Unknown base_strategy '{base_name}'. "
                f"Available: {list(_BASE_STRATEGY_MAP.keys())}"
            )
        self._base = base_cls(symbols=symbols, parameters=base_params)

        # Load ML model
        if model_path:
            artifact = joblib.load(model_path)
            self._model = artifact["model"]
            self._feature_names: list[str] = artifact.get(
                "feature_names", FEATURE_NAMES
            )
            logger.info(
                "Loaded ML filter from %s (threshold=%.2f)",
                model_path,
                self._threshold,
            )
        else:
            self._model = None
            self._feature_names = FEATURE_NAMES
            logger.warning("No model_path — ML filter disabled, passing all signals")

        # Stats
        self.signals_total = 0
        self.signals_passed = 0
        self.signals_blocked = 0

    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        """Run base strategy, then filter entry signals through ML model."""
        # Keep inner strategy positions in sync
        self._base.current_positions = self.current_positions.copy()

        raw_signals = self._base.calculate_signals(timestamp, data_handler)

        if self._model is None:
            return raw_signals

        filtered: list[SignalEvent] = []

        for signal in raw_signals:
            # Always pass EXIT signals
            if signal.signal_type == SignalType.EXIT:
                filtered.append(signal)
                continue

            # Extract features
            bars = data_handler.get_latest_bars(signal.symbol, MIN_BARS_REQUIRED)
            feats = extract_features(bars)
            if feats is None:
                filtered.append(signal)
                continue

            # Predict
            x = np.array([[feats[f] for f in self._feature_names]])
            prob = float(self._model.predict_proba(x)[0, 1])

            self.signals_total += 1
            if prob >= self._threshold:
                signal.strength = signal.strength * prob
                filtered.append(signal)
                self.signals_passed += 1
            else:
                self.signals_blocked += 1
                logger.debug(
                    "ML filter blocked %s %s (prob=%.3f < %.3f)",
                    signal.signal_type.name,
                    signal.symbol,
                    prob,
                    self._threshold,
                )

        return filtered

    def update_position(self, symbol: str, quantity: int) -> None:
        """Update tracked position for both wrapper and base strategy."""
        super().update_position(symbol, quantity)
        self._base.update_position(symbol, quantity)

    def get_filter_stats(self) -> dict[str, int]:
        """Return ML filter statistics."""
        return {
            "signals_total": self.signals_total,
            "signals_passed": self.signals_passed,
            "signals_blocked": self.signals_blocked,
        }

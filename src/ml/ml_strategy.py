"""ML-based trading strategy.

Uses a trained MLModel + FeaturePipeline to generate signals.
At each bar, computes features from recent data and passes them
to the model for prediction. The model's signal and confidence
are used to generate LONG/SHORT/EXIT events.
"""

from datetime import datetime
from typing import Any

from ..data.data_handler import DataHandler
from ..events.event import SignalEvent, SignalType
from ..features.pipeline import FeaturePipeline
from ..strategy.base_strategy import Strategy
from .base_model import MLModel


class MLStrategy(Strategy):
    """Strategy that generates signals from ML model predictions.

    The strategy:
    1. Collects enough bars for feature computation (lookback)
    2. Computes features using the FeaturePipeline
    3. Passes features to the MLModel for prediction
    4. Maps model prediction to SignalEvent

    Signal mapping:
        prediction.signal > signal_threshold  -> LONG (strength = confidence)
        prediction.signal < -signal_threshold -> EXIT (if long) or SHORT (if enabled)
        |prediction.signal| < exit_threshold  -> EXIT (if in position)

    Args:
        symbols: List of symbols to trade.
        pipeline: Feature computation pipeline.
        model: Trained ML model for prediction.
        parameters: Strategy parameters dict. Recognized keys:
            signal_threshold (float): Minimum |signal| for entry. Default 0.1.
            exit_threshold (float): |signal| below which to exit. Default 0.0.
            lookback_bars (int): Bars to fetch for features. Default pipeline.total_lookback + 50.
            shorts_enabled (bool): Whether to allow SHORT signals. Default False.
    """

    def __init__(
        self,
        symbols: list[str],
        pipeline: FeaturePipeline,
        model: MLModel,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(symbols, parameters)
        self.pipeline = pipeline
        self.model = model

        self.signal_threshold: float = self.parameters.get("signal_threshold", 0.1)
        self.exit_threshold: float = self.parameters.get("exit_threshold", 0.0)
        self.lookback_bars: int = self.parameters.get(
            "lookback_bars", pipeline.total_lookback + 50
        )
        self.shorts_enabled: bool = self.parameters.get("shorts_enabled", False)

    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        signals = []
        for symbol in self.symbols:
            signal = self._generate_ml_signal(timestamp, symbol, data_handler)
            if signal is not None:
                signals.append(signal)
        return signals

    def _generate_ml_signal(
        self,
        timestamp: datetime,
        symbol: str,
        data_handler: DataHandler,
    ) -> SignalEvent | None:
        bars = data_handler.get_latest_bars(symbol, self.lookback_bars)

        if len(bars) < self.pipeline.total_lookback + 1:
            return None

        feature_df = self.pipeline.compute_features_only(bars)

        if feature_df.empty:
            return None

        latest_features = feature_df.iloc[[-1]]
        predictions = self.model.predict(latest_features)
        pred = predictions[0]

        current_pos = self.current_positions.get(symbol, 0)

        if pred.signal > self.signal_threshold and current_pos <= 0:
            return self._create_signal(
                timestamp, symbol, SignalType.LONG, strength=pred.confidence
            )
        elif pred.signal < -self.signal_threshold:
            if current_pos > 0:
                return self._create_signal(
                    timestamp, symbol, SignalType.EXIT, strength=1.0
                )
            if current_pos == 0 and self.shorts_enabled:
                return self._create_signal(
                    timestamp, symbol, SignalType.SHORT, strength=pred.confidence
                )
        elif abs(pred.signal) < self.exit_threshold and current_pos != 0:
            return self._create_signal(
                timestamp, symbol, SignalType.EXIT, strength=1.0
            )

        return None

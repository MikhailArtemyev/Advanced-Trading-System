"""Z-Score Mean Reversion Strategy.

Computes a rolling z-score of the close price.  Enters long when the
z-score drops below ``-entry_threshold`` (oversold) and short when it
rises above ``+entry_threshold`` (overbought).  Exits when the z-score
reverts toward zero past ``exit_threshold``, or after
``max_holding_period`` bars.
"""

from datetime import datetime
from typing import Any

from ..data.data_handler import DataHandler
from ..events.event import SignalEvent, SignalType
from .base_strategy import Strategy


class MeanReversionStrategy(Strategy):
    """Z-score based mean reversion.

    Parameters (via ``parameters`` dict):
        lookback_period: Rolling window for mean and std (default 20).
        entry_threshold: Absolute z-score to trigger entry (default 2.0).
        exit_threshold: Absolute z-score to trigger exit (default 0.5).
        max_holding_period: Force-exit after this many bars (default 10).
            Set to 0 to disable.
    """

    def __init__(
        self,
        symbols: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(symbols, parameters)

        self.lookback_period: int = self.parameters.get("lookback_period", 20)
        self.entry_threshold: float = self.parameters.get("entry_threshold", 2.0)
        self.exit_threshold: float = self.parameters.get("exit_threshold", 0.5)
        self.max_holding_period: int = self.parameters.get("max_holding_period", 10)

        if self.lookback_period < 2:
            raise ValueError("lookback_period must be at least 2")
        if self.entry_threshold <= 0:
            raise ValueError("entry_threshold must be > 0")
        if self.exit_threshold < 0:
            raise ValueError("exit_threshold must be >= 0")

        # Track how many bars each position has been held
        self._holding_bars: dict[str, int] = dict.fromkeys(symbols, 0)

    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        """Compute z-score for each symbol and emit entry/exit signals."""
        signals: list[SignalEvent] = []

        for symbol in self.symbols:
            signal = self._evaluate_symbol(timestamp, symbol, data_handler)
            if signal is not None:
                signals.append(signal)

        return signals

    def _evaluate_symbol(
        self,
        timestamp: datetime,
        symbol: str,
        data_handler: DataHandler,
    ) -> SignalEvent | None:
        """Evaluate a single symbol for mean-reversion signals."""
        bars = data_handler.get_latest_bars(symbol, self.lookback_period)
        if len(bars) < self.lookback_period:
            return None

        closes = bars["close"]
        mean = closes.mean()
        std = closes.std()
        if std == 0:
            return None

        z = (closes.iloc[-1] - mean) / std
        pos = self.current_positions.get(symbol, 0)

        # --- Currently flat: look for entry ---
        if pos == 0:
            self._holding_bars[symbol] = 0
            if z <= -self.entry_threshold:
                strength = min(abs(z) / self.entry_threshold, 1.0)
                return self._create_signal(
                    timestamp, symbol, SignalType.LONG, strength=strength
                )
            if z >= self.entry_threshold:
                strength = min(abs(z) / self.entry_threshold, 1.0)
                return self._create_signal(
                    timestamp, symbol, SignalType.SHORT, strength=strength
                )
            return None

        # --- Currently in a position: look for exit ---
        self._holding_bars[symbol] = self._holding_bars.get(symbol, 0) + 1

        # Max holding period exceeded
        if (
            self.max_holding_period > 0
            and self._holding_bars[symbol] >= self.max_holding_period
        ):
            self._holding_bars[symbol] = 0
            return self._create_signal(timestamp, symbol, SignalType.EXIT, strength=1.0)

        # Z-score has reverted close enough to mean
        if pos > 0 and z >= -self.exit_threshold:
            self._holding_bars[symbol] = 0
            return self._create_signal(timestamp, symbol, SignalType.EXIT, strength=1.0)
        if pos < 0 and z <= self.exit_threshold:
            self._holding_bars[symbol] = 0
            return self._create_signal(timestamp, symbol, SignalType.EXIT, strength=1.0)

        return None

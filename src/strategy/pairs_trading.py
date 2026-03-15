"""Statistical Pairs Trading Strategy.

Trades the spread between two cointegrated symbols.  The spread is
computed as ``price_A - hedge_ratio * price_B`` where the hedge ratio
comes from OLS regression.  Entry occurs when the spread z-score
exceeds ``entry_threshold``; exit when it reverts past
``exit_threshold``.
"""

from datetime import datetime
from typing import Any

import numpy as np

from ..data.data_handler import DataHandler
from ..events.event import SignalEvent, SignalType
from .base_strategy import Strategy


class PairsTradingStrategy(Strategy):
    """Statistical arbitrage on a pair of cointegrated symbols.

    The strategy requires exactly two symbols.  Symbol A is the
    "dependent" leg, symbol B the "independent" leg.

    Parameters (via ``parameters`` dict):
        lookback_period: Window for spread stats and hedge ratio (default 60).
        entry_threshold: Spread z-score to open a position (default 2.0).
        exit_threshold: Spread z-score to close (default 0.5).
        min_bars: Minimum bars before trading (default 60).
    """

    def __init__(
        self,
        symbols: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        if len(symbols) != 2:
            raise ValueError("PairsTradingStrategy requires exactly 2 symbols")

        super().__init__(symbols, parameters)

        self.lookback_period: int = self.parameters.get("lookback_period", 60)
        self.entry_threshold: float = self.parameters.get("entry_threshold", 2.0)
        self.exit_threshold: float = self.parameters.get("exit_threshold", 0.5)
        self.min_bars: int = self.parameters.get("min_bars", 60)

        if self.lookback_period < 2:
            raise ValueError("lookback_period must be at least 2")
        if self.entry_threshold <= 0:
            raise ValueError("entry_threshold must be > 0")
        if self.exit_threshold < 0:
            raise ValueError("exit_threshold must be >= 0")

        self.symbol_a: str = symbols[0]
        self.symbol_b: str = symbols[1]

        self._bar_count: int = 0
        self._hedge_ratio: float = 1.0
        self._in_position: bool = False

    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        """Compute spread z-score and emit pair signals."""
        self._bar_count += 1

        if self._bar_count < self.min_bars:
            return []

        bars_a = data_handler.get_latest_bars(self.symbol_a, self.lookback_period)
        bars_b = data_handler.get_latest_bars(self.symbol_b, self.lookback_period)

        if len(bars_a) < self.lookback_period or len(bars_b) < self.lookback_period:
            return []

        closes_a = bars_a["close"].values.astype(float)
        closes_b = bars_b["close"].values.astype(float)

        # Compute hedge ratio via OLS: A = hedge_ratio * B + intercept
        self._hedge_ratio = self._calculate_hedge_ratio(closes_a, closes_b)

        # Compute spread and z-score
        spread = closes_a - self._hedge_ratio * closes_b
        mean = float(np.mean(spread))
        std = float(np.std(spread))
        if std == 0:
            return []

        z = (spread[-1] - mean) / std

        return self._generate_pair_signals(timestamp, z)

    def _generate_pair_signals(
        self, timestamp: datetime, z: float
    ) -> list[SignalEvent]:
        """Generate long/short/exit signals for both legs based on spread z."""
        signals: list[SignalEvent] = []
        pos_a = self.current_positions.get(self.symbol_a, 0)

        if pos_a == 0 and not self._in_position:
            # Not in position — check for entry
            if z >= self.entry_threshold:
                # Spread too high: short A, long B
                strength = min(abs(z) / self.entry_threshold, 1.0)
                signals.append(
                    self._create_signal(
                        timestamp, self.symbol_a, SignalType.SHORT, strength=strength
                    )
                )
                signals.append(
                    self._create_signal(
                        timestamp, self.symbol_b, SignalType.LONG, strength=strength
                    )
                )
                self._in_position = True
            elif z <= -self.entry_threshold:
                # Spread too low: long A, short B
                strength = min(abs(z) / self.entry_threshold, 1.0)
                signals.append(
                    self._create_signal(
                        timestamp, self.symbol_a, SignalType.LONG, strength=strength
                    )
                )
                signals.append(
                    self._create_signal(
                        timestamp, self.symbol_b, SignalType.SHORT, strength=strength
                    )
                )
                self._in_position = True
        elif self._in_position:
            # In position — check for exit
            if abs(z) <= self.exit_threshold:
                signals.append(
                    self._create_signal(
                        timestamp, self.symbol_a, SignalType.EXIT, strength=1.0
                    )
                )
                signals.append(
                    self._create_signal(
                        timestamp, self.symbol_b, SignalType.EXIT, strength=1.0
                    )
                )
                self._in_position = False

        return signals

    @staticmethod
    def _calculate_hedge_ratio(closes_a: np.ndarray, closes_b: np.ndarray) -> float:
        """OLS hedge ratio: regress A on B, return slope."""
        b_mean = np.mean(closes_b)
        a_mean = np.mean(closes_a)
        cov = np.sum((closes_b - b_mean) * (closes_a - a_mean))
        var = np.sum((closes_b - b_mean) ** 2)
        if var == 0:
            return 1.0
        return float(cov / var)

    @property
    def hedge_ratio(self) -> float:
        """Current hedge ratio."""
        return self._hedge_ratio

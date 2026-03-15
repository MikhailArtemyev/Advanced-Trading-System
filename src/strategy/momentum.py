"""Cross-sectional Momentum Strategy.

Ranks symbols by trailing return over a lookback window, then goes long
the top N performers (and optionally short the bottom N).  Rebalances
every ``rebalance_frequency`` bars.
"""

from datetime import datetime
from typing import Any

from ..data.data_handler import DataHandler
from ..events.event import SignalEvent, SignalType
from .base_strategy import Strategy


class MomentumStrategy(Strategy):
    """Cross-sectional momentum: long winners, short losers.

    Parameters (via ``parameters`` dict):
        lookback_period: Bars used to compute trailing return (default 20).
        top_n: Number of top-ranked symbols to go long (default 3).
        bottom_n: Number of bottom-ranked symbols to short (default 0,
            i.e. long-only).
        rebalance_frequency: Bars between rebalances (default 5).
        min_bars: Minimum bars before any trading (default 30).
    """

    def __init__(
        self,
        symbols: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(symbols, parameters)

        self.lookback_period: int = self.parameters.get("lookback_period", 20)
        self.top_n: int = self.parameters.get("top_n", 3)
        self.bottom_n: int = self.parameters.get("bottom_n", 0)
        self.rebalance_frequency: int = self.parameters.get("rebalance_frequency", 5)
        self.min_bars: int = self.parameters.get("min_bars", 30)

        if self.lookback_period < 2:
            raise ValueError("lookback_period must be at least 2")
        if self.top_n < 0:
            raise ValueError("top_n must be >= 0")
        if self.bottom_n < 0:
            raise ValueError("bottom_n must be >= 0")
        if self.top_n + self.bottom_n > len(symbols) and len(symbols) > 0:
            raise ValueError("top_n + bottom_n exceeds number of symbols")

        self._bar_count: int = 0

    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        """Rank symbols by return and emit long/short/exit signals."""
        self._bar_count += 1

        if self._bar_count < self.min_bars:
            return []
        if self._bar_count % self.rebalance_frequency != 0:
            return []

        # Compute trailing returns for each symbol
        returns: dict[str, float] = {}
        for symbol in self.symbols:
            bars = data_handler.get_latest_bars(symbol, self.lookback_period)
            if len(bars) < self.lookback_period:
                continue
            closes = bars["close"]
            ret = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
            returns[symbol] = ret

        if not returns:
            return []

        ranked = sorted(returns.keys(), key=lambda s: returns[s], reverse=True)

        long_symbols = set(ranked[: self.top_n])
        short_symbols = set(ranked[-self.bottom_n :]) if self.bottom_n > 0 else set()

        signals: list[SignalEvent] = []

        for symbol in self.symbols:
            pos = self.current_positions.get(symbol, 0)

            if symbol in long_symbols:
                if pos <= 0:
                    # Normalise strength by rank position within long bucket
                    idx = ranked.index(symbol)
                    strength = 1.0 - idx / max(len(ranked), 1)
                    signals.append(
                        self._create_signal(
                            timestamp, symbol, SignalType.LONG, strength=strength
                        )
                    )
            elif symbol in short_symbols:
                if pos >= 0:
                    idx = ranked.index(symbol)
                    strength = idx / max(len(ranked), 1)
                    signals.append(
                        self._create_signal(
                            timestamp, symbol, SignalType.SHORT, strength=strength
                        )
                    )
            else:
                # Not in long or short bucket — exit any position
                if pos != 0:
                    signals.append(
                        self._create_signal(
                            timestamp, symbol, SignalType.EXIT, strength=1.0
                        )
                    )

        return signals

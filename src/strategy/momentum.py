"""Cross-sectional Momentum Strategy.

Ranks symbols by trailing return over a lookback window, then goes long
the top N performers (and optionally short the bottom N).  Rebalances
every ``rebalance_frequency`` bars.

Per-position risk management: percentage trailing stop-loss cuts losers
between rebalances.  Optional regime filter and minimum return threshold.

Market crash detector: monitors short-term average return across the
universe every bar.  On sharp drops, exits all positions immediately
and waits for recovery before re-entering (fast out, slow in).
"""

from datetime import datetime
from typing import Any

import numpy as np

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
        min_return: Minimum trailing return to qualify for long entry
            (default 0.0). Stocks below this threshold are skipped.
        regime_sma_period: SMA period for the regime filter (default 0,
            disabled).
        trailing_stop_pct: Trailing stop as a percentage (default 0,
            disabled). E.g. 0.08 = exit if price drops 8% from its
            high since entry. Checked every bar.
        crash_lookback: Short-term bars to detect market crash (default 0,
            disabled). E.g. 2 = check average 2-day return across
            the universe.
        crash_threshold: Average return below this triggers emergency
            exit (default -0.03). E.g. -0.03 = exit if the universe
            average drops 3% over ``crash_lookback`` bars.
        recovery_bars: Bars to stay in cash after crash detected
            (default 10). Slow re-entry prevents whipsaw.
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
        self.min_return: float = self.parameters.get("min_return", 0.0)
        self.regime_sma_period: int = self.parameters.get("regime_sma_period", 0)
        self.trailing_stop_pct: float = self.parameters.get("trailing_stop_pct", 0)
        self.crash_lookback: int = self.parameters.get("crash_lookback", 0)
        self.crash_threshold: float = self.parameters.get("crash_threshold", -0.03)
        self.recovery_bars: int = self.parameters.get("recovery_bars", 10)

        if self.lookback_period < 2:
            raise ValueError("lookback_period must be at least 2")
        if self.top_n < 0:
            raise ValueError("top_n must be >= 0")
        if self.bottom_n < 0:
            raise ValueError("bottom_n must be >= 0")
        if self.top_n + self.bottom_n > len(symbols) and len(symbols) > 0:
            raise ValueError("top_n + bottom_n exceeds number of symbols")

        self._bar_count: int = 0
        self._high_water: dict[str, float] = {}
        # Market crash detector state
        self._cooldown_until: int = 0  # bar count when cooldown ends

    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        """Rank symbols by return and emit long/short/exit signals."""
        self._bar_count += 1

        if self._bar_count < self.min_bars:
            return []

        # Market crash detector — fast exit, slow re-entry
        if self.crash_lookback > 0:
            if self._detect_crash(data_handler):
                self._cooldown_until = self._bar_count + self.recovery_bars
                return self._exit_all(timestamp)

        # During cooldown after crash: stay flat
        if self._bar_count < self._cooldown_until:
            return self._exit_all(timestamp)

        # Check trailing stops every bar
        signals = self._check_stops(timestamp, data_handler)

        # Only rebalance on schedule
        if self._bar_count % self.rebalance_frequency != 0:
            return signals

        # Regime filter
        risk_off = self._is_risk_off(data_handler)

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
            return signals

        ranked = sorted(returns.keys(), key=lambda s: returns[s], reverse=True)

        if risk_off:
            long_symbols: set[str] = set()
            short_symbols: set[str] = set()
        else:
            long_symbols = set()
            for s in ranked[: self.top_n]:
                if returns[s] >= self.min_return:
                    long_symbols.add(s)
            short_symbols = (
                set(ranked[-self.bottom_n :]) if self.bottom_n > 0 else set()
            )

        stopped_out = {s.symbol for s in signals}

        for symbol in self.symbols:
            if symbol in stopped_out:
                continue

            pos = self.current_positions.get(symbol, 0)

            if symbol in long_symbols:
                if pos <= 0:
                    idx = ranked.index(symbol)
                    strength = 1.0 - idx / max(len(ranked), 1)
                    signals.append(
                        self._create_signal(
                            timestamp, symbol, SignalType.LONG, strength=strength
                        )
                    )
                    # Set high water mark at entry
                    bars = data_handler.get_latest_bars(symbol, 1)
                    self._high_water[symbol] = float(bars["close"].iloc[-1])
            elif symbol in short_symbols:
                if pos >= 0:
                    idx = ranked.index(symbol)
                    strength = idx / max(len(ranked), 1)
                    signals.append(
                        self._create_signal(
                            timestamp, symbol, SignalType.SHORT, strength=strength
                        )
                    )
                    bars = data_handler.get_latest_bars(symbol, 1)
                    self._high_water[symbol] = float(bars["close"].iloc[-1])
            else:
                if pos != 0:
                    signals.append(
                        self._create_signal(
                            timestamp, symbol, SignalType.EXIT, strength=1.0
                        )
                    )
                    self._high_water.pop(symbol, None)

        return signals

    def _check_stops(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        """Check percentage trailing stop for all open positions."""
        if self.trailing_stop_pct <= 0:
            return []

        signals: list[SignalEvent] = []

        for symbol in self.symbols:
            pos = self.current_positions.get(symbol, 0)
            if pos == 0 or symbol not in self._high_water:
                continue

            bars = data_handler.get_latest_bars(symbol, 1)
            if bars.empty:
                continue

            price = float(bars["close"].iloc[-1])

            if pos > 0:
                hw = self._high_water[symbol]
                if price > hw:
                    self._high_water[symbol] = price
                    hw = price
                drawdown = (hw - price) / hw
                if drawdown >= self.trailing_stop_pct:
                    signals.append(
                        self._create_signal(
                            timestamp, symbol, SignalType.EXIT, strength=1.0
                        )
                    )
                    self._high_water.pop(symbol, None)

            elif pos < 0:
                lw = self._high_water[symbol]
                if price < lw:
                    self._high_water[symbol] = price
                    lw = price
                drawup = (price - lw) / lw if lw > 0 else 0
                if drawup >= self.trailing_stop_pct:
                    signals.append(
                        self._create_signal(
                            timestamp, symbol, SignalType.EXIT, strength=1.0
                        )
                    )
                    self._high_water.pop(symbol, None)

        return signals

    def _is_risk_off(self, data_handler: DataHandler) -> bool:
        """Check if the market regime is bearish."""
        if self.regime_sma_period <= 0:
            return False

        n_bars = self.regime_sma_period
        closes_list: list[float] = []
        sma_list: list[float] = []

        for symbol in self.symbols:
            bars = data_handler.get_latest_bars(symbol, n_bars)
            if len(bars) < n_bars:
                continue
            closes_list.append(float(bars["close"].iloc[-1]))
            sma_list.append(float(np.mean(bars["close"].values)))

        if not closes_list:
            return False

        avg_close = np.mean(closes_list)
        avg_sma = np.mean(sma_list)
        return bool(avg_close < avg_sma)

    def _detect_crash(self, data_handler: DataHandler) -> bool:
        """Detect a market crash via short-term average return.

        Computes the average return across the universe over
        ``crash_lookback`` bars.  Returns True if the average return
        is below ``crash_threshold``.
        """
        n = self.crash_lookback
        returns: list[float] = []

        for symbol in self.symbols:
            bars = data_handler.get_latest_bars(symbol, n + 1)
            if len(bars) < n + 1:
                continue
            closes = bars["close"].values
            ret = (closes[-1] - closes[-(n + 1)]) / closes[-(n + 1)]
            returns.append(ret)

        if len(returns) < 3:
            return False

        avg_return = float(np.mean(returns))
        return avg_return <= self.crash_threshold

    def _exit_all(self, timestamp: datetime) -> list[SignalEvent]:
        """Exit all open positions."""
        signals: list[SignalEvent] = []
        for symbol in self.symbols:
            pos = self.current_positions.get(symbol, 0)
            if pos != 0:
                signals.append(
                    self._create_signal(
                        timestamp, symbol, SignalType.EXIT, strength=1.0
                    )
                )
                self._high_water.pop(symbol, None)
        return signals

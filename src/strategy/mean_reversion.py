"""Z-Score Mean Reversion Strategy.

Computes a rolling z-score of the close price.  Enters long when the
z-score drops below ``-entry_threshold`` (oversold) and short when it
rises above ``+entry_threshold`` (overbought).  Exits when the z-score
reverts toward zero past ``exit_threshold``, or after
``max_holding_period`` bars.

Risk management per trade:
- ATR-based stop-loss cuts losers early.
- ATR-based take-profit locks in winners.
- ADX regime filter skips trending markets where mean reversion fails.
"""

from datetime import datetime
from typing import Any

import numpy as np

from ..data.data_handler import DataHandler
from ..events.event import SignalEvent, SignalType
from .base_strategy import Strategy


class MeanReversionStrategy(Strategy):
    """Z-score based mean reversion with per-trade risk management.

    Parameters (via ``parameters`` dict):
        lookback_period: Rolling window for mean and std (default 20).
        entry_threshold: Absolute z-score to trigger entry (default 2.0).
        exit_threshold: Absolute z-score to trigger exit (default 0.5).
        max_holding_period: Force-exit after this many bars (default 10).
            Set to 0 to disable.
        stop_loss_atr: Stop-loss distance in ATR multiples (default 1.5).
            Set to 0 to disable.
        take_profit_atr: Take-profit distance in ATR multiples (default 2.5).
            Set to 0 to disable.
        adx_max: Maximum ADX for entry (default 25.0). Higher ADX means
            stronger trend — mean reversion works best in ranging markets.
            Set to 0 to disable the filter.
        atr_period: Lookback for ATR and ADX computation (default 14).
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
        self.stop_loss_atr: float = self.parameters.get("stop_loss_atr", 1.5)
        self.take_profit_atr: float = self.parameters.get("take_profit_atr", 2.5)
        self.adx_max: float = self.parameters.get("adx_max", 25.0)
        self.atr_period: int = self.parameters.get("atr_period", 14)

        if self.lookback_period < 2:
            raise ValueError("lookback_period must be at least 2")
        if self.entry_threshold <= 0:
            raise ValueError("entry_threshold must be > 0")
        if self.exit_threshold < 0:
            raise ValueError("exit_threshold must be >= 0")

        # Track how many bars each position has been held
        self._holding_bars: dict[str, int] = dict.fromkeys(symbols, 0)
        # Track entry price and ATR at entry for stop/take-profit
        self._entry_price: dict[str, float] = {}
        self._entry_atr: dict[str, float] = {}

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
        n_bars = max(self.lookback_period, self.atr_period + 1)
        bars = data_handler.get_latest_bars(symbol, n_bars)
        if len(bars) < self.lookback_period:
            return None

        closes = bars["close"].values
        highs = bars["high"].values
        lows = bars["low"].values

        # Z-score over lookback window
        lb = closes[-self.lookback_period :]
        mean = float(np.mean(lb))
        std = float(np.std(lb, ddof=1))
        if std == 0:
            return None

        z = (closes[-1] - mean) / std
        pos = self.current_positions.get(symbol, 0)
        current_price = float(closes[-1])

        # ATR for stop/take-profit
        atr = self._compute_atr(highs, lows, closes)

        # --- Currently flat: look for entry ---
        if pos == 0:
            self._holding_bars[symbol] = 0
            return self._check_entry(timestamp, symbol, z, current_price, atr, bars)

        # --- Currently in a position: look for exit ---
        self._holding_bars[symbol] = self._holding_bars.get(symbol, 0) + 1

        # Stop-loss check
        if self.stop_loss_atr > 0 and symbol in self._entry_price:
            stop_dist = self._entry_atr[symbol] * self.stop_loss_atr
            entry = self._entry_price[symbol]
            if pos > 0 and current_price <= entry - stop_dist:
                return self._exit(timestamp, symbol)
            if pos < 0 and current_price >= entry + stop_dist:
                return self._exit(timestamp, symbol)

        # Take-profit check
        if self.take_profit_atr > 0 and symbol in self._entry_price:
            tp_dist = self._entry_atr[symbol] * self.take_profit_atr
            entry = self._entry_price[symbol]
            if pos > 0 and current_price >= entry + tp_dist:
                return self._exit(timestamp, symbol)
            if pos < 0 and current_price <= entry - tp_dist:
                return self._exit(timestamp, symbol)

        # Max holding period exceeded
        if (
            self.max_holding_period > 0
            and self._holding_bars[symbol] >= self.max_holding_period
        ):
            return self._exit(timestamp, symbol)

        # Z-score has reverted close enough to mean
        if pos > 0 and z >= -self.exit_threshold:
            return self._exit(timestamp, symbol)
        if pos < 0 and z <= self.exit_threshold:
            return self._exit(timestamp, symbol)

        return None

    def _check_entry(
        self,
        timestamp: datetime,
        symbol: str,
        z: float,
        price: float,
        atr: float,
        bars: "Any",
    ) -> SignalEvent | None:
        """Check entry conditions including ADX regime filter."""
        # ADX filter: skip trending markets
        if self.adx_max > 0:
            adx = self._compute_adx(
                bars["high"].values, bars["low"].values, bars["close"].values
            )
            if adx > self.adx_max:
                return None

        if z <= -self.entry_threshold:
            strength = min(abs(z) / self.entry_threshold, 1.0)
            self._entry_price[symbol] = price
            self._entry_atr[symbol] = atr
            return self._create_signal(
                timestamp, symbol, SignalType.LONG, strength=strength
            )
        if z >= self.entry_threshold:
            strength = min(abs(z) / self.entry_threshold, 1.0)
            self._entry_price[symbol] = price
            self._entry_atr[symbol] = atr
            return self._create_signal(
                timestamp, symbol, SignalType.SHORT, strength=strength
            )
        return None

    def _exit(self, timestamp: datetime, symbol: str) -> SignalEvent:
        """Emit EXIT signal and clean up tracking state."""
        self._holding_bars[symbol] = 0
        self._entry_price.pop(symbol, None)
        self._entry_atr.pop(symbol, None)
        return self._create_signal(timestamp, symbol, SignalType.EXIT, strength=1.0)

    def _compute_atr(
        self,
        highs: "np.ndarray[Any, Any]",
        lows: "np.ndarray[Any, Any]",
        closes: "np.ndarray[Any, Any]",
    ) -> float:
        """Compute Average True Range over atr_period bars."""
        n = min(self.atr_period, len(closes) - 1)
        if n < 1:
            return float(highs[-1] - lows[-1])

        tr = np.maximum(
            highs[-n:] - lows[-n:],
            np.maximum(
                np.abs(highs[-n:] - closes[-(n + 1) : -1]),
                np.abs(lows[-n:] - closes[-(n + 1) : -1]),
            ),
        )
        return float(np.mean(tr))

    def _compute_adx(
        self,
        highs: "np.ndarray[Any, Any]",
        lows: "np.ndarray[Any, Any]",
        closes: "np.ndarray[Any, Any]",
    ) -> float:
        """Compute ADX (Average Directional Index) over atr_period bars.

        Returns 0.0 if insufficient data.
        """
        period = self.atr_period
        if len(closes) < period + 1:
            return 0.0

        # True Range
        h = highs[-(period + 1) :]
        lo = lows[-(period + 1) :]
        c = closes[-(period + 1) :]

        tr = np.maximum(
            h[1:] - lo[1:],
            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])),
        )

        # Directional movement
        up_move = h[1:] - h[:-1]
        down_move = lo[:-1] - lo[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr = float(np.mean(tr))
        if atr == 0:
            return 0.0

        plus_di = 100.0 * float(np.mean(plus_dm)) / atr
        minus_di = 100.0 * float(np.mean(minus_dm)) / atr

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0

        dx = 100.0 * abs(plus_di - minus_di) / di_sum
        return dx

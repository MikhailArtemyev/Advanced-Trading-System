"""Aggregates raw ticks into OHLCV bars on a configurable time interval.

The BarAggregator collects ticks and emits completed bars when the
time boundary is crossed. Supports any interval (1s, 1m, 5m, 1h, etc.).
Partial bars are available via get_current_bar() for real-time display.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .data_feed import Tick

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single OHLCV bar.

    Attributes:
        symbol: Instrument identifier.
        timestamp: Bar open time (start of the interval).
        open: Opening price.
        high: Highest price during the bar.
        low: Lowest price during the bar.
        close: Closing price (last tick price).
        volume: Total volume during the bar.
        tick_count: Number of ticks aggregated.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int


class BarAggregator:
    """Aggregates ticks into time-based OHLCV bars.

    Args:
        interval: Bar duration (e.g. timedelta(minutes=1)).
        on_bar: Optional callback invoked when a bar completes.

    Raises:
        ValueError: If interval is not positive.
    """

    def __init__(
        self,
        interval: timedelta,
        on_bar: Callable[[Bar], None] | None = None,
    ) -> None:
        if interval.total_seconds() <= 0:
            raise ValueError("interval must be positive")
        self._interval = interval
        self._on_bar = on_bar
        self._current_bars: dict[str, Bar] = {}
        self._completed_bars: dict[str, list[Bar]] = {}
        # Gap detection
        self._expected_next: dict[str, datetime] = {}
        self._gaps: list[dict[str, Any]] = []

    @property
    def interval(self) -> timedelta:
        """The bar interval duration."""
        return self._interval

    @property
    def bar_callback(self) -> Callable[[Bar], None] | None:
        """The current bar-completion callback."""
        return self._on_bar

    @bar_callback.setter
    def bar_callback(self, callback: Callable[[Bar], None] | None) -> None:
        """Set the bar-completion callback."""
        self._on_bar = callback

    def on_tick(self, tick: Tick) -> None:
        """Process a tick, updating the current bar or completing one.

        This method is designed to be registered as a LiveDataFeed listener.
        """
        bar_start = self._floor_timestamp(tick.timestamp)

        if tick.symbol not in self._current_bars:
            self._current_bars[tick.symbol] = Bar(
                symbol=tick.symbol,
                timestamp=bar_start,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.volume,
                tick_count=1,
            )
            return

        current = self._current_bars[tick.symbol]

        if bar_start > current.timestamp:
            # New bar period — emit the completed bar and start fresh
            self._emit_bar(current)

            # Gap detection
            expected = current.timestamp + self._interval
            if bar_start > expected + self._interval:
                gap_duration = bar_start - expected
                self._gaps.append(
                    {
                        "symbol": tick.symbol,
                        "expected": expected,
                        "actual": bar_start,
                        "duration_seconds": gap_duration.total_seconds(),
                    }
                )
                logger.warning(
                    "Data gap detected for %s: expected %s, got %s (%.0fs)",
                    tick.symbol,
                    expected,
                    bar_start,
                    gap_duration.total_seconds(),
                )

            self._current_bars[tick.symbol] = Bar(
                symbol=tick.symbol,
                timestamp=bar_start,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.volume,
                tick_count=1,
            )
        else:
            # Same bar period — update OHLCV
            current.high = max(current.high, tick.price)
            current.low = min(current.low, tick.price)
            current.close = tick.price
            current.volume += tick.volume
            current.tick_count += 1

    def get_current_bar(self, symbol: str) -> Bar | None:
        """Return the in-progress (partial) bar for a symbol."""
        return self._current_bars.get(symbol)

    def get_completed_bars(self, symbol: str, n: int | None = None) -> list[Bar]:
        """Return the last n completed bars for a symbol."""
        bars = self._completed_bars.get(symbol, [])
        if n is not None:
            return bars[-n:]
        return list(bars)

    def get_bars_as_dataframe(self, symbol: str, n: int | None = None) -> pd.DataFrame:
        """Return completed bars as a DataFrame with OHLCV columns.

        Compatible with the DataHandler interface — same column names
        as HistoricalCSVDataHandler.
        """
        bars = self.get_completed_bars(symbol, n)
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        records = [
            {
                "datetime": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        df = pd.DataFrame(records)
        df.set_index("datetime", inplace=True)
        return df

    def get_gaps(self) -> list[dict[str, Any]]:
        """Return detected data gaps.

        Each gap is a dict with keys: symbol, expected, actual, duration_seconds.
        """
        return list(self._gaps)

    def flush(self, symbol: str | None = None) -> None:
        """Force-complete the current partial bar(s).

        Useful at session end or when switching to a new trading day.

        Args:
            symbol: Flush only this symbol. If None, flush all.
        """
        if symbol is not None:
            bar = self._current_bars.pop(symbol, None)
            if bar is not None:
                self._emit_bar(bar)
        else:
            for sym in list(self._current_bars.keys()):
                bar = self._current_bars.pop(sym)
                self._emit_bar(bar)

    def reset(self) -> None:
        """Clear all bars and state."""
        self._current_bars.clear()
        self._completed_bars.clear()
        self._expected_next.clear()
        self._gaps.clear()

    def _floor_timestamp(self, ts: datetime) -> datetime:
        """Round timestamp down to the nearest bar boundary."""
        seconds = int(self._interval.total_seconds())
        epoch = ts.timestamp()
        floored = (int(epoch) // seconds) * seconds
        return datetime.fromtimestamp(floored, tz=ts.tzinfo)

    def _emit_bar(self, bar: Bar) -> None:
        """Store a completed bar and notify the callback."""
        if bar.symbol not in self._completed_bars:
            self._completed_bars[bar.symbol] = []
        self._completed_bars[bar.symbol].append(bar)

        if self._on_bar is not None:
            self._on_bar(bar)

        logger.debug(
            "Bar completed: %s %s O=%.4f H=%.4f L=%.4f C=%.4f V=%.1f (%d ticks)",
            bar.symbol,
            bar.timestamp,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            bar.tick_count,
        )

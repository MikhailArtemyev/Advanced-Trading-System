"""LiveDataHandler — bridges live data feeds to the DataHandler interface.

Wraps a BarAggregator to provide the same get_latest_bars() / update_bars()
interface that strategies and the engine expect. This allows strategies to
run unchanged against both historical and live data.
"""

import logging
from datetime import datetime

import pandas as pd

from src.data.data_handler import DataHandler

from .bar_aggregator import Bar, BarAggregator

logger = logging.getLogger(__name__)


class LiveDataHandler(DataHandler):
    """DataHandler implementation backed by a live data feed.

    Wraps a BarAggregator and presents completed bars through the
    standard DataHandler interface. Strategies using get_latest_bars()
    work identically to backtest mode.

    Args:
        symbols: List of symbols to track.
        bar_aggregator: BarAggregator receiving live ticks.
        max_history: Maximum number of bars to retain per symbol.
    """

    def __init__(
        self,
        symbols: list[str],
        bar_aggregator: BarAggregator,
        max_history: int = 5000,
    ) -> None:
        self._symbols = symbols
        self._bar_aggregator = bar_aggregator
        self._max_history = max_history
        self._current_timestamp: datetime | None = None
        self._bar_count = 0
        self._new_bar_available = False

        # Register as a bar listener
        self._bar_aggregator.bar_callback = self._on_new_bar

    def _on_new_bar(self, bar: Bar) -> None:
        """Called by the BarAggregator when a bar completes."""
        self._current_timestamp = bar.timestamp
        self._bar_count += 1
        self._new_bar_available = True
        logger.debug("New bar: %s at %s", bar.symbol, bar.timestamp)

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return the last n completed bars as a DataFrame.

        Returns the same format as HistoricalCSVDataHandler:
        columns = [open, high, low, close, volume], DatetimeIndex.
        """
        return self._bar_aggregator.get_bars_as_dataframe(symbol, n)

    def update_bars(self) -> bool:
        """Check if new bars are available since last call.

        In live mode this doesn't advance an index — it checks whether
        the BarAggregator has emitted new bars since the last call.
        Returns True if new data was available, False otherwise.
        """
        if self._new_bar_available:
            self._new_bar_available = False
            return True
        return False

    def get_current_timestamp(self) -> datetime:
        """Return the timestamp of the most recent completed bar."""
        if self._current_timestamp is None:
            return datetime.now()
        return self._current_timestamp

    def get_all_symbols(self) -> list[str]:
        """Return all tracked symbols."""
        return list(self._symbols)

    def get_latest_bar_value(self, symbol: str, field: str) -> float:
        """Return a single field from the most recent bar.

        Args:
            symbol: The instrument symbol.
            field: Column name (e.g., 'close', 'volume').

        Returns:
            The value of the specified field.

        Raises:
            ValueError: If no bars are available for the symbol.
        """
        bars = self.get_latest_bars(symbol, n=1)
        if bars.empty:
            raise ValueError(f"No bars available for {symbol}")
        return float(bars[field].iloc[-1])

    @property
    def continue_backtest(self) -> bool:
        """Always True in live mode — runs until explicitly stopped."""
        return True

    def reset(self) -> None:
        """Reset the live data handler state."""
        self._bar_aggregator.reset()
        self._current_timestamp = None
        self._bar_count = 0
        self._new_bar_available = False

    @property
    def bar_count(self) -> int:
        """Number of bars received since creation or last reset."""
        return self._bar_count

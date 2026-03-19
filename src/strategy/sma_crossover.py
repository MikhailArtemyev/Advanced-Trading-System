"""Simple Moving Average Crossover Strategy.

Generates trading signals based on the crossover of two moving averages:
- LONG signal when short SMA crosses above long SMA (bullish)
- EXIT signal when short SMA crosses below long SMA (bearish)
"""

from datetime import datetime
from typing import Any

from ..data.data_handler import DataHandler
from ..events.event import SignalEvent, SignalType
from .base_strategy import Strategy
from .crossover import detect_sma_crossover


class SMACrossoverStrategy(Strategy):
    """Simple Moving Average Crossover Strategy.

    Generates LONG signal when short SMA crosses above long SMA.
    Generates EXIT signal when short SMA crosses below long SMA.

    Parameters:
        short_window: Period for short SMA (default: 20)
        long_window: Period for long SMA (default: 50)

    Attributes:
        short_window: Short moving average period
        long_window: Long moving average period
        prev_short_sma: Previous short SMA values per symbol
        prev_long_sma: Previous long SMA values per symbol
    """

    def __init__(
        self,
        symbols: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the SMA Crossover strategy.

        Args:
            symbols: List of symbols to trade
            parameters: Must include 'short_window' and 'long_window'

        Raises:
            ValueError: If short_window >= long_window
        """
        super().__init__(symbols, parameters)

        # Extract parameters with defaults
        self.short_window: int = self.parameters.get("short_window", 20)
        self.long_window: int = self.parameters.get("long_window", 50)

        # Validate parameters
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")

        if self.short_window < 1:
            raise ValueError("short_window must be at least 1")

        # Track previous SMA values for crossover detection
        self.prev_short_sma: dict[str, float] = {}
        self.prev_long_sma: dict[str, float] = {}

    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        """Calculate SMA crossover signals for all symbols.

        Logic:
        - If short SMA crosses ABOVE long SMA -> LONG signal
        - If short SMA crosses BELOW long SMA -> EXIT signal

        Args:
            timestamp: Current simulation time
            data_handler: Access to market data

        Returns:
            List of SignalEvent objects (may be empty)
        """
        signals = []

        for symbol in self.symbols:
            signal = self._check_crossover(timestamp, symbol, data_handler)
            if signal is not None:
                signals.append(signal)

        return signals

    def _check_crossover(
        self,
        timestamp: datetime,
        symbol: str,
        data_handler: DataHandler,
    ) -> SignalEvent | None:
        """Check for SMA crossover on a single symbol.

        Args:
            timestamp: Current simulation time
            symbol: The symbol to check
            data_handler: Access to market data

        Returns:
            SignalEvent if crossover detected, None otherwise
        """
        bars = data_handler.get_latest_bars(symbol, self.long_window + 1)

        if len(bars) < self.long_window:
            return None

        crossover, short_sma, long_sma = detect_sma_crossover(
            bars["close"],
            self.short_window,
            self.long_window,
            self.prev_short_sma.get(symbol),
            self.prev_long_sma.get(symbol),
        )
        self.prev_short_sma[symbol] = short_sma
        self.prev_long_sma[symbol] = long_sma

        if crossover == "bullish" and self.current_positions.get(symbol, 0) <= 0:
            return self._create_signal(timestamp, symbol, SignalType.LONG, strength=1.0)
        elif crossover == "bearish" and self.current_positions.get(symbol, 0) > 0:
            return self._create_signal(timestamp, symbol, SignalType.EXIT, strength=1.0)

        return None

    def get_sma_values(self, symbol: str) -> tuple[float | None, float | None]:
        """Get current SMA values for a symbol.

        Useful for debugging and visualization.

        Args:
            symbol: The symbol

        Returns:
            Tuple of (short_sma, long_sma), values may be None if not yet calculated
        """
        return (
            self.prev_short_sma.get(symbol),
            self.prev_long_sma.get(symbol),
        )

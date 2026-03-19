"""Multi-Asset SMA Crossover Strategy with optional portfolio optimization.

Extends the SMA crossover approach to trade multiple assets simultaneously,
with periodic weight rebalancing via a PortfolioOptimizer.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from ..data.data_handler import DataHandler
from ..events.event import SignalEvent, SignalType
from ..optimization.base_optimizer import PortfolioOptimizer
from .base_strategy import Strategy
from .crossover import detect_sma_crossover


class MultiAssetSMAStrategy(Strategy):
    """Multi-asset SMA crossover with optional portfolio optimization.

    Per-symbol SMA crossover detection, with signal strength scaled by
    target weights from a PortfolioOptimizer. Without an optimizer,
    equal weights are used for all symbols.

    Parameters:
        short_window: Period for short SMA (default: 20)
        long_window: Period for long SMA (default: 50)
        rebalance_frequency: Bars between weight recalculations (default: 20)

    Attributes:
        short_window: Short moving average period
        long_window: Long moving average period
        rebalance_frequency: Number of bars between optimizer runs
        optimizer: Optional portfolio optimizer for weight calculation
        target_weights: Current target allocation weights per symbol
        prev_short_sma: Previous short SMA values per symbol
        prev_long_sma: Previous long SMA values per symbol
        bars_since_rebalance: Counter tracking bars since last rebalance
    """

    def __init__(
        self,
        symbols: list[str],
        parameters: dict[str, Any] | None = None,
        optimizer: PortfolioOptimizer | None = None,
    ) -> None:
        """Initialize the Multi-Asset SMA strategy.

        Args:
            symbols: List of symbols to trade
            parameters: Strategy parameters (short_window, long_window,
                       rebalance_frequency)
            optimizer: Optional portfolio optimizer for weight rebalancing

        Raises:
            ValueError: If short_window >= long_window, short_window < 1,
                       or rebalance_frequency < 1
        """
        super().__init__(symbols, parameters)

        self.short_window: int = self.parameters.get("short_window", 20)
        self.long_window: int = self.parameters.get("long_window", 50)
        self.rebalance_frequency: int = self.parameters.get("rebalance_frequency", 20)

        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        if self.short_window < 1:
            raise ValueError("short_window must be at least 1")
        if self.rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be at least 1")

        self.optimizer = optimizer

        # Equal weights initially
        n = len(symbols)
        self.target_weights: dict[str, float] = (
            dict.fromkeys(symbols, 1.0 / n) if n > 0 else {}
        )

        # Per-symbol SMA tracking
        self.prev_short_sma: dict[str, float] = {}
        self.prev_long_sma: dict[str, float] = {}

        # Rebalance counter
        self.bars_since_rebalance: int = 0

    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        """Calculate SMA crossover signals for all symbols.

        Periodically rebalances target weights via the optimizer (if provided),
        then checks each symbol for SMA crossovers.

        Args:
            timestamp: Current simulation time
            data_handler: Access to market data

        Returns:
            List of SignalEvent objects (may be empty)
        """
        self.bars_since_rebalance += 1
        self._maybe_rebalance(data_handler)

        signals = []
        for symbol in self.symbols:
            signal = self._check_crossover(timestamp, symbol, data_handler)
            if signal is not None:
                signals.append(signal)
        return signals

    def _maybe_rebalance(self, data_handler: DataHandler) -> None:
        """Rebalance target weights using the optimizer if due.

        Skips if no optimizer is configured or if the rebalance counter
        has not reached rebalance_frequency yet.
        """
        if self.optimizer is None:
            return
        if self.bars_since_rebalance < self.rebalance_frequency:
            return

        self.bars_since_rebalance = 0

        returns_df = self._build_returns_df(data_handler)
        if returns_df.empty:
            return

        result = self.optimizer.optimize(
            self.symbols,
            returns_df,
            self.target_weights.copy(),
        )
        self.target_weights = result.weights

    def _build_returns_df(
        self,
        data_handler: DataHandler,
        lookback: int = 60,
    ) -> pd.DataFrame:
        """Build a returns DataFrame from recent price data for all symbols.

        Args:
            data_handler: Access to market data
            lookback: Number of bars to use for returns calculation

        Returns:
            DataFrame with daily returns per symbol (columns = symbols).
            Empty DataFrame if insufficient data for any symbol.
        """
        returns_dict: dict[str, pd.Series] = {}
        for symbol in self.symbols:
            bars = data_handler.get_latest_bars(symbol, lookback)
            if len(bars) >= 2:
                closes = bars["close"]
                returns_dict[symbol] = closes.pct_change().dropna()

        if not returns_dict:
            return pd.DataFrame()

        return pd.DataFrame(returns_dict)

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
            strength = self.target_weights.get(symbol, 0.0)
            return self._create_signal(
                timestamp, symbol, SignalType.LONG, strength=strength
            )
        elif crossover == "bearish" and self.current_positions.get(symbol, 0) > 0:
            return self._create_signal(timestamp, symbol, SignalType.EXIT, strength=1.0)

        return None

    def get_target_weights(self) -> dict[str, float]:
        """Return current target weights.

        Returns:
            Copy of the target weights dictionary
        """
        return self.target_weights.copy()

    def get_sma_values(self, symbol: str) -> tuple[float | None, float | None]:
        """Get current SMA values for a symbol.

        Args:
            symbol: The symbol

        Returns:
            Tuple of (short_sma, long_sma), values may be None if not yet calculated
        """
        return (
            self.prev_short_sma.get(symbol),
            self.prev_long_sma.get(symbol),
        )

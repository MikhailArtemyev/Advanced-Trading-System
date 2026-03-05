"""Data handlers for loading and iterating through market data.

CRITICAL: These classes enforce no look-ahead bias by only exposing
data up to the current simulation index.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataHandler(ABC):
    """Abstract base class for data handlers.

    All data handlers must implement these methods to ensure
    consistent behavior and prevent look-ahead bias.
    """

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return last N bars for symbol (NEVER future data).

        Args:
            symbol: The instrument symbol
            n: Number of bars to return

        Returns:
            DataFrame with OHLCV data, up to n rows
        """
        raise NotImplementedError

    @abstractmethod
    def update_bars(self) -> bool:
        """Advance to next bar.

        Returns:
            True if successful, False when data exhausted
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_timestamp(self) -> datetime:
        """Return current simulation timestamp."""
        raise NotImplementedError

    @property
    @abstractmethod
    def continue_backtest(self) -> bool:
        """Check if more data is available."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the data handler to the beginning."""
        raise NotImplementedError


class HistoricalCSVDataHandler(DataHandler):
    """Data handler for CSV files with OHLCV data.

    CRITICAL: This class enforces no look-ahead bias by only
    exposing data up to current_index. The get_latest_bars method
    is the ONLY way strategies should access market data.

    Expected CSV format:
        date,open,high,low,close,volume
        2020-01-02,74.06,75.15,73.79,75.08,135480400
        ...

    Attributes:
        data_path: Directory containing CSV files
        symbols: List of symbols to load
        start_date: Start of date range
        end_date: End of date range
        data: Dictionary mapping symbols to DataFrames
        current_index: Current position in the data
        max_index: Maximum index (length of longest symbol data)
    """

    def __init__(
        self,
        data_path: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> None:
        """Initialize the data handler.

        Args:
            data_path: Path to directory containing CSV files
            symbols: List of symbol names (e.g., ['AAPL', 'MSFT'])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.data_path = data_path
        self.symbols = symbols
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

        # Storage for each symbol's data
        self.data: dict[str, pd.DataFrame] = {}
        self.current_index: int = 0
        self.max_index: int = 0

        # Load all data
        self._load_data()

    def _load_data(self) -> None:
        """Load CSV data for all symbols."""
        for symbol in self.symbols:
            filepath = f"{self.data_path}/{symbol}.csv"

            df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")

            # Filter to date range
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            # Ensure columns are lowercase
            df.columns = df.columns.str.lower()

            # Required columns check
            required = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns for {symbol}: {missing}")

            df = df.sort_index()
            self.data[symbol] = df

            # Track max length
            if len(df) > self.max_index:
                self.max_index = len(df)

        print(
            f"Loaded {len(self.symbols)} symbols, "
            f"{self.max_index} bars from {self.start_date.date()} to {self.end_date.date()}"
        )

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return the last N bars for a symbol.

        CRITICAL: Only returns data up to current_index to prevent
        look-ahead bias. This is the ONLY way strategies should
        access market data.

        Args:
            symbol: The instrument symbol
            n: Number of bars to return

        Returns:
            DataFrame with up to n rows of OHLCV data

        Raises:
            KeyError: If symbol not found in data
        """
        if symbol not in self.data:
            raise KeyError(f"Symbol {symbol} not found in data")

        df = self.data[symbol]

        # Only return data up to current position (inclusive)
        available = df.iloc[: self.current_index + 1]

        # Return last N bars
        return available.tail(n).copy()

    def update_bars(self) -> bool:
        """Advance to next bar.

        Returns:
            True if successful, False if data exhausted
        """
        self.current_index += 1
        return self.current_index < self.max_index

    def get_current_timestamp(self) -> datetime:
        """Return timestamp of current bar.

        Uses the first symbol's index as reference.
        """
        first_symbol = self.symbols[0]
        df = self.data[first_symbol]

        if self.current_index < len(df):
            ts: datetime = df.index[self.current_index].to_pydatetime()
            return ts
        ts = df.index[-1].to_pydatetime()
        return ts

    @property
    def continue_backtest(self) -> bool:
        """Check if more data is available."""
        return self.current_index < self.max_index - 1

    def get_all_symbols(self) -> list[str]:
        """Return list of available symbols."""
        return self.symbols.copy()

    def get_latest_bar_value(self, symbol: str, field: str) -> float:
        """Return a single value from the latest bar.

        Args:
            symbol: The instrument symbol
            field: Column name (e.g., 'close', 'volume')

        Returns:
            The value of the specified field
        """
        bars = self.get_latest_bars(symbol, 1)
        if bars.empty:
            raise ValueError(f"No data available for {symbol}")
        value: float = float(bars[field].iloc[-1])
        return value

    def reset(self) -> None:
        """Reset the data handler to the beginning."""
        self.current_index = 0

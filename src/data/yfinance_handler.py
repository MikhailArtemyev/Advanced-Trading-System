"""YFinance data handler with CSV caching.

Downloads OHLCV data from Yahoo Finance and caches to disk as CSV files.
Delegates bar iteration to HistoricalCSVDataHandler to reuse existing
look-ahead bias prevention logic.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from .data_handler import DataHandler, HistoricalCSVDataHandler


class YFinanceDataHandler(DataHandler):
    """Data handler that downloads from Yahoo Finance with CSV caching.

    Downloads data on first use, caches as CSV files in cache_dir.
    Subsequent runs reuse cached data if the date range is covered.
    All bar iteration is delegated to HistoricalCSVDataHandler.

    Attributes:
        symbols: List of symbols to download
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        cache_dir: Directory for cached CSV files
    """

    def __init__(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        cache_dir: str = "data/cache",
    ) -> None:
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self._ensure_data()

        self._csv_handler = HistoricalCSVDataHandler(
            data_path=self.cache_dir,
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
        )

    def _ensure_data(self) -> None:
        """Download data for symbols that need it."""
        for symbol in self.symbols:
            if not self._cache_is_valid(symbol):
                self._download_symbol(symbol)

    def _cache_is_valid(self, symbol: str) -> bool:
        """Check if cached CSV exists and covers the required date range."""
        csv_path = Path(self.cache_dir) / f"{symbol}.csv"
        if not csv_path.exists():
            return False

        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            if df.empty:
                return False

            cached_start = df["date"].min()
            cached_end = df["date"].max()
            requested_start = pd.Timestamp(self.start_date)
            requested_end = pd.Timestamp(self.end_date)

            return bool(
                cached_start <= requested_start
                and cached_end >= requested_end
            )
        except Exception:
            return False

    def _download_symbol(self, symbol: str) -> None:
        """Download OHLCV data for a single symbol and save as CSV."""
        print(f"Downloading {symbol} from yfinance...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=self.start_date, end=self.end_date)

        if df.empty:
            msg = (
                f"No data returned from yfinance for {symbol} "
                f"({self.start_date} to {self.end_date})"
            )
            raise ValueError(msg)

        df = df.reset_index()
        df.columns = df.columns.str.lower()

        required = ["date", "open", "high", "low", "close", "volume"]
        available = [c for c in required if c in df.columns]
        df = df[available]

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        csv_path = Path(self.cache_dir) / f"{symbol}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Cached {len(df)} bars to {csv_path}")

    # --- Delegate all DataHandler methods to _csv_handler ---

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return last N bars for symbol."""
        return self._csv_handler.get_latest_bars(symbol, n)

    def update_bars(self) -> bool:
        """Advance to next bar."""
        return self._csv_handler.update_bars()

    def get_current_timestamp(self) -> datetime:
        """Return current simulation timestamp."""
        return self._csv_handler.get_current_timestamp()

    @property
    def continue_backtest(self) -> bool:
        """Check if more data is available."""
        return self._csv_handler.continue_backtest

    def reset(self) -> None:
        """Reset the data handler to the beginning."""
        self._csv_handler.reset()

    def get_all_symbols(self) -> list[str]:
        """Return list of all symbols."""
        return self._csv_handler.get_all_symbols()

    def get_latest_bar_value(self, symbol: str, field: str) -> float:
        """Return latest value for a specific field."""
        return self._csv_handler.get_latest_bar_value(symbol, field)

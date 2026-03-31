"""Alpaca intraday data handler with CSV caching.

Downloads OHLCV bars from Alpaca's Market Data REST API and caches
to disk as CSV files. Supports any Alpaca timeframe (1Min, 5Min, etc.).
Delegates bar iteration to HistoricalCSVDataHandler.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .data_handler import DataHandler, HistoricalCSVDataHandler

logger = logging.getLogger(__name__)


class AlpacaBarDataHandler(DataHandler):
    """Data handler that downloads intraday bars from Alpaca REST API.

    Downloads data on first use, caches as CSV files in cache_dir.
    Subsequent runs reuse cached data if the date range is covered.
    All bar iteration is delegated to HistoricalCSVDataHandler.

    Args:
        symbols: List of symbols to download.
        start_date: Start datetime string (YYYY-MM-DD or YYYY-MM-DD HH:MM).
        end_date: End datetime string.
        timeframe: Alpaca timeframe — "1Min", "5Min", "15Min", "1Hour".
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        feed: Data feed tier ("iex" or "sip").
        cache_dir: Directory for cached CSV files.
    """

    def __init__(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        timeframe: str = "5Min",
        api_key: str = "",
        api_secret: str = "",
        feed: str = "iex",
        cache_dir: str = "data/alpaca_cache",
    ) -> None:
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self._api_key = api_key
        self._api_secret = api_secret
        self._feed = feed
        self.cache_dir = cache_dir

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Download data synchronously (run async in event loop)
        self._ensure_data()

        self._csv_handler = HistoricalCSVDataHandler(
            data_path=self.cache_dir,
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
        )

    def _ensure_data(self) -> None:
        """Download data for symbols that need it."""
        symbols_to_download = [
            s for s in self.symbols if not self._cache_is_valid(s)
        ]
        if not symbols_to_download:
            logger.info("All symbols have valid cached data")
            return

        # Run async download in a new event loop
        asyncio.run(self._download_all(symbols_to_download))

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
                cached_start <= requested_start and cached_end >= requested_end
            )
        except Exception:
            logger.warning(
                "Failed to validate cache for %s, will re-download", symbol
            )
            return False

    async def _download_all(self, symbols: list[str]) -> None:
        """Download bars for all symbols from Alpaca."""
        import aiohttp

        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
        }
        base_url = "https://data.alpaca.markets"

        start_dt = pd.Timestamp(self.start_date, tz=UTC)
        end_dt = pd.Timestamp(self.end_date, tz=UTC)

        async with aiohttp.ClientSession(headers=headers) as session:
            for symbol in symbols:
                bars = await self._fetch_symbol(
                    session, base_url, symbol, start_dt, end_dt
                )
                if not bars:
                    raise ValueError(
                        f"No data returned from Alpaca for {symbol} "
                        f"({self.start_date} to {self.end_date})"
                    )
                self._save_csv(symbol, bars)

    async def _fetch_symbol(
        self,
        session: "aiohttp.ClientSession",  # noqa: F821
        base_url: str,
        symbol: str,
        start: "pd.Timestamp",
        end: "pd.Timestamp",
    ) -> list[dict[str, object]]:
        """Fetch all bars for a symbol with pagination."""
        url = f"{base_url}/v2/stocks/{symbol}/bars"
        all_bars: list[dict[str, object]] = []
        page_token: str | None = None

        print(f"Downloading {symbol} {self.timeframe} bars from Alpaca...")

        while True:
            params: dict[str, object] = {
                "timeframe": self.timeframe,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": 10000,
                "feed": self._feed,
                "sort": "asc",
                "adjustment": "all",
            }
            if page_token:
                params["page_token"] = page_token

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise ConnectionError(
                        f"Alpaca API error for {symbol} (HTTP {resp.status}): "
                        f"{text[:200]}"
                    )
                data = await resp.json()
                raw_bars = data.get("bars") or []
                all_bars.extend(raw_bars)

                page_token = data.get("next_page_token")
                if not page_token:
                    break

        print(f"  Downloaded {len(all_bars)} bars for {symbol}")
        return all_bars

    def _save_csv(self, symbol: str, bars: list[dict[str, object]]) -> None:
        """Save bars to CSV in the format HistoricalCSVDataHandler expects."""
        records = []
        for bar in bars:
            ts = bar.get("t", "")
            if isinstance(ts, str):
                ts = ts.replace("Z", "+00:00")
            records.append(
                {
                    "date": ts,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v", 0),
                }
            )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

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

"""Alpaca historical bar fetcher — backfills BarAggregator at startup.

Fetches recent OHLCV bars from Alpaca's REST API so that strategies
(especially ML models) have enough history to generate predictions
from the first live bar.

Requires: ALPACA_API_KEY and ALPACA_API_SECRET.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import aiohttp

from .bar_aggregator import Bar, BarAggregator

logger = logging.getLogger(__name__)

ALPACA_DATA_REST_URL = "https://data.alpaca.markets"


class AlpacaHistoricalClient:
    """Fetches historical bars from Alpaca's Market Data API.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        base_url: Override data API base URL.
        feed: Data feed (``"iex"`` or ``"sip"``).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "",
        feed: str = "iex",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = (base_url or ALPACA_DATA_REST_URL).rstrip("/")
        self._feed = feed
        self._session: aiohttp.ClientSession | None = None

    def _build_headers(self) -> dict[str, str]:
        """Build API authentication headers."""
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
        }

    async def open(self) -> None:
        """Open HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._build_headers(),
            )

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 500,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Bar]:
        """Fetch historical bars for a single symbol.

        Args:
            symbol: Stock symbol (e.g. ``"AAPL"``).
            timeframe: Bar timeframe — ``"1Min"``, ``"5Min"``, ``"1Hour"``,
                ``"1Day"``, etc.
            limit: Maximum number of bars to return (Alpaca max: 10000).
            start: Start time (inclusive). Defaults to ``limit`` bars ago.
            end: End time (exclusive). Defaults to now.

        Returns:
            List of Bar objects, oldest first.
        """
        await self.open()
        assert self._session is not None

        params: dict[str, Any] = {
            "timeframe": timeframe,
            "limit": min(limit, 10000),
            "feed": self._feed,
            "sort": "asc",
            "adjustment": "all",
        }
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()

        url = f"{self._base_url}/v2/stocks/{symbol}/bars"
        bars: list[Bar] = []
        page_token: str | None = None

        while True:
            if page_token:
                params["page_token"] = page_token
            elif "page_token" in params:
                del params["page_token"]

            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(
                            "Alpaca bars request failed (HTTP %d): %s",
                            resp.status,
                            text[:200],
                        )
                        break

                    data = await resp.json()
                    raw_bars = data.get("bars") or []
                    for raw in raw_bars:
                        bar = _parse_bar(symbol, raw)
                        if bar is not None:
                            bars.append(bar)

                    page_token = data.get("next_page_token")
                    if not page_token or len(bars) >= limit:
                        break

            except aiohttp.ClientError as e:
                logger.error("Network error fetching bars for %s: %s", symbol, e)
                break

        return bars[:limit]

    async def fetch_bars_multi(
        self,
        symbols: list[str],
        timeframe: str = "1Min",
        limit: int = 500,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, list[Bar]]:
        """Fetch historical bars for multiple symbols.

        Args:
            symbols: List of stock symbols.
            timeframe: Bar timeframe.
            limit: Maximum bars per symbol.
            start: Start time (inclusive).
            end: End time (exclusive).

        Returns:
            Dict mapping symbol → list of Bar objects.
        """
        result: dict[str, list[Bar]] = {}
        for symbol in symbols:
            result[symbol] = await self.fetch_bars(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                start=start,
                end=end,
            )
        return result


async def backfill_aggregator(
    aggregator: BarAggregator,
    client: AlpacaHistoricalClient,
    symbols: list[str],
    num_bars: int = 500,
    timeframe: str = "1Min",
    end: datetime | None = None,
) -> dict[str, int]:
    """Fetch historical bars and seed a BarAggregator.

    Inserts bars directly into the aggregator's completed bars list
    so that ``get_latest_bars()`` returns data immediately.

    Args:
        aggregator: The BarAggregator to populate.
        client: AlpacaHistoricalClient to fetch from.
        symbols: Symbols to backfill.
        num_bars: Number of historical bars to fetch per symbol.
        timeframe: Alpaca timeframe string.
        end: End time for historical range (defaults to now).

    Returns:
        Dict mapping symbol → number of bars loaded.
    """
    if end is None:
        end = datetime.now(tz=UTC)

    bars_by_symbol = await client.fetch_bars_multi(
        symbols=symbols,
        timeframe=timeframe,
        limit=num_bars,
        end=end,
    )

    counts: dict[str, int] = {}
    for symbol, bars in bars_by_symbol.items():
        if not bars:
            counts[symbol] = 0
            continue

        # Insert directly into completed bars
        if symbol not in aggregator._completed_bars:
            aggregator._completed_bars[symbol] = []
        aggregator._completed_bars[symbol] = bars + aggregator._completed_bars[symbol]

        counts[symbol] = len(bars)
        logger.info(
            "Backfilled %d bars for %s (from %s to %s)",
            len(bars),
            symbol,
            bars[0].timestamp,
            bars[-1].timestamp,
        )

    return counts


# ── Helpers ──────────────────────────────────────────────────────────


def _parse_bar(symbol: str, raw: dict[str, Any]) -> Bar | None:
    """Convert an Alpaca REST bar response to a Bar object."""
    try:
        ts_str = raw.get("t")
        if ts_str is None:
            return None
        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return Bar(
            symbol=symbol,
            timestamp=timestamp,
            open=float(raw["o"]),
            high=float(raw["h"]),
            low=float(raw["l"]),
            close=float(raw["c"]),
            volume=float(raw.get("v", 0)),
            tick_count=int(raw.get("n", 0)),
        )
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("Failed to parse bar for %s: %s", symbol, e)
        return None

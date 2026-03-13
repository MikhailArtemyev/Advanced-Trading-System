"""Tests for AlpacaHistoricalClient and backfill_aggregator.

All tests use mocked HTTP responses. No real API calls.
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.live.alpaca_historical import (
    ALPACA_DATA_REST_URL,
    AlpacaHistoricalClient,
    _parse_bar,
    backfill_aggregator,
)
from src.live.bar_aggregator import Bar, BarAggregator

# ── Helpers ──────────────────────────────────────────────────────────


def _mock_response(status: int = 200, json_data: dict | list | None = None):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=json.dumps(json_data or {}))
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_client(
    api_key: str = "test_key",
    api_secret: str = "test_secret",
    base_url: str = "",
    feed: str = "iex",
) -> AlpacaHistoricalClient:
    """Create an AlpacaHistoricalClient for testing."""
    return AlpacaHistoricalClient(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        feed=feed,
    )


def _raw_bar(
    ts: str = "2025-06-01T14:30:00Z",
    o: float = 150.0,
    h: float = 151.0,
    low: float = 149.0,
    c: float = 150.5,
    v: float = 1000,
    n: int = 50,
) -> dict:
    """Create a raw Alpaca bar dict."""
    return {"t": ts, "o": o, "h": h, "l": low, "c": c, "v": v, "n": n}


def _bars_response(num_bars: int = 3, next_page_token: str | None = None) -> dict:
    """Create a mock Alpaca bars API response."""
    bars = []
    base = datetime(2025, 6, 1, 14, 0, tzinfo=UTC)
    for i in range(num_bars):
        ts = base + timedelta(minutes=i)
        bars.append(
            _raw_bar(
                ts=ts.isoformat(),
                o=150.0 + i,
                h=151.0 + i,
                low=149.0 + i,
                c=150.5 + i,
            )
        )
    result: dict = {"bars": bars}
    if next_page_token:
        result["next_page_token"] = next_page_token
    return result


# ── TestInit ─────────────────────────────────────────────────────────


class TestInit:
    def test_default_url(self):
        client = _make_client()
        assert client._base_url == ALPACA_DATA_REST_URL

    def test_custom_url(self):
        client = _make_client(base_url="https://custom.api.com/")
        assert client._base_url == "https://custom.api.com"

    def test_default_feed(self):
        client = _make_client()
        assert client._feed == "iex"

    def test_custom_feed(self):
        client = _make_client(feed="sip")
        assert client._feed == "sip"


# ── TestOpenClose ────────────────────────────────────────────────────


class TestOpenClose:
    @pytest.mark.asyncio
    async def test_open_creates_session(self):
        client = _make_client()
        with patch("aiohttp.ClientSession") as mock_cls:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_cls.return_value = mock_session
            await client.open()
            assert client._session is not None

    @pytest.mark.asyncio
    async def test_close_closes_session(self):
        client = _make_client()
        mock_session = AsyncMock()
        mock_session.closed = False
        client._session = mock_session
        await client.close()
        mock_session.close.assert_called_once()
        assert client._session is None

    @pytest.mark.asyncio
    async def test_close_when_no_session(self):
        client = _make_client()
        await client.close()  # Should not raise


# ── TestFetchBars ────────────────────────────────────────────────────


class TestFetchBars:
    @pytest.mark.asyncio
    async def test_fetch_returns_bars(self):
        client = _make_client()
        mock_resp = _mock_response(200, _bars_response(3))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        bars = await client.fetch_bars("AAPL", limit=3)
        assert len(bars) == 3
        assert all(isinstance(b, Bar) for b in bars)
        assert all(b.symbol == "AAPL" for b in bars)

    @pytest.mark.asyncio
    async def test_fetch_preserves_ohlcv(self):
        client = _make_client()
        raw = _bars_response(1)
        mock_resp = _mock_response(200, raw)
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        bars = await client.fetch_bars("AAPL", limit=1)
        assert bars[0].open == 150.0
        assert bars[0].high == 151.0
        assert bars[0].low == 149.0
        assert bars[0].close == 150.5
        assert bars[0].volume == 1000

    @pytest.mark.asyncio
    async def test_fetch_respects_limit(self):
        client = _make_client()
        mock_resp = _mock_response(200, _bars_response(10))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        bars = await client.fetch_bars("AAPL", limit=5)
        assert len(bars) == 5

    @pytest.mark.asyncio
    async def test_fetch_empty_response(self):
        client = _make_client()
        mock_resp = _mock_response(200, {"bars": []})
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        bars = await client.fetch_bars("AAPL")
        assert bars == []

    @pytest.mark.asyncio
    async def test_fetch_null_bars(self):
        client = _make_client()
        mock_resp = _mock_response(200, {"bars": None})
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        bars = await client.fetch_bars("AAPL")
        assert bars == []

    @pytest.mark.asyncio
    async def test_fetch_http_error(self):
        client = _make_client()
        mock_resp = _mock_response(403, {"message": "forbidden"})
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        bars = await client.fetch_bars("AAPL")
        assert bars == []

    @pytest.mark.asyncio
    async def test_fetch_passes_params(self):
        client = _make_client(feed="sip")
        mock_resp = _mock_response(200, _bars_response(1))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 6, 1, tzinfo=UTC)
        await client.fetch_bars("AAPL", timeframe="5Min", start=start, end=end)

        call_kwargs = mock_session.get.call_args[1]
        params = call_kwargs["params"]
        assert params["timeframe"] == "5Min"
        assert params["feed"] == "sip"
        assert "start" in params
        assert "end" in params

    @pytest.mark.asyncio
    async def test_fetch_url_includes_symbol(self):
        client = _make_client()
        mock_resp = _mock_response(200, _bars_response(1))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        await client.fetch_bars("TSLA")

        url = mock_session.get.call_args[0][0]
        assert "/TSLA/bars" in url


# ── TestFetchBarsMulti ───────────────────────────────────────────────


class TestFetchBarsMulti:
    @pytest.mark.asyncio
    async def test_fetches_multiple_symbols(self):
        client = _make_client()
        mock_resp = _mock_response(200, _bars_response(2))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        result = await client.fetch_bars_multi(["AAPL", "MSFT"], limit=2)
        assert "AAPL" in result
        assert "MSFT" in result
        assert len(result["AAPL"]) == 2
        assert len(result["MSFT"]) == 2

    @pytest.mark.asyncio
    async def test_empty_symbols_list(self):
        client = _make_client()
        result = await client.fetch_bars_multi([])
        assert result == {}


# ── TestParseBar ─────────────────────────────────────────────────────


class TestParseBar:
    def test_valid_bar(self):
        bar = _parse_bar("AAPL", _raw_bar())
        assert bar is not None
        assert bar.symbol == "AAPL"
        assert bar.open == 150.0
        assert bar.high == 151.0
        assert bar.low == 149.0
        assert bar.close == 150.5
        assert bar.volume == 1000
        assert bar.tick_count == 50

    def test_missing_timestamp(self):
        raw = _raw_bar()
        del raw["t"]
        assert _parse_bar("AAPL", raw) is None

    def test_missing_ohlc_field(self):
        raw = _raw_bar()
        del raw["o"]
        assert _parse_bar("AAPL", raw) is None

    def test_z_suffix_parsed(self):
        bar = _parse_bar("AAPL", _raw_bar(ts="2025-06-01T14:30:00Z"))
        assert bar is not None
        assert bar.timestamp.year == 2025
        assert bar.timestamp.tzinfo is not None

    def test_offset_parsed(self):
        bar = _parse_bar("AAPL", _raw_bar(ts="2025-06-01T14:30:00+00:00"))
        assert bar is not None

    def test_default_volume_zero(self):
        raw = _raw_bar()
        del raw["v"]
        bar = _parse_bar("AAPL", raw)
        assert bar is not None
        assert bar.volume == 0

    def test_default_tick_count_zero(self):
        raw = _raw_bar()
        del raw["n"]
        bar = _parse_bar("AAPL", raw)
        assert bar is not None
        assert bar.tick_count == 0


# ── TestBackfillAggregator ───────────────────────────────────────────


class TestBackfillAggregator:
    @pytest.mark.asyncio
    async def test_backfill_populates_completed_bars(self):
        aggregator = BarAggregator(interval=timedelta(minutes=1))
        client = _make_client()
        mock_resp = _mock_response(200, _bars_response(5))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        counts = await backfill_aggregator(aggregator, client, ["AAPL"], num_bars=5)

        assert counts["AAPL"] == 5
        bars = aggregator.get_completed_bars("AAPL")
        assert len(bars) == 5

    @pytest.mark.asyncio
    async def test_backfill_bars_accessible_via_dataframe(self):
        aggregator = BarAggregator(interval=timedelta(minutes=1))
        client = _make_client()
        mock_resp = _mock_response(200, _bars_response(3))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        await backfill_aggregator(aggregator, client, ["AAPL"], num_bars=3)

        df = aggregator.get_bars_as_dataframe("AAPL", n=3)
        assert len(df) == 3
        assert "close" in df.columns
        assert "open" in df.columns

    @pytest.mark.asyncio
    async def test_backfill_multiple_symbols(self):
        aggregator = BarAggregator(interval=timedelta(minutes=1))
        client = _make_client()
        mock_resp = _mock_response(200, _bars_response(2))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        counts = await backfill_aggregator(
            aggregator, client, ["AAPL", "MSFT"], num_bars=2
        )

        assert counts["AAPL"] == 2
        assert counts["MSFT"] == 2
        assert len(aggregator.get_completed_bars("AAPL")) == 2
        assert len(aggregator.get_completed_bars("MSFT")) == 2

    @pytest.mark.asyncio
    async def test_backfill_preserves_existing_bars(self):
        aggregator = BarAggregator(interval=timedelta(minutes=1))
        # Pre-add a bar
        existing = Bar(
            symbol="AAPL",
            timestamp=datetime(2025, 6, 2, tzinfo=UTC),
            open=160.0,
            high=161.0,
            low=159.0,
            close=160.5,
            volume=500,
            tick_count=10,
        )
        aggregator._completed_bars["AAPL"] = [existing]

        client = _make_client()
        mock_resp = _mock_response(200, _bars_response(2))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        await backfill_aggregator(aggregator, client, ["AAPL"], num_bars=2)

        all_bars = aggregator.get_completed_bars("AAPL")
        # 2 historical + 1 existing
        assert len(all_bars) == 3
        # Historical bars come first (prepended)
        assert all_bars[-1].open == 160.0

    @pytest.mark.asyncio
    async def test_backfill_empty_response(self):
        aggregator = BarAggregator(interval=timedelta(minutes=1))
        client = _make_client()
        mock_resp = _mock_response(200, {"bars": []})
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        counts = await backfill_aggregator(aggregator, client, ["AAPL"], num_bars=100)

        assert counts["AAPL"] == 0
        assert aggregator.get_completed_bars("AAPL") == []

    @pytest.mark.asyncio
    async def test_backfill_returns_count_per_symbol(self):
        aggregator = BarAggregator(interval=timedelta(minutes=1))
        client = _make_client()
        mock_resp = _mock_response(200, _bars_response(4))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        counts = await backfill_aggregator(aggregator, client, ["AAPL"], num_bars=4)

        assert isinstance(counts, dict)
        assert counts["AAPL"] == 4

"""Tests for AlpacaDataFeed — Alpaca real-time market data via WebSocket.

All tests use mocked WebSocket responses. No real API calls.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.live.alpaca_feed import (
    ALPACA_DATA_WS_URL,
    AlpacaDataFeed,
    _parse_bar,
    _parse_quote,
    _parse_timestamp,
    _parse_trade,
)
from src.live.data_feed import FeedStatus, Tick

# ── Helpers ──────────────────────────────────────────────────────────


def _make_feed(
    api_key: str = "test_key",
    api_secret: str = "test_secret",
    feed: str = "iex",
    base_ws_url: str = "",
) -> AlpacaDataFeed:
    """Create an AlpacaDataFeed for testing."""
    return AlpacaDataFeed(
        api_key=api_key,
        api_secret=api_secret,
        feed=feed,
        base_ws_url=base_ws_url,
    )


def _ws_text_msg(data: list | dict) -> MagicMock:
    """Create a mock aiohttp WebSocket TEXT message."""
    import aiohttp

    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.TEXT
    msg.data = json.dumps(data)
    return msg


def _welcome_msg() -> MagicMock:
    """Alpaca welcome message."""
    return _ws_text_msg([{"T": "success", "msg": "connected"}])


def _auth_success_msg() -> MagicMock:
    """Alpaca auth success message."""
    return _ws_text_msg([{"T": "success", "msg": "authenticated"}])


def _auth_error_msg(reason: str = "auth failed") -> MagicMock:
    """Alpaca auth error message."""
    return _ws_text_msg([{"T": "error", "msg": reason}])


def _mock_ws_session(welcome=None, auth_resp=None):
    """Create mock session + ws with scripted receive() responses."""
    if welcome is None:
        welcome = _welcome_msg()
    if auth_resp is None:
        auth_resp = _auth_success_msg()

    mock_ws = AsyncMock()
    mock_ws.closed = False
    mock_ws.close = AsyncMock()
    mock_ws.send_str = AsyncMock()

    receive_responses = [welcome, auth_resp]
    mock_ws.receive = AsyncMock(side_effect=receive_responses)

    mock_session = AsyncMock()
    mock_session.ws_connect = AsyncMock(return_value=mock_ws)
    mock_session.closed = False
    mock_session.close = AsyncMock()

    return mock_session, mock_ws


# ── Tests ────────────────────────────────────────────────────────────


class TestInit:
    def test_default_url(self):
        feed = _make_feed()
        assert ALPACA_DATA_WS_URL in feed._base_ws_url

    def test_custom_url(self):
        feed = _make_feed(base_ws_url="wss://custom.example.com/v2")
        assert feed._base_ws_url == "wss://custom.example.com/v2"

    def test_trailing_slash_stripped(self):
        feed = _make_feed(base_ws_url="wss://custom.example.com/v2/")
        assert not feed._base_ws_url.endswith("/")

    def test_default_feed_tier(self):
        feed = _make_feed()
        assert feed.feed == "iex"

    def test_sip_feed_tier(self):
        feed = _make_feed(feed="sip")
        assert feed.feed == "sip"

    def test_not_authenticated_initially(self):
        feed = _make_feed()
        assert feed.authenticated is False

    def test_disconnected_initially(self):
        feed = _make_feed()
        assert feed.status == FeedStatus.DISCONNECTED


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self):
        feed = _make_feed()
        mock_session, mock_ws = _mock_ws_session()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await feed.connect()

        assert feed.status == FeedStatus.CONNECTED
        assert feed.authenticated is True

        # Cleanup listen task
        feed._should_run = False
        if feed._listen_task:
            feed._listen_task.cancel()
            try:
                await feed._listen_task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_connect_sends_auth(self):
        feed = _make_feed(api_key="my_key", api_secret="my_secret")
        mock_session, mock_ws = _mock_ws_session()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await feed.connect()

        # Find the auth call
        calls = mock_ws.send_str.call_args_list
        assert len(calls) >= 1
        auth_data = json.loads(calls[0][0][0])
        assert auth_data["action"] == "auth"
        assert auth_data["key"] == "my_key"
        assert auth_data["secret"] == "my_secret"

        feed._should_run = False
        if feed._listen_task:
            feed._listen_task.cancel()
            try:
                await feed._listen_task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_connect_auth_failure(self):
        feed = _make_feed()
        mock_session, _ = _mock_ws_session(auth_resp=_auth_error_msg("unauthorized"))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ConnectionError, match="authentication failed"):
                await feed.connect()

        assert feed.authenticated is False

    @pytest.mark.asyncio
    async def test_connect_bad_welcome(self):
        feed = _make_feed()
        bad_welcome = _ws_text_msg([{"T": "error", "msg": "unexpected"}])
        mock_session, _ = _mock_ws_session(welcome=bad_welcome)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ConnectionError, match="Unexpected welcome"):
                await feed.connect()

    @pytest.mark.asyncio
    async def test_ws_url_includes_feed(self):
        feed = _make_feed(feed="sip")
        mock_session, _ = _mock_ws_session()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await feed.connect()

        ws_url = mock_session.ws_connect.call_args[0][0]
        assert ws_url.endswith("/sip")

        feed._should_run = False
        if feed._listen_task:
            feed._listen_task.cancel()
            try:
                await feed._listen_task
            except (asyncio.CancelledError, Exception):
                pass


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect(self):
        feed = _make_feed()
        feed._status = FeedStatus.CONNECTED
        feed._authenticated = True
        feed._session = AsyncMock()
        feed._session.closed = False
        feed._ws = AsyncMock()
        feed._ws.closed = False

        await feed.disconnect()

        assert feed.status == FeedStatus.DISCONNECTED
        assert feed.authenticated is False


class TestSubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_sends_message(self):
        feed = _make_feed()
        feed._ws = AsyncMock()
        feed._ws.closed = False

        await feed.subscribe(["AAPL", "MSFT"])

        call_data = json.loads(feed._ws.send_str.call_args[0][0])
        assert call_data["action"] == "subscribe"
        assert "AAPL" in call_data["trades"]
        assert "MSFT" in call_data["quotes"]

    @pytest.mark.asyncio
    async def test_subscribe_updates_symbols(self):
        feed = _make_feed()
        feed._ws = AsyncMock()
        feed._ws.closed = False

        await feed.subscribe(["AAPL"])
        assert "AAPL" in feed.subscribed_symbols

    @pytest.mark.asyncio
    async def test_subscribe_no_ws_does_not_error(self):
        feed = _make_feed()
        await feed.subscribe(["AAPL"])
        assert "AAPL" in feed.subscribed_symbols


class TestUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe_sends_message(self):
        feed = _make_feed()
        feed._ws = AsyncMock()
        feed._ws.closed = False
        feed._symbols = {"AAPL", "MSFT"}

        await feed.unsubscribe(["AAPL"])

        call_data = json.loads(feed._ws.send_str.call_args[0][0])
        assert call_data["action"] == "unsubscribe"
        assert "AAPL" in call_data["trades"]

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_symbol(self):
        feed = _make_feed()
        feed._ws = AsyncMock()
        feed._ws.closed = False
        feed._symbols = {"AAPL", "MSFT"}

        await feed.unsubscribe(["AAPL"])
        assert "AAPL" not in feed.subscribed_symbols
        assert "MSFT" in feed.subscribed_symbols


class TestHandleMessage:
    def test_trade_message_emits_tick(self):
        feed = _make_feed()
        ticks: list[Tick] = []
        feed.add_listener(ticks.append)

        trade_msg = json.dumps(
            [
                {
                    "T": "t",
                    "S": "AAPL",
                    "p": 150.25,
                    "s": 100,
                    "t": "2025-06-01T14:30:00Z",
                }
            ]
        )
        feed._handle_message(trade_msg)

        assert len(ticks) == 1
        assert ticks[0].symbol == "AAPL"
        assert ticks[0].price == 150.25
        assert ticks[0].volume == 100.0

    def test_quote_message_emits_tick(self):
        feed = _make_feed()
        ticks: list[Tick] = []
        feed.add_listener(ticks.append)

        quote_msg = json.dumps(
            [
                {
                    "T": "q",
                    "S": "MSFT",
                    "bp": 380.00,
                    "ap": 380.50,
                    "t": "2025-06-01T14:30:00Z",
                }
            ]
        )
        feed._handle_message(quote_msg)

        assert len(ticks) == 1
        assert ticks[0].symbol == "MSFT"
        assert ticks[0].price == pytest.approx(380.25)
        assert ticks[0].bid == 380.00
        assert ticks[0].ask == 380.50

    def test_bar_message_emits_tick(self):
        feed = _make_feed()
        ticks: list[Tick] = []
        feed.add_listener(ticks.append)

        bar_msg = json.dumps(
            [
                {
                    "T": "b",
                    "S": "TSLA",
                    "o": 200.0,
                    "h": 205.0,
                    "l": 199.0,
                    "c": 203.0,
                    "v": 5000,
                    "t": "2025-06-01T14:30:00Z",
                }
            ]
        )
        feed._handle_message(bar_msg)

        assert len(ticks) == 1
        assert ticks[0].symbol == "TSLA"
        assert ticks[0].price == 203.0
        assert ticks[0].volume == 5000.0

    def test_multiple_messages_in_array(self):
        feed = _make_feed()
        ticks: list[Tick] = []
        feed.add_listener(ticks.append)

        multi_msg = json.dumps(
            [
                {"T": "t", "S": "AAPL", "p": 150.0, "s": 50},
                {"T": "t", "S": "MSFT", "p": 380.0, "s": 30},
            ]
        )
        feed._handle_message(multi_msg)

        assert len(ticks) == 2
        symbols = {t.symbol for t in ticks}
        assert symbols == {"AAPL", "MSFT"}

    def test_control_messages_ignored(self):
        feed = _make_feed()
        ticks: list[Tick] = []
        feed.add_listener(ticks.append)

        control_msg = json.dumps([{"T": "success", "msg": "authenticated"}])
        feed._handle_message(control_msg)

        assert len(ticks) == 0

    def test_invalid_json_handled(self):
        feed = _make_feed()
        ticks: list[Tick] = []
        feed.add_listener(ticks.append)

        feed._handle_message("not json at all")
        assert len(ticks) == 0

    def test_missing_fields_skipped(self):
        feed = _make_feed()
        ticks: list[Tick] = []
        feed.add_listener(ticks.append)

        # Trade missing price
        msg = json.dumps([{"T": "t", "S": "AAPL"}])
        feed._handle_message(msg)
        assert len(ticks) == 0


class TestParseTrade:
    def test_valid_trade(self):
        tick = _parse_trade(
            {"T": "t", "S": "AAPL", "p": 150.0, "s": 100, "t": "2025-06-01T14:30:00Z"}
        )
        assert tick is not None
        assert tick.symbol == "AAPL"
        assert tick.price == 150.0
        assert tick.volume == 100.0

    def test_missing_symbol(self):
        assert _parse_trade({"T": "t", "p": 150.0}) is None

    def test_missing_price(self):
        assert _parse_trade({"T": "t", "S": "AAPL"}) is None


class TestParseQuote:
    def test_valid_quote(self):
        tick = _parse_quote(
            {
                "T": "q",
                "S": "MSFT",
                "bp": 380.0,
                "ap": 381.0,
                "t": "2025-06-01T14:30:00Z",
            }
        )
        assert tick is not None
        assert tick.symbol == "MSFT"
        assert tick.price == pytest.approx(380.5)
        assert tick.bid == 380.0
        assert tick.ask == 381.0

    def test_missing_bid(self):
        assert _parse_quote({"T": "q", "S": "MSFT", "ap": 381.0}) is None

    def test_missing_ask(self):
        assert _parse_quote({"T": "q", "S": "MSFT", "bp": 380.0}) is None

    def test_missing_symbol(self):
        assert _parse_quote({"T": "q", "bp": 380.0, "ap": 381.0}) is None


class TestParseBar:
    def test_valid_bar(self):
        tick = _parse_bar(
            {"T": "b", "S": "TSLA", "c": 200.0, "v": 5000, "t": "2025-06-01T14:30:00Z"}
        )
        assert tick is not None
        assert tick.symbol == "TSLA"
        assert tick.price == 200.0
        assert tick.volume == 5000.0

    def test_missing_close(self):
        assert _parse_bar({"T": "b", "S": "TSLA"}) is None

    def test_missing_symbol(self):
        assert _parse_bar({"T": "b", "c": 200.0}) is None


class TestParseTimestamp:
    def test_rfc3339_with_z(self):
        ts = _parse_timestamp("2025-06-01T14:30:00Z")
        assert ts.year == 2025
        assert ts.month == 6
        assert ts.tzinfo is not None

    def test_rfc3339_with_offset(self):
        ts = _parse_timestamp("2025-06-01T14:30:00+00:00")
        assert ts.year == 2025

    def test_none_returns_now(self):
        ts = _parse_timestamp(None)
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_invalid_returns_now(self):
        ts = _parse_timestamp("not-a-date")
        assert isinstance(ts, datetime)


class TestConfigFeedType:
    def test_alpaca_is_valid_feed_type(self):
        from src.config import LiveDataConfig

        config = LiveDataConfig(feed_type="alpaca")
        assert config.feed_type == "alpaca"

    def test_websocket_still_valid(self):
        from src.config import LiveDataConfig

        config = LiveDataConfig(feed_type="websocket")
        assert config.feed_type == "websocket"

    def test_invalid_feed_type_rejected(self):
        from src.config import LiveDataConfig

        with pytest.raises(ValueError):
            LiveDataConfig(feed_type="bogus")

"""Tests for live data feed abstraction and WebSocket implementation."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.live.data_feed import FeedStatus, LiveDataFeed, Tick, WebSocketDataFeed

# ---------------------------------------------------------------------------
# Tick tests
# ---------------------------------------------------------------------------


class TestTick:
    def test_tick_creation(self):
        tick = Tick(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 15, 10, 30, 0),
            price=150.25,
            volume=100.0,
        )
        assert tick.symbol == "AAPL"
        assert tick.price == 150.25
        assert tick.volume == 100.0

    def test_tick_defaults(self):
        tick = Tick(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 15),
            price=150.0,
        )
        assert tick.volume == 0.0
        assert tick.bid is None
        assert tick.ask is None

    def test_tick_with_bid_ask(self):
        tick = Tick(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 15),
            price=150.0,
            bid=149.95,
            ask=150.05,
        )
        assert tick.bid == 149.95
        assert tick.ask == 150.05

    def test_tick_is_frozen(self):
        tick = Tick(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 15),
            price=150.0,
        )
        with pytest.raises(AttributeError):
            tick.price = 151.0  # type: ignore[misc]

    def test_tick_equality(self):
        ts = datetime(2025, 1, 15, 10, 30)
        t1 = Tick(symbol="AAPL", timestamp=ts, price=150.0)
        t2 = Tick(symbol="AAPL", timestamp=ts, price=150.0)
        assert t1 == t2

    def test_tick_inequality(self):
        ts = datetime(2025, 1, 15, 10, 30)
        t1 = Tick(symbol="AAPL", timestamp=ts, price=150.0)
        t2 = Tick(symbol="AAPL", timestamp=ts, price=151.0)
        assert t1 != t2


# ---------------------------------------------------------------------------
# FeedStatus tests
# ---------------------------------------------------------------------------


class TestFeedStatus:
    def test_all_statuses_exist(self):
        assert FeedStatus.DISCONNECTED
        assert FeedStatus.CONNECTING
        assert FeedStatus.CONNECTED
        assert FeedStatus.RECONNECTING
        assert FeedStatus.ERROR

    def test_statuses_are_distinct(self):
        statuses = list(FeedStatus)
        assert len(statuses) == 5
        assert len(set(statuses)) == 5


# ---------------------------------------------------------------------------
# LiveDataFeed ABC tests (using a concrete subclass)
# ---------------------------------------------------------------------------


class MockDataFeed(LiveDataFeed):
    """Minimal concrete implementation for testing the ABC."""

    async def connect(self) -> None:
        self._status = FeedStatus.CONNECTED

    async def disconnect(self) -> None:
        self._status = FeedStatus.DISCONNECTED

    async def subscribe(self, symbols: list[str]) -> None:
        self._symbols.update(symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        self._symbols -= set(symbols)

    def emit_tick(self, tick: Tick) -> None:
        """Helper to simulate a tick arriving."""
        self._on_tick(tick)


class TestLiveDataFeed:
    def test_initial_status(self):
        feed = MockDataFeed()
        assert feed.status == FeedStatus.DISCONNECTED

    def test_initial_no_symbols(self):
        feed = MockDataFeed()
        assert feed.subscribed_symbols == set()

    def test_subscribed_symbols_is_copy(self):
        feed = MockDataFeed()
        symbols = feed.subscribed_symbols
        symbols.add("AAPL")
        assert "AAPL" not in feed.subscribed_symbols

    def test_add_tick_callback(self):
        feed = MockDataFeed()
        received: list[Tick] = []
        feed.add_tick_callback(received.append)

        tick = Tick(symbol="AAPL", timestamp=datetime(2025, 1, 15), price=150.0)
        feed.emit_tick(tick)

        assert len(received) == 1
        assert received[0] == tick

    def test_multiple_tick_callbacks(self):
        feed = MockDataFeed()
        received_a: list[Tick] = []
        received_b: list[Tick] = []
        feed.add_tick_callback(received_a.append)
        feed.add_tick_callback(received_b.append)

        tick = Tick(symbol="AAPL", timestamp=datetime(2025, 1, 15), price=150.0)
        feed.emit_tick(tick)

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_remove_tick_callback(self):
        feed = MockDataFeed()
        received: list[Tick] = []
        feed.add_tick_callback(received.append)
        feed.remove_tick_callback(received.append)

        tick = Tick(symbol="AAPL", timestamp=datetime(2025, 1, 15), price=150.0)
        feed.emit_tick(tick)

        assert len(received) == 0

    def test_remove_nonexistent_tick_callback_raises(self):
        feed = MockDataFeed()
        with pytest.raises(ValueError):
            feed.remove_tick_callback(lambda t: None)

    @pytest.mark.asyncio
    async def test_connect_sets_status(self):
        feed = MockDataFeed()
        await feed.connect()
        assert feed.status == FeedStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_sets_status(self):
        feed = MockDataFeed()
        await feed.connect()
        await feed.disconnect()
        assert feed.status == FeedStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_subscribe_tracks_symbols(self):
        feed = MockDataFeed()
        await feed.subscribe(["AAPL", "MSFT"])
        assert feed.subscribed_symbols == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_symbols(self):
        feed = MockDataFeed()
        await feed.subscribe(["AAPL", "MSFT", "GOOGL"])
        await feed.unsubscribe(["MSFT"])
        assert feed.subscribed_symbols == {"AAPL", "GOOGL"}

    def test_on_tick_with_no_callbacks(self):
        feed = MockDataFeed()
        tick = Tick(symbol="AAPL", timestamp=datetime(2025, 1, 15), price=150.0)
        # Should not raise
        feed.emit_tick(tick)

    def test_dispatch_multiple_ticks(self):
        feed = MockDataFeed()
        received: list[Tick] = []
        feed.add_tick_callback(received.append)

        for i in range(5):
            tick = Tick(
                symbol="AAPL",
                timestamp=datetime(2025, 1, 15, 10, i),
                price=150.0 + i,
            )
            feed.emit_tick(tick)

        assert len(received) == 5
        assert received[0].price == 150.0
        assert received[4].price == 154.0


# ---------------------------------------------------------------------------
# WebSocketDataFeed tests
# ---------------------------------------------------------------------------


class TestWebSocketDataFeed:
    def _make_parser(self) -> MagicMock:
        """Create a mock parse_message function."""
        return MagicMock(return_value=[])

    def test_initial_state(self):
        feed = WebSocketDataFeed(
            url="ws://localhost:8080",
            parse_message=self._make_parser(),
        )
        assert feed.status == FeedStatus.DISCONNECTED
        assert feed.subscribed_symbols == set()
        assert feed._reconnect_count == 0
        assert feed._should_run is False

    def test_custom_reconnect_params(self):
        feed = WebSocketDataFeed(
            url="ws://localhost:8080",
            parse_message=self._make_parser(),
            max_reconnect_attempts=5,
            initial_reconnect_delay=2.0,
        )
        assert feed._max_reconnect_attempts == 5
        assert feed._initial_reconnect_delay == 2.0

    @pytest.mark.asyncio
    async def test_connect_success(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(url="ws://test", parse_message=parser)

        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_session = AsyncMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        mock_session.closed = False

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Prevent listen loop from running forever
            with patch.object(feed, "_listen_loop", new_callable=AsyncMock):
                await feed.connect()

        assert feed.status == FeedStatus.CONNECTED
        assert feed._should_run is True
        assert feed._reconnect_count == 0

    @pytest.mark.asyncio
    async def test_connect_failure_sets_error(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(url="ws://bad", parse_message=parser)

        mock_session = AsyncMock()
        mock_session.ws_connect = AsyncMock(side_effect=ConnectionError("refused"))
        mock_session.closed = False

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ConnectionError):
                await feed.connect()

        assert feed.status == FeedStatus.ERROR

    @pytest.mark.asyncio
    async def test_disconnect(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(url="ws://test", parse_message=parser)

        # Simulate connected state
        feed._should_run = True
        feed._status = FeedStatus.CONNECTED

        mock_ws = AsyncMock()
        mock_ws.closed = False
        feed._ws = mock_ws

        mock_session = AsyncMock()
        mock_session.closed = False
        feed._session = mock_session

        await feed.disconnect()

        assert feed.status == FeedStatus.DISCONNECTED
        assert feed._should_run is False
        mock_ws.close.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_already_disconnected(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(url="ws://test", parse_message=parser)
        # Should not raise
        await feed.disconnect()
        assert feed.status == FeedStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_subscribe_sends_message(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(url="ws://test", parse_message=parser)

        mock_ws = AsyncMock()
        mock_ws.closed = False
        feed._ws = mock_ws

        await feed.subscribe(["AAPL", "MSFT"])

        assert feed.subscribed_symbols == {"AAPL", "MSFT"}
        mock_ws.send_str.assert_called_once()
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["action"] == "subscribe"
        assert set(sent["symbols"]) == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_subscribe_no_ws(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(url="ws://test", parse_message=parser)

        # No ws connection — should still track symbols but not send
        await feed.subscribe(["AAPL"])
        assert feed.subscribed_symbols == {"AAPL"}

    @pytest.mark.asyncio
    async def test_unsubscribe_sends_message(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(url="ws://test", parse_message=parser)

        mock_ws = AsyncMock()
        mock_ws.closed = False
        feed._ws = mock_ws
        feed._symbols = {"AAPL", "MSFT"}

        await feed.unsubscribe(["MSFT"])

        assert feed.subscribed_symbols == {"AAPL"}
        mock_ws.send_str.assert_called_once()
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["action"] == "unsubscribe"

    @pytest.mark.asyncio
    async def test_reconnect_increments_count(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(
            url="ws://test",
            parse_message=parser,
            max_reconnect_attempts=3,
            initial_reconnect_delay=0.01,
        )
        feed._should_run = True

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.ws_connect = AsyncMock(side_effect=ConnectionError("fail"))
        feed._session = mock_session

        await feed._reconnect()

        assert feed._reconnect_count == 1
        assert feed.status == FeedStatus.ERROR  # connection failed

    @pytest.mark.asyncio
    async def test_reconnect_max_attempts(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(
            url="ws://test",
            parse_message=parser,
            max_reconnect_attempts=3,
            initial_reconnect_delay=0.01,
        )
        feed._should_run = True
        feed._reconnect_count = 3  # Already at max

        await feed._reconnect()

        assert feed.status == FeedStatus.ERROR
        assert feed._should_run is False

    @pytest.mark.asyncio
    async def test_reconnect_exponential_backoff_delay(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(
            url="ws://test",
            parse_message=parser,
            max_reconnect_attempts=10,
            initial_reconnect_delay=1.0,
        )
        feed._should_run = True
        feed._reconnect_count = 2  # Next delay = 1.0 * 2^2 = 4.0

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        feed._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await feed._reconnect()

        mock_sleep.assert_called_once_with(4.0)

    @pytest.mark.asyncio
    async def test_reconnect_delay_capped_at_60(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(
            url="ws://test",
            parse_message=parser,
            max_reconnect_attempts=20,
            initial_reconnect_delay=1.0,
        )
        feed._should_run = True
        feed._reconnect_count = 10  # delay = 1.0 * 2^10 = 1024 → capped at 60

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        feed._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await feed._reconnect()

        mock_sleep.assert_called_once_with(60.0)

    @pytest.mark.asyncio
    async def test_reconnect_resubscribes(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(
            url="ws://test",
            parse_message=parser,
            initial_reconnect_delay=0.01,
        )
        feed._should_run = True
        feed._symbols = {"AAPL", "MSFT"}

        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        feed._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await feed._reconnect()

        assert feed.status == FeedStatus.CONNECTED
        assert feed._reconnect_count == 0
        # subscribe was called — send_str invoked
        mock_ws.send_str.assert_called()

    @pytest.mark.asyncio
    async def test_reconnect_stops_when_not_running(self):
        parser = self._make_parser()
        feed = WebSocketDataFeed(
            url="ws://test",
            parse_message=parser,
        )
        feed._should_run = False

        await feed._reconnect()

        # Should return immediately without changing status
        assert feed.status == FeedStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_listen_loop_dispatches_ticks(self):
        tick = Tick(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 15, 10, 30),
            price=150.0,
        )
        parser = MagicMock(return_value=[tick])
        feed = WebSocketDataFeed(url="ws://test", parse_message=parser)
        feed._should_run = True

        received: list[Tick] = []
        feed.add_tick_callback(received.append)

        import aiohttp

        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.TEXT
        mock_msg.data = '{"price": 150}'

        mock_ws = AsyncMock()
        mock_ws.closed = False
        call_count = 0

        async def receive_side_effect(timeout: float = 30.0) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_msg
            # Stop after first message
            feed._should_run = False
            return mock_msg

        mock_ws.receive = receive_side_effect
        feed._ws = mock_ws

        await feed._listen_loop()

        assert len(received) >= 1
        assert received[0] == tick
        parser.assert_called_with('{"price": 150}')

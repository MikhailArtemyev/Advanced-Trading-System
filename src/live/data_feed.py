"""Live market data feed abstraction.

Provides an ABC for streaming market data and a WebSocket implementation.
Data feeds emit raw ticks/quotes that the BarAggregator converts to OHLCV bars.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class FeedStatus(Enum):
    """Connection status of a data feed."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()


@dataclass(frozen=True)
class Tick:
    """A single price update from the market.

    Attributes:
        symbol: Instrument identifier.
        timestamp: Exchange timestamp of the tick.
        price: Last trade price.
        volume: Trade volume (0 for quotes without volume).
        bid: Best bid price (None if unavailable).
        ask: Best ask price (None if unavailable).
    """

    symbol: str
    timestamp: datetime
    price: float
    volume: float = 0.0
    bid: float | None = None
    ask: float | None = None


class LiveDataFeed(ABC):
    """Abstract base class for live market data feeds.

    Subclasses must implement connect/disconnect and subscribe/unsubscribe.
    When ticks arrive, subclasses call _on_tick() to notify listeners.
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[Tick], None]] = []
        self._status: FeedStatus = FeedStatus.DISCONNECTED
        self._symbols: set[str] = set()

    @property
    def status(self) -> FeedStatus:
        """Current connection status."""
        return self._status

    @property
    def subscribed_symbols(self) -> set[str]:
        """Set of currently subscribed symbols (copy)."""
        return self._symbols.copy()

    def add_listener(self, callback: Callable[[Tick], None]) -> None:
        """Register a callback invoked on each tick."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[Tick], None]) -> None:
        """Unregister a tick callback."""
        self._listeners.remove(callback)

    def _on_tick(self, tick: Tick) -> None:
        """Dispatch a tick to all registered listeners."""
        for listener in self._listeners:
            listener(tick)

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        ...

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to live data for the given symbols."""
        ...

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from live data for the given symbols."""
        ...


class WebSocketDataFeed(LiveDataFeed):
    """WebSocket-based live data feed.

    Connects to a WebSocket endpoint, subscribes to symbols,
    and parses incoming messages into Tick objects. Includes
    automatic reconnection with exponential backoff.

    Args:
        url: WebSocket server URL.
        parse_message: Callable that converts a raw WebSocket message
            (str or bytes) into a list of Tick objects. Allows plugging
            in exchange-specific message formats.
        max_reconnect_attempts: Max consecutive reconnection attempts
            before giving up (0 = unlimited).
        initial_reconnect_delay: Starting delay in seconds for
            exponential backoff reconnection.
    """

    def __init__(
        self,
        url: str,
        parse_message: Callable[[str | bytes], list[Tick]],
        max_reconnect_attempts: int = 10,
        initial_reconnect_delay: float = 1.0,
    ) -> None:
        super().__init__()
        self._url = url
        self._parse_message = parse_message
        self._max_reconnect_attempts = max_reconnect_attempts
        self._initial_reconnect_delay = initial_reconnect_delay
        self._ws: Any = None  # aiohttp.ClientWebSocketResponse
        self._session: Any = None  # aiohttp.ClientSession
        self._listen_task: asyncio.Task[None] | None = None
        self._reconnect_count = 0
        self._should_run = False

    async def connect(self) -> None:
        """Connect to the WebSocket server and start listening."""
        import aiohttp

        self._should_run = True
        self._status = FeedStatus.CONNECTING
        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(self._url)
            self._status = FeedStatus.CONNECTED
            self._reconnect_count = 0
            self._listen_task = asyncio.create_task(self._listen_loop())
            logger.info("WebSocket connected to %s", self._url)
        except Exception:
            self._status = FeedStatus.ERROR
            await self._session.close()
            logger.exception("Failed to connect to %s", self._url)
            raise

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._should_run = False
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        self._status = FeedStatus.DISCONNECTED
        logger.info("WebSocket disconnected from %s", self._url)

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols via WebSocket.

        Sends a JSON subscribe message.
        """
        import json

        self._symbols.update(symbols)
        if self._ws and not self._ws.closed:
            msg = json.dumps({"action": "subscribe", "symbols": symbols})
            await self._ws.send_str(msg)
            logger.info("Subscribed to %s", symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols via WebSocket."""
        import json

        self._symbols -= set(symbols)
        if self._ws and not self._ws.closed:
            msg = json.dumps({"action": "unsubscribe", "symbols": symbols})
            await self._ws.send_str(msg)
            logger.info("Unsubscribed from %s", symbols)

    async def _listen_loop(self) -> None:
        """Read messages from WebSocket and dispatch ticks."""
        import aiohttp

        while self._should_run:
            try:
                if self._ws is None or self._ws.closed:
                    await self._reconnect()
                    continue

                msg = await self._ws.receive(timeout=30.0)

                if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    ticks = self._parse_message(msg.data)
                    for tick in ticks:
                        self._on_tick(tick)
                elif msg.type == aiohttp.WSMsgType.PING:
                    await self._ws.pong()
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    logger.warning("WebSocket closed by server")
                    await self._reconnect()
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error: %s", self._ws.exception())
                    await self._reconnect()

            except TimeoutError:
                logger.debug("WebSocket receive timeout, continuing")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in WebSocket listen loop")
                await self._reconnect()

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        if not self._should_run:
            return

        self._status = FeedStatus.RECONNECTING

        if (
            self._max_reconnect_attempts > 0
            and self._reconnect_count >= self._max_reconnect_attempts
        ):
            self._status = FeedStatus.ERROR
            logger.error(
                "Max reconnect attempts (%d) reached",
                self._max_reconnect_attempts,
            )
            self._should_run = False
            return

        delay = self._initial_reconnect_delay * (2**self._reconnect_count)
        delay = min(delay, 60.0)  # cap at 60 seconds
        self._reconnect_count += 1
        logger.info(
            "Reconnecting in %.1fs (attempt %d)",
            delay,
            self._reconnect_count,
        )
        await asyncio.sleep(delay)

        try:
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._session is None or self._session.closed:
                import aiohttp

                self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(self._url)

            self._status = FeedStatus.CONNECTED
            self._reconnect_count = 0

            # Re-subscribe to all symbols
            if self._symbols:
                await self.subscribe(list(self._symbols))

            logger.info("Reconnected successfully")
        except Exception:
            logger.exception("Reconnection failed")
            self._status = FeedStatus.ERROR

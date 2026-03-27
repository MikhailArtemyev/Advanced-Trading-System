"""Alpaca real-time market data feed via WebSocket.

Connects to Alpaca's market data streaming API (v2) and converts
trade/quote/bar messages into Tick objects for the BarAggregator.

Supports both IEX (free) and SIP (premium) data feeds.
Requires: ALPACA_API_KEY and ALPACA_API_SECRET.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import aiohttp

from .data_feed import FeedStatus, LiveDataFeed, Tick

logger = logging.getLogger(__name__)

ALPACA_DATA_WS_URL = "wss://stream.data.alpaca.markets/v2"


class AlpacaDataFeed(LiveDataFeed):
    """Real-time market data feed from Alpaca.

    Connects to Alpaca's WebSocket streaming API, authenticates,
    subscribes to trade/quote updates, and emits Tick objects.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        feed: Data feed tier — ``"iex"`` (free) or ``"sip"`` (premium).
        base_ws_url: Override WebSocket URL (for testing).
        max_reconnect_attempts: Max reconnect tries before giving up.
        initial_reconnect_delay: Base delay (seconds) for exponential backoff.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        feed: str = "iex",
        base_ws_url: str = "",
        max_reconnect_attempts: int = 10,
        initial_reconnect_delay: float = 1.0,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._api_secret = api_secret
        self._feed = feed
        self._base_ws_url = (base_ws_url.rstrip("/") or ALPACA_DATA_WS_URL).rstrip("/")
        self._max_reconnect_attempts = max_reconnect_attempts
        self._initial_reconnect_delay = initial_reconnect_delay

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._listen_task: asyncio.Task[None] | None = None
        self._reconnect_count = 0
        self._should_run = False
        self._authenticated = False
        self._last_data_time: datetime | None = None
        self._stale_timeout = 90.0  # seconds with no data before reconnect

    @property
    def authenticated(self) -> bool:
        """Whether the feed has been authenticated with Alpaca."""
        return self._authenticated

    @property
    def feed(self) -> str:
        """Data feed tier (iex or sip)."""
        return self._feed

    async def connect(self) -> None:
        """Connect to Alpaca WebSocket and authenticate."""
        self._should_run = True
        self._status = FeedStatus.CONNECTING
        self._session = aiohttp.ClientSession()

        ws_url = f"{self._base_ws_url}/{self._feed}"
        try:
            self._ws = await self._session.ws_connect(ws_url)

            # Read welcome message
            welcome = await self._ws.receive(timeout=10.0)
            if welcome.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(welcome.data)
                if not _is_message_type(data, "success", "connected"):
                    raise ConnectionError(f"Unexpected welcome message: {welcome.data}")

            # Authenticate
            auth_msg = json.dumps(
                {
                    "action": "auth",
                    "key": self._api_key,
                    "secret": self._api_secret,
                }
            )
            await self._ws.send_str(auth_msg)

            auth_resp = await self._ws.receive(timeout=10.0)
            if auth_resp.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(auth_resp.data)
                if _is_message_type(data, "success", "authenticated"):
                    self._authenticated = True
                    self._status = FeedStatus.CONNECTED
                    self._reconnect_count = 0
                    logger.info("Authenticated with Alpaca data feed (%s)", self._feed)
                elif _is_message_type(data, "error"):
                    error_msg = _get_message_field(data, "msg")
                    raise ConnectionError(f"Alpaca authentication failed: {error_msg}")
                else:
                    raise ConnectionError(f"Unexpected auth response: {auth_resp.data}")
            else:
                raise ConnectionError(f"Non-text auth response: type={auth_resp.type}")

            # Start listen loop
            self._listen_task = asyncio.create_task(self._listen_loop())
            logger.info("Alpaca data feed connected (%s)", self._feed)

        except ConnectionError:
            self._status = FeedStatus.ERROR
            await self._cleanup()
            raise
        except Exception as e:
            self._status = FeedStatus.ERROR
            await self._cleanup()
            raise ConnectionError(f"Failed to connect to Alpaca data: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from Alpaca WebSocket."""
        self._should_run = False

        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        await self._cleanup()
        self._status = FeedStatus.DISCONNECTED
        self._authenticated = False
        logger.info("Alpaca data feed disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to trade and quote updates for symbols."""
        self._symbols.update(symbols)
        if self._ws and not self._ws.closed:
            msg = json.dumps(
                {
                    "action": "subscribe",
                    "trades": symbols,
                    "quotes": symbols,
                }
            )
            await self._ws.send_str(msg)
            logger.info("Subscribed to %s", symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from trade and quote updates."""
        self._symbols -= set(symbols)
        if self._ws and not self._ws.closed:
            msg = json.dumps(
                {
                    "action": "unsubscribe",
                    "trades": symbols,
                    "quotes": symbols,
                }
            )
            await self._ws.send_str(msg)
            logger.info("Unsubscribed from %s", symbols)

    async def _listen_loop(self) -> None:
        """Read and dispatch WebSocket messages."""
        while self._should_run:
            try:
                if self._ws is None or self._ws.closed:
                    logger.warning("WebSocket is closed/None, reconnecting")
                    await self._reconnect()
                    continue

                msg = await self._ws.receive(timeout=30.0)

                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.PING:
                    await self._ws.pong()
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    logger.warning("Alpaca WebSocket closed by server")
                    await self._reconnect()
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("Alpaca WebSocket error: %s", self._ws.exception())
                    await self._reconnect()

            except TimeoutError:
                # Check for stale connection — no data for too long
                if self._last_data_time is not None:
                    elapsed = (
                        datetime.now(tz=UTC) - self._last_data_time
                    ).total_seconds()
                    if elapsed > self._stale_timeout:
                        logger.warning("No data for %.0fs, forcing reconnect", elapsed)
                        await self._reconnect()
                        continue
                logger.debug("Alpaca WebSocket receive timeout, continuing")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in Alpaca listen loop")
                await self._reconnect()

    def _handle_message(self, raw: str) -> None:
        """Parse a raw Alpaca WebSocket message and emit ticks.

        Alpaca sends arrays of message objects, e.g.:
        ``[{"T": "t", "S": "AAPL", "p": 150.25, "s": 100, "t": "..."}]``
        """
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Alpaca message: %s", raw[:200])
            return

        if not isinstance(messages, list):
            messages = [messages]

        for item in messages:
            if not isinstance(item, dict):
                continue

            msg_type = item.get("T", "")

            if msg_type == "t":
                tick = _parse_trade(item)
                if tick is not None:
                    self._last_data_time = datetime.now(tz=UTC)
                    self._on_tick(tick)
            elif msg_type == "q":
                tick = _parse_quote(item)
                if tick is not None:
                    self._last_data_time = datetime.now(tz=UTC)
                    self._on_tick(tick)
            elif msg_type == "b":
                tick = _parse_bar(item)
                if tick is not None:
                    self._last_data_time = datetime.now(tz=UTC)
                    self._on_tick(tick)
            elif msg_type in ("success", "error", "subscription"):
                logger.debug("Alpaca control message: %s", msg_type)

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        if not self._should_run:
            return

        self._status = FeedStatus.RECONNECTING
        self._authenticated = False

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
        delay = min(delay, 60.0)
        self._reconnect_count += 1
        logger.info(
            "Reconnecting Alpaca data in %.1fs (attempt %d)",
            delay,
            self._reconnect_count,
        )
        await asyncio.sleep(delay)

        try:
            await self._cleanup()

            self._session = aiohttp.ClientSession()
            ws_url = f"{self._base_ws_url}/{self._feed}"
            self._ws = await self._session.ws_connect(ws_url)

            # Re-authenticate
            welcome = await self._ws.receive(timeout=10.0)
            if welcome.type != aiohttp.WSMsgType.TEXT:
                raise ConnectionError("No welcome on reconnect")

            auth_msg = json.dumps(
                {
                    "action": "auth",
                    "key": self._api_key,
                    "secret": self._api_secret,
                }
            )
            await self._ws.send_str(auth_msg)

            auth_resp = await self._ws.receive(timeout=10.0)
            if auth_resp.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(auth_resp.data)
                if _is_message_type(data, "success", "authenticated"):
                    self._authenticated = True
                else:
                    raise ConnectionError("Re-authentication failed")

            self._status = FeedStatus.CONNECTED
            self._reconnect_count = 0
            self._last_data_time = datetime.now(tz=UTC)

            # Re-subscribe
            if self._symbols:
                await self.subscribe(list(self._symbols))

            logger.info(
                "Alpaca data feed reconnected (subscribed to %d symbols)",
                len(self._symbols),
            )
        except Exception:
            logger.exception("Alpaca reconnection failed")
            self._status = FeedStatus.ERROR

    async def _cleanup(self) -> None:
        """Close WebSocket and HTTP session."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        self._ws = None
        self._session = None


# ── Message helpers ──────────────────────────────────────────────────


def _parse_timestamp(ts: str | None) -> datetime:
    """Parse an RFC-3339 timestamp from Alpaca."""
    if ts is None:
        return datetime.now(tz=UTC)
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.now(tz=UTC)


def _parse_trade(item: dict[str, Any]) -> Tick | None:
    """Convert an Alpaca trade message to a Tick."""
    symbol = item.get("S")
    price = item.get("p")
    if symbol is None or price is None:
        return None
    return Tick(
        symbol=str(symbol),
        timestamp=_parse_timestamp(item.get("t")),
        price=float(price),
        volume=float(item.get("s", 0)),
    )


def _parse_quote(item: dict[str, Any]) -> Tick | None:
    """Convert an Alpaca quote message to a Tick."""
    symbol = item.get("S")
    bid = item.get("bp")
    ask = item.get("ap")
    if symbol is None or bid is None or ask is None:
        return None
    mid = (float(bid) + float(ask)) / 2.0
    return Tick(
        symbol=str(symbol),
        timestamp=_parse_timestamp(item.get("t")),
        price=mid,
        volume=0.0,
        bid=float(bid),
        ask=float(ask),
    )


def _parse_bar(item: dict[str, Any]) -> Tick | None:
    """Convert an Alpaca bar message to a Tick (uses close price)."""
    symbol = item.get("S")
    close = item.get("c")
    if symbol is None or close is None:
        return None
    return Tick(
        symbol=str(symbol),
        timestamp=_parse_timestamp(item.get("t")),
        price=float(close),
        volume=float(item.get("v", 0)),
    )


def _is_message_type(
    data: list[Any] | dict[str, Any],
    msg_type: str,
    msg_value: str = "",
) -> bool:
    """Check if Alpaca message array contains a message of given type.

    Alpaca sends messages as arrays: ``[{"T": "success", "msg": "connected"}]``
    """
    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("T") == msg_type:
            if msg_value and item.get("msg") != msg_value:
                continue
            return True
    return False


def _get_message_field(
    data: list[Any] | dict[str, Any],
    field: str,
) -> str:
    """Extract a field from the first message in an Alpaca message array."""
    items = data if isinstance(data, list) else [data]
    for item in items:
        if isinstance(item, dict) and field in item:
            return str(item[field])
    return ""

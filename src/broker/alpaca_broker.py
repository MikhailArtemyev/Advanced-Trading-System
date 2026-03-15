"""Alpaca broker adapter — connects to Alpaca Trade API.

Implements BrokerAdapter ABC for paper and live trading via Alpaca's
REST API. Supports market and limit orders, position queries, and
account information.

Requires: ALPACA_API_KEY and ALPACA_API_SECRET (env vars or config).
"""

import logging
from datetime import datetime
from typing import Any

import aiohttp

from .base_broker import BrokerAdapter, BrokerOrder, OrderStatus

logger = logging.getLogger(__name__)

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"

# Map Alpaca order status strings to our OrderStatus enum
_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.ACCEPTED,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.ACCEPTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.EXPIRED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "replaced": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    "stopped": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "calculated": OrderStatus.ACCEPTED,
}


class AlpacaBroker(BrokerAdapter):
    """Broker adapter for Alpaca Trade API.

    Supports both paper and live trading endpoints. All methods are
    async and use aiohttp for non-blocking HTTP requests.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        base_url: Override API base URL. If empty, auto-selects based on paper_mode.
        paper_mode: Use paper trading endpoint (default True).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "",
        paper_mode: bool = True,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._paper_mode = paper_mode

        if base_url:
            self._base_url = base_url.rstrip("/")
        else:
            self._base_url = ALPACA_PAPER_URL if paper_mode else ALPACA_LIVE_URL

        self._session: aiohttp.ClientSession | None = None
        self._connected = False

        # Track orders by our internal ID → broker order ID
        self._order_map: dict[str, str] = {}
        # Track orders by our internal ID → BrokerOrder
        self._orders: dict[str, BrokerOrder] = {}

    @property
    def connected(self) -> bool:
        """Whether the broker is connected."""
        return self._connected

    @property
    def paper_mode(self) -> bool:
        """Whether using paper trading."""
        return self._paper_mode

    def _build_headers(self) -> dict[str, str]:
        """Build API authentication headers."""
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "Content-Type": "application/json",
        }

    async def connect(self) -> None:
        """Connect to Alpaca and verify credentials."""
        self._session = aiohttp.ClientSession(
            headers=self._build_headers(),
        )

        # Verify credentials by fetching account info
        try:
            async with self._session.get(f"{self._base_url}/v2/account") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._connected = True
                    logger.info(
                        "Connected to Alpaca (%s). Account: %s, Equity: $%s",
                        "paper" if self._paper_mode else "LIVE",
                        data.get("account_number", "unknown"),
                        data.get("equity", "?"),
                    )
                elif resp.status == 401:
                    raise ConnectionError("Invalid Alpaca API credentials")
                elif resp.status == 403:
                    raise ConnectionError(
                        "Alpaca API access forbidden — check account status"
                    )
                else:
                    text = await resp.text()
                    raise ConnectionError(
                        f"Alpaca connect failed (HTTP {resp.status}): {text}"
                    )
        except aiohttp.ClientError as e:
            await self._cleanup_session()
            raise ConnectionError(f"Failed to connect to Alpaca: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        await self._cleanup_session()
        self._connected = False
        logger.info("Disconnected from Alpaca")

    async def _cleanup_session(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        """Submit an order to Alpaca.

        Args:
            order: BrokerOrder to submit.

        Returns:
            Updated BrokerOrder with broker_order_id and status.
        """
        self._ensure_connected()

        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.lower(),
            "type": self._map_order_type(order.order_type),
            "time_in_force": "day",
        }

        if order.order_type == "LIMIT" and order.limit_price is not None:
            payload["limit_price"] = str(order.limit_price)

        assert self._session is not None
        try:
            async with self._session.post(
                f"{self._base_url}/v2/orders",
                json=payload,
            ) as resp:
                data = await resp.json()

                if resp.status in (200, 201):
                    order.broker_order_id = data["id"]
                    order.submitted_at = datetime.now()
                    order.status = _STATUS_MAP.get(
                        data.get("status", "new"), OrderStatus.SUBMITTED
                    )
                    self._order_map[order.order_id] = data["id"]
                    self._orders[order.order_id] = order
                    logger.info(
                        "Order submitted to Alpaca: %s %s %d %s → %s",
                        order.side,
                        order.symbol,
                        order.quantity,
                        order.order_type,
                        order.broker_order_id,
                    )
                elif resp.status == 403:
                    order.status = OrderStatus.REJECTED
                    logger.warning(
                        "Order rejected by Alpaca (forbidden): %s",
                        data.get("message", ""),
                    )
                elif resp.status == 422:
                    order.status = OrderStatus.REJECTED
                    logger.warning(
                        "Order rejected by Alpaca (unprocessable): %s",
                        data.get("message", ""),
                    )
                else:
                    order.status = OrderStatus.REJECTED
                    logger.warning(
                        "Order failed (HTTP %d): %s",
                        resp.status,
                        data.get("message", str(data)),
                    )
        except aiohttp.ClientError as e:
            order.status = OrderStatus.REJECTED
            logger.error("Network error submitting order: %s", e)

        return order

    async def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel an order on Alpaca.

        Args:
            order_id: Our internal order ID.

        Returns:
            Updated BrokerOrder with cancelled status.
        """
        self._ensure_connected()

        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Unknown order: {order_id}")

        broker_id = self._order_map.get(order_id)
        if broker_id is None:
            raise ValueError(f"No broker order ID for: {order_id}")

        assert self._session is not None
        try:
            async with self._session.delete(
                f"{self._base_url}/v2/orders/{broker_id}"
            ) as resp:
                if resp.status in (200, 204):
                    order.status = OrderStatus.CANCELLED
                    logger.info("Order cancelled: %s", order_id)
                elif resp.status == 404:
                    logger.warning("Order not found on Alpaca: %s", broker_id)
                elif resp.status == 422:
                    logger.warning(
                        "Order cannot be cancelled (may be filled): %s",
                        order_id,
                    )
                else:
                    text = await resp.text()
                    logger.warning("Cancel failed (HTTP %d): %s", resp.status, text)
        except aiohttp.ClientError as e:
            logger.error("Network error cancelling order: %s", e)

        return order

    async def get_order_status(self, order_id: str) -> BrokerOrder:
        """Query order status from Alpaca.

        Args:
            order_id: Our internal order ID.

        Returns:
            Updated BrokerOrder with current status.
        """
        self._ensure_connected()

        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Unknown order: {order_id}")

        broker_id = self._order_map.get(order_id)
        if broker_id is None:
            raise ValueError(f"No broker order ID for: {order_id}")

        assert self._session is not None
        try:
            async with self._session.get(
                f"{self._base_url}/v2/orders/{broker_id}"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    order.status = _STATUS_MAP.get(data.get("status", ""), order.status)
                    filled_qty = data.get("filled_qty")
                    if filled_qty is not None:
                        order.filled_quantity = int(filled_qty)
                    filled_price = data.get("filled_avg_price")
                    if filled_price is not None:
                        order.filled_avg_price = float(filled_price)
                    if order.status == OrderStatus.FILLED and order.filled_at is None:
                        filled_at = data.get("filled_at")
                        if filled_at:
                            order.filled_at = datetime.fromisoformat(
                                filled_at.replace("Z", "+00:00")
                            )
                        else:
                            order.filled_at = datetime.now()
                else:
                    logger.warning("Failed to get order status (HTTP %d)", resp.status)
        except aiohttp.ClientError as e:
            logger.error("Network error querying order: %s", e)

        return order

    async def get_positions(self) -> dict[str, dict[str, float]]:
        """Get current positions from Alpaca.

        Returns:
            Dict of {symbol: {quantity, avg_cost, market_value}}.
        """
        self._ensure_connected()

        assert self._session is not None
        positions: dict[str, dict[str, float]] = {}
        try:
            async with self._session.get(f"{self._base_url}/v2/positions") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for pos in data:
                        positions[pos["symbol"]] = {
                            "quantity": float(pos.get("qty", 0)),
                            "avg_cost": float(pos.get("avg_entry_price", 0)),
                            "market_value": float(pos.get("market_value", 0)),
                        }
                else:
                    logger.warning("Failed to get positions (HTTP %d)", resp.status)
        except aiohttp.ClientError as e:
            logger.error("Network error fetching positions: %s", e)

        return positions

    async def get_account_info(self) -> dict[str, float]:
        """Get account info from Alpaca.

        Returns:
            Dict with cash, equity, buying_power.
        """
        self._ensure_connected()

        assert self._session is not None
        try:
            async with self._session.get(f"{self._base_url}/v2/account") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "cash": float(data.get("cash", 0)),
                        "equity": float(data.get("equity", 0)),
                        "buying_power": float(data.get("buying_power", 0)),
                    }
                else:
                    logger.warning("Failed to get account info (HTTP %d)", resp.status)
        except aiohttp.ClientError as e:
            logger.error("Network error fetching account: %s", e)

        return {"cash": 0.0, "equity": 0.0, "buying_power": 0.0}

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca — call connect() first")

    @staticmethod
    def _map_order_type(order_type: str) -> str:
        """Map internal order type to Alpaca format."""
        mapping = {
            "MARKET": "market",
            "LIMIT": "limit",
            "STOP": "stop",
            "STOP_LIMIT": "stop_limit",
        }
        return mapping.get(order_type.upper(), "market")

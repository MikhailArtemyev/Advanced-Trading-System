"""Paper trading broker — simulates order execution without real money.

Fills orders based on the last known price from the data handler,
applying configurable slippage and commission models. Supports market
and limit orders with realistic fill simulation.
"""

import logging
import uuid
from collections.abc import Callable
from datetime import datetime

from src.data.data_handler import DataHandler

from .base_broker import BrokerAdapter, BrokerFill, BrokerOrder, OrderStatus

logger = logging.getLogger(__name__)


class PaperBroker(BrokerAdapter):
    """Simulated broker for paper trading.

    Args:
        initial_capital: Starting cash balance.
        commission_pct: Commission as a fraction of trade value.
        slippage_pct: Slippage as a fraction of price.
        fill_delay_seconds: Simulated delay before fills (0 for instant).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        fill_delay_seconds: float = 0.0,
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._commission_pct = commission_pct
        self._slippage_pct = slippage_pct
        self._fill_delay_seconds = fill_delay_seconds
        self._connected = False

        self._orders: dict[str, BrokerOrder] = {}
        self._fills: list[BrokerFill] = []
        self._positions: dict[str, dict[str, float]] = {}

        self._fill_callbacks: list[Callable[[BrokerFill], None]] = []
        self._data_handler: DataHandler | None = None

    @property
    def connected(self) -> bool:
        """Whether the broker is connected."""
        return self._connected

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def fills(self) -> list[BrokerFill]:
        """All fills executed so far."""
        return list(self._fills)

    def set_data_handler(self, data_handler: DataHandler) -> None:
        """Set the data handler for price lookups."""
        self._data_handler = data_handler

    def set_price(self, symbol: str, price: float) -> None:
        """Manually set a price for a symbol (useful for testing).

        Creates a minimal data handler stub that returns this price.
        """
        if not hasattr(self, "_manual_prices"):
            self._manual_prices: dict[str, float] = {}
        self._manual_prices[symbol] = price

    def add_fill_callback(self, callback: Callable[[BrokerFill], None]) -> None:
        """Register a callback invoked when an order is filled."""
        self._fill_callbacks.append(callback)

    async def connect(self) -> None:
        """Connect the paper broker."""
        self._connected = True
        logger.info("Paper broker connected (capital=%.2f)", self._cash)

    async def disconnect(self) -> None:
        """Disconnect the paper broker."""
        self._connected = False
        logger.info("Paper broker disconnected")

    async def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        """Submit an order for simulated execution.

        Market orders fill immediately. Limit orders fill if the
        price condition is met at the current price.
        """
        if not self._connected:
            order.status = OrderStatus.REJECTED
            return order

        order.order_id = order.order_id or str(uuid.uuid4())
        order.broker_order_id = f"PAPER-{order.order_id[:8]}"
        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED
        self._orders[order.order_id] = order

        logger.info(
            "Order submitted: %s %s %d %s @ %s",
            order.side,
            order.symbol,
            order.quantity,
            order.order_type,
            order.limit_price or "MARKET",
        )

        if order.order_type == "MARKET":
            self._execute_fill(order)
        elif order.order_type == "LIMIT":
            self._try_limit_fill(order)

        return order

    async def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel a pending or submitted order."""
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Unknown order: {order_id}")
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return order
        order.status = OrderStatus.CANCELLED
        logger.info("Order cancelled: %s", order_id)
        return order

    async def get_order_status(self, order_id: str) -> BrokerOrder:
        """Query the current status of an order."""
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Unknown order: {order_id}")
        return order

    async def get_positions(self) -> dict[str, dict[str, float]]:
        """Get current positions."""
        return {k: dict(v) for k, v in self._positions.items()}

    async def get_account_info(self) -> dict[str, float]:
        """Get account information."""
        total_value = sum(
            pos.get("market_value", 0.0) for pos in self._positions.values()
        )
        equity = self._cash + total_value
        return {
            "cash": self._cash,
            "equity": equity,
            "buying_power": self._cash,
            "initial_capital": self._initial_capital,
        }

    def _execute_fill(self, order: BrokerOrder) -> None:
        """Fill a market order at the current price with slippage."""
        price = self._get_current_price(order.symbol)
        if price is None:
            order.status = OrderStatus.REJECTED
            logger.warning("No price available for %s — order rejected", order.symbol)
            return

        # Apply slippage
        if order.side == "BUY":
            fill_price = price * (1 + self._slippage_pct)
        else:
            fill_price = price * (1 - self._slippage_pct)

        commission = abs(fill_price * order.quantity * self._commission_pct)

        order.filled_quantity = order.quantity
        order.filled_avg_price = fill_price
        order.filled_at = datetime.now()
        order.status = OrderStatus.FILLED

        # Update cash
        if order.side == "BUY":
            self._cash -= fill_price * order.quantity + commission
        else:
            self._cash += fill_price * order.quantity - commission

        self._update_position(order.symbol, order.side, order.quantity, fill_price)

        fill = BrokerFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=order.filled_at,
        )
        self._fills.append(fill)

        for cb in self._fill_callbacks:
            cb(fill)

        logger.info(
            "Order filled: %s %s %d @ %.4f (commission=%.2f)",
            order.side,
            order.symbol,
            order.quantity,
            fill_price,
            commission,
        )

    def _try_limit_fill(self, order: BrokerOrder) -> None:
        """Check if a limit order can be filled at the current price."""
        price = self._get_current_price(order.symbol)
        if price is None:
            return  # Leave as submitted — will check on next tick

        can_fill = False
        if order.side == "BUY" and order.limit_price is not None:
            can_fill = price <= order.limit_price
        elif order.side == "SELL" and order.limit_price is not None:
            can_fill = price >= order.limit_price

        if can_fill and order.limit_price is not None:
            fill_price = order.limit_price
            commission = abs(fill_price * order.quantity * self._commission_pct)

            order.filled_quantity = order.quantity
            order.filled_avg_price = fill_price
            order.filled_at = datetime.now()
            order.status = OrderStatus.FILLED

            if order.side == "BUY":
                self._cash -= fill_price * order.quantity + commission
            else:
                self._cash += fill_price * order.quantity - commission

            self._update_position(order.symbol, order.side, order.quantity, fill_price)

            fill = BrokerFill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                commission=commission,
                timestamp=order.filled_at,
            )
            self._fills.append(fill)
            for cb in self._fill_callbacks:
                cb(fill)

    def _get_current_price(self, symbol: str) -> float | None:
        """Get the latest price from the data handler or manual prices."""
        # Check manual prices first (for testing)
        if hasattr(self, "_manual_prices") and symbol in self._manual_prices:
            return self._manual_prices[symbol]

        if self._data_handler is None:
            return None
        try:
            bars = self._data_handler.get_latest_bars(symbol, n=1)
            if bars.empty:
                return None
            return float(bars["close"].iloc[-1])
        except (ValueError, KeyError):
            return None

    def _update_position(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
    ) -> None:
        """Update position tracking after a fill."""
        if symbol not in self._positions:
            self._positions[symbol] = {
                "quantity": 0.0,
                "avg_cost": 0.0,
                "market_value": 0.0,
            }
        pos = self._positions[symbol]
        if side == "BUY":
            total_cost = pos["avg_cost"] * pos["quantity"] + price * quantity
            pos["quantity"] += quantity
            if pos["quantity"] > 0:
                pos["avg_cost"] = total_cost / pos["quantity"]
        else:
            pos["quantity"] -= quantity
            if pos["quantity"] <= 0:
                pos["quantity"] = 0.0
                pos["avg_cost"] = 0.0
        pos["market_value"] = pos["quantity"] * price

"""Order lifecycle management with state machine.

Tracks all orders through their lifecycle, validates state transitions,
and provides query interfaces for order status. Bridges between the
event-driven engine (OrderEvent) and the broker adapter (BrokerOrder).
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from src.events.event import FillEvent, OrderEvent, OrderSide

from .base_broker import BrokerAdapter, BrokerFill, BrokerOrder, OrderStatus

logger = logging.getLogger(__name__)

# Valid state transitions
VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING: {OrderStatus.SUBMITTED, OrderStatus.REJECTED},
    OrderStatus.SUBMITTED: {
        OrderStatus.ACCEPTED,
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.ACCEPTED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.REJECTED: set(),
    OrderStatus.EXPIRED: set(),
}


class OrderManager:
    """Manages the full order lifecycle.

    Converts OrderEvents from the engine into BrokerOrders,
    submits them via the BrokerAdapter, tracks state transitions,
    and converts BrokerFills back into FillEvents for the engine.

    Args:
        broker: The broker adapter to submit orders through.
    """

    def __init__(self, broker: BrokerAdapter) -> None:
        self._broker = broker
        self._orders: dict[str, BrokerOrder] = {}
        self._order_history: list[dict[str, Any]] = []
        self._pending_cancels: set[str] = set()

    @property
    def active_orders(self) -> list[BrokerOrder]:
        """Return all non-terminal orders."""
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        return [o for o in self._orders.values() if o.status not in terminal]

    @property
    def all_orders(self) -> dict[str, BrokerOrder]:
        """Return all orders (active and historical)."""
        return dict(self._orders)

    async def submit_order_event(self, order_event: OrderEvent) -> BrokerOrder:
        """Convert an OrderEvent to a BrokerOrder and submit it.

        This is the main entry point from the engine's event loop.
        """
        order_id = str(uuid.uuid4())
        broker_order = BrokerOrder(
            order_id=order_id,
            symbol=order_event.symbol,
            side=order_event.side.name,
            order_type=order_event.order_type.name,
            quantity=order_event.quantity,
            limit_price=order_event.limit_price,
        )

        self._orders[order_id] = broker_order
        self._record_transition(order_id, None, OrderStatus.PENDING)

        updated = await self._broker.submit_order(broker_order)
        self._orders[order_id] = updated

        return updated

    async def cancel_order(self, order_id: str) -> BrokerOrder:
        """Request cancellation of an order."""
        if order_id not in self._orders:
            raise ValueError(f"Unknown order: {order_id}")

        order = self._orders[order_id]
        old_status = order.status
        updated = await self._broker.cancel_order(order_id)

        if updated.status != old_status:
            self._record_transition(order_id, old_status, updated.status)

        self._orders[order_id] = updated
        return updated

    def on_fill(self, fill: BrokerFill) -> FillEvent:
        """Convert a BrokerFill into a FillEvent for the engine.

        Called by the paper trading engine when it receives a fill
        notification from the broker.
        """
        side = OrderSide.BUY if fill.side == "BUY" else OrderSide.SELL

        return FillEvent(
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            side=side,
            quantity=fill.quantity,
            fill_price=fill.price,
            commission=fill.commission,
        )

    def get_order(self, order_id: str) -> BrokerOrder | None:
        """Look up an order by ID."""
        return self._orders.get(order_id)

    def get_orders_for_symbol(self, symbol: str) -> list[BrokerOrder]:
        """Return all orders for a given symbol."""
        return [o for o in self._orders.values() if o.symbol == symbol]

    def get_order_history(self) -> list[dict[str, Any]]:
        """Return the full state transition history."""
        return list(self._order_history)

    def reset(self) -> None:
        """Clear all order state."""
        self._orders.clear()
        self._order_history.clear()
        self._pending_cancels.clear()

    def _record_transition(
        self,
        order_id: str,
        from_status: OrderStatus | None,
        to_status: OrderStatus,
    ) -> None:
        """Record an order state transition for audit trail."""
        self._order_history.append(
            {
                "order_id": order_id,
                "from": from_status.name if from_status else None,
                "to": to_status.name,
                "timestamp": datetime.now(),
            }
        )
        logger.debug(
            "Order %s: %s → %s",
            order_id[:8],
            from_status.name if from_status else "NEW",
            to_status.name,
        )

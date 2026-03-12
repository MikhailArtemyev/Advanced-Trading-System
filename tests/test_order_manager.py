"""Tests for order lifecycle manager."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from src.broker.base_broker import BrokerFill, BrokerOrder, OrderStatus
from src.broker.order_manager import VALID_TRANSITIONS, OrderManager
from src.events.event import FillEvent, OrderEvent, OrderSide, OrderType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order_event(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=100,
    limit_price=None,
):
    return OrderEvent(
        timestamp=datetime(2025, 1, 15, 10, 30),
        symbol=symbol,
        order_type=order_type,
        side=side,
        quantity=quantity,
        limit_price=limit_price,
    )


def _make_broker_fill(
    order_id="test-001",
    symbol="AAPL",
    side="BUY",
    quantity=100,
    price=150.0,
    commission=0.15,
):
    return BrokerFill(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        commission=commission,
        timestamp=datetime(2025, 1, 15, 10, 30, 1),
    )


def _mock_broker():
    """Create a mock broker that returns submitted orders as filled."""
    broker = AsyncMock()

    async def submit_side_effect(order):
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_avg_price = 150.0
        order.filled_at = datetime.now()
        return order

    broker.submit_order = AsyncMock(side_effect=submit_side_effect)

    async def cancel_side_effect(order_id):
        order = BrokerOrder(
            order_id=order_id,
            symbol="AAPL",
            side="BUY",
            order_type="MARKET",
            quantity=100,
            status=OrderStatus.CANCELLED,
        )
        return order

    broker.cancel_order = AsyncMock(side_effect=cancel_side_effect)

    return broker


# ---------------------------------------------------------------------------
# VALID_TRANSITIONS
# ---------------------------------------------------------------------------


class TestValidTransitions:
    def test_terminal_states_have_empty_sets(self):
        terminals = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        for status in terminals:
            assert VALID_TRANSITIONS[status] == set()

    def test_pending_can_go_to_submitted(self):
        assert OrderStatus.SUBMITTED in VALID_TRANSITIONS[OrderStatus.PENDING]

    def test_pending_can_go_to_rejected(self):
        assert OrderStatus.REJECTED in VALID_TRANSITIONS[OrderStatus.PENDING]

    def test_submitted_can_go_to_filled(self):
        assert OrderStatus.FILLED in VALID_TRANSITIONS[OrderStatus.SUBMITTED]

    def test_submitted_can_go_to_cancelled(self):
        assert OrderStatus.CANCELLED in VALID_TRANSITIONS[OrderStatus.SUBMITTED]

    def test_accepted_can_go_to_partially_filled(self):
        assert OrderStatus.PARTIALLY_FILLED in VALID_TRANSITIONS[OrderStatus.ACCEPTED]

    def test_partially_filled_can_go_to_filled(self):
        assert OrderStatus.FILLED in VALID_TRANSITIONS[OrderStatus.PARTIALLY_FILLED]

    def test_all_statuses_in_transitions(self):
        for status in OrderStatus:
            assert status in VALID_TRANSITIONS


# ---------------------------------------------------------------------------
# submit_order_event
# ---------------------------------------------------------------------------


class TestSubmitOrderEvent:
    @pytest.mark.asyncio
    async def test_converts_order_event_to_broker_order(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        event = _make_order_event()

        result = await manager.submit_order_event(event)

        assert result.symbol == "AAPL"
        assert result.side == "BUY"
        assert result.order_type == "MARKET"
        assert result.quantity == 100

    @pytest.mark.asyncio
    async def test_assigns_uuid(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        event = _make_order_event()

        result = await manager.submit_order_event(event)
        assert result.order_id is not None
        assert len(result.order_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_calls_broker_submit(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        event = _make_order_event()

        await manager.submit_order_event(event)
        broker.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_tracks_order(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        event = _make_order_event()

        result = await manager.submit_order_event(event)
        assert manager.get_order(result.order_id) is not None

    @pytest.mark.asyncio
    async def test_limit_order_with_price(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        event = _make_order_event(order_type=OrderType.LIMIT, limit_price=155.0)

        result = await manager.submit_order_event(event)
        assert result.limit_price == 155.0

    @pytest.mark.asyncio
    async def test_sell_side(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        event = _make_order_event(side=OrderSide.SELL)

        result = await manager.submit_order_event(event)
        assert result.side == "SELL"

    @pytest.mark.asyncio
    async def test_records_transition_to_pending(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        event = _make_order_event()

        await manager.submit_order_event(event)

        history = manager.get_order_history()
        assert len(history) >= 1
        assert history[0]["from"] is None
        assert history[0]["to"] == "PENDING"


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_calls_broker(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        event = _make_order_event()
        result = await manager.submit_order_event(event)

        await manager.cancel_order(result.order_id)
        broker.cancel_order.assert_called_once_with(result.order_id)

    @pytest.mark.asyncio
    async def test_cancel_updates_status(self):
        broker = _mock_broker()

        # Override: make submit return SUBMITTED (not FILLED) so cancel is meaningful
        async def submit_submitted(order):
            order.status = OrderStatus.SUBMITTED
            return order

        broker.submit_order = AsyncMock(side_effect=submit_submitted)
        manager = OrderManager(broker)
        event = _make_order_event()
        result = await manager.submit_order_event(event)

        cancelled = await manager.cancel_order(result.order_id)
        assert cancelled.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_unknown_raises(self):
        broker = _mock_broker()
        manager = OrderManager(broker)

        with pytest.raises(ValueError, match="Unknown order"):
            await manager.cancel_order("nonexistent")

    @pytest.mark.asyncio
    async def test_cancel_records_transition(self):
        broker = _mock_broker()

        async def submit_submitted(order):
            order.status = OrderStatus.SUBMITTED
            return order

        broker.submit_order = AsyncMock(side_effect=submit_submitted)
        manager = OrderManager(broker)
        event = _make_order_event()
        result = await manager.submit_order_event(event)

        await manager.cancel_order(result.order_id)

        history = manager.get_order_history()
        cancel_records = [h for h in history if h["to"] == "CANCELLED"]
        assert len(cancel_records) == 1


# ---------------------------------------------------------------------------
# on_fill
# ---------------------------------------------------------------------------


class TestOnFill:
    def test_converts_buy_fill_to_fill_event(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        fill = _make_broker_fill(side="BUY")

        event = manager.on_fill(fill)

        assert isinstance(event, FillEvent)
        assert event.symbol == "AAPL"
        assert event.side == OrderSide.BUY
        assert event.quantity == 100
        assert event.fill_price == 150.0
        assert event.commission == 0.15

    def test_converts_sell_fill_to_fill_event(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        fill = _make_broker_fill(side="SELL")

        event = manager.on_fill(fill)
        assert event.side == OrderSide.SELL

    def test_preserves_timestamp(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        fill = _make_broker_fill()

        event = manager.on_fill(fill)
        assert event.timestamp == datetime(2025, 1, 15, 10, 30, 1)


# ---------------------------------------------------------------------------
# active_orders / all_orders
# ---------------------------------------------------------------------------


class TestOrderQueries:
    @pytest.mark.asyncio
    async def test_active_orders_empty_initially(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        assert manager.active_orders == []

    @pytest.mark.asyncio
    async def test_filled_orders_not_in_active(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        event = _make_order_event()
        await manager.submit_order_event(event)

        # Order is filled → not active
        assert manager.active_orders == []

    @pytest.mark.asyncio
    async def test_submitted_orders_are_active(self):
        broker = _mock_broker()

        async def submit_submitted(order):
            order.status = OrderStatus.SUBMITTED
            return order

        broker.submit_order = AsyncMock(side_effect=submit_submitted)
        manager = OrderManager(broker)
        event = _make_order_event()
        await manager.submit_order_event(event)

        assert len(manager.active_orders) == 1

    @pytest.mark.asyncio
    async def test_all_orders_includes_everything(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        await manager.submit_order_event(_make_order_event())
        await manager.submit_order_event(_make_order_event(symbol="MSFT"))

        assert len(manager.all_orders) == 2

    @pytest.mark.asyncio
    async def test_get_orders_for_symbol(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        await manager.submit_order_event(_make_order_event(symbol="AAPL"))
        await manager.submit_order_event(_make_order_event(symbol="MSFT"))
        await manager.submit_order_event(_make_order_event(symbol="AAPL"))

        aapl_orders = manager.get_orders_for_symbol("AAPL")
        assert len(aapl_orders) == 2
        assert all(o.symbol == "AAPL" for o in aapl_orders)

    @pytest.mark.asyncio
    async def test_get_order_returns_none_for_unknown(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        assert manager.get_order("nonexistent") is None


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    @pytest.mark.asyncio
    async def test_clears_orders(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        await manager.submit_order_event(_make_order_event())

        manager.reset()

        assert len(manager.all_orders) == 0
        assert len(manager.active_orders) == 0

    @pytest.mark.asyncio
    async def test_clears_history(self):
        broker = _mock_broker()
        manager = OrderManager(broker)
        await manager.submit_order_event(_make_order_event())

        manager.reset()

        assert len(manager.get_order_history()) == 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.asyncio
    async def test_submit_fill_to_fill_event(self):
        """Full flow: OrderEvent → BrokerOrder → submit → fill → FillEvent."""
        from src.broker.paper_broker import PaperBroker

        broker = PaperBroker(
            initial_capital=100_000.0,
            commission_pct=0.001,
            slippage_pct=0.0,
        )
        await broker.connect()
        broker.set_price("AAPL", 150.0)

        manager = OrderManager(broker)

        fills_received: list[BrokerFill] = []
        broker.add_fill_callback(fills_received.append)

        event = _make_order_event()
        result = await manager.submit_order_event(event)

        assert result.status == OrderStatus.FILLED
        assert len(fills_received) == 1

        fill_event = manager.on_fill(fills_received[0])
        assert isinstance(fill_event, FillEvent)
        assert fill_event.symbol == "AAPL"
        assert fill_event.side == OrderSide.BUY
        assert fill_event.quantity == 100
        assert fill_event.fill_price == pytest.approx(150.0)
        assert fill_event.commission == pytest.approx(15.0)

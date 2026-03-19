"""Tests for paper broker — simulated order execution."""

import pytest

from src.broker.base_broker import BrokerFill, BrokerOrder, OrderStatus
from src.broker.paper_broker import PaperBroker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market_buy(symbol="AAPL", quantity=100):
    return BrokerOrder(
        order_id="test-001",
        symbol=symbol,
        side="BUY",
        order_type="MARKET",
        quantity=quantity,
    )


def _market_sell(symbol="AAPL", quantity=100):
    return BrokerOrder(
        order_id="test-002",
        symbol=symbol,
        side="SELL",
        order_type="MARKET",
        quantity=quantity,
    )


def _limit_buy(symbol="AAPL", quantity=100, limit_price=150.0):
    return BrokerOrder(
        order_id="test-003",
        symbol=symbol,
        side="BUY",
        order_type="LIMIT",
        quantity=quantity,
        limit_price=limit_price,
    )


def _limit_sell(symbol="AAPL", quantity=100, limit_price=155.0):
    return BrokerOrder(
        order_id="test-004",
        symbol=symbol,
        side="SELL",
        order_type="LIMIT",
        quantity=quantity,
        limit_price=limit_price,
    )


async def _connected_broker(capital=100_000.0, slippage=0.0, commission=0.0):
    broker = PaperBroker(
        initial_capital=capital,
        slippage_pct=slippage,
        commission_pct=commission,
    )
    await broker.connect()
    broker.set_price("AAPL", 150.0)
    return broker


# ---------------------------------------------------------------------------
# OrderStatus enum
# ---------------------------------------------------------------------------


class TestOrderStatus:
    def test_statuses_are_distinct(self):
        statuses = list(OrderStatus)
        assert len(statuses) == len(set(statuses))


# ---------------------------------------------------------------------------
# BrokerOrder dataclass
# ---------------------------------------------------------------------------


class TestBrokerOrder:
    def test_defaults(self):
        order = BrokerOrder(
            order_id="1",
            symbol="AAPL",
            side="BUY",
            order_type="MARKET",
            quantity=100,
        )
        assert order.status == OrderStatus.PENDING
        assert order.limit_price is None
        assert order.submitted_at is None
        assert order.filled_at is None
        assert order.filled_quantity == 0
        assert order.filled_avg_price == 0.0
        assert order.broker_order_id is None

    def test_with_limit_price(self):
        order = BrokerOrder(
            order_id="1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            quantity=100,
            limit_price=150.0,
        )
        assert order.limit_price == 150.0

    def test_mutable(self):
        order = BrokerOrder(
            order_id="1",
            symbol="AAPL",
            side="BUY",
            order_type="MARKET",
            quantity=100,
        )
        order.status = OrderStatus.FILLED
        assert order.status == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# BrokerFill dataclass
# ---------------------------------------------------------------------------


class TestBrokerFill:
    def test_creation(self):
        from datetime import datetime

        fill = BrokerFill(
            order_id="1",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            commission=0.15,
            timestamp=datetime(2025, 1, 15, 10, 30),
        )
        assert fill.symbol == "AAPL"
        assert fill.price == 150.0

    def test_frozen(self):
        from datetime import datetime

        fill = BrokerFill(
            order_id="1",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            commission=0.15,
            timestamp=datetime(2025, 1, 15),
        )
        with pytest.raises(AttributeError):
            fill.price = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Connect / disconnect
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    @pytest.mark.asyncio
    async def test_connect(self):
        broker = PaperBroker()
        await broker.connect()
        assert broker.connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self):
        broker = PaperBroker()
        await broker.connect()
        await broker.disconnect()
        assert broker.connected is False

    @pytest.mark.asyncio
    async def test_initial_state(self):
        broker = PaperBroker()
        assert broker.connected is False
        assert broker.cash == 100_000.0

    @pytest.mark.asyncio
    async def test_custom_capital(self):
        broker = PaperBroker(initial_capital=50_000.0)
        assert broker.cash == 50_000.0


# ---------------------------------------------------------------------------
# Market orders
# ---------------------------------------------------------------------------


class TestMarketOrders:
    @pytest.mark.asyncio
    async def test_buy_fills_immediately(self):
        broker = await _connected_broker()
        order = _market_buy()
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 100
        assert result.filled_at is not None

    @pytest.mark.asyncio
    async def test_sell_fills_immediately(self):
        broker = await _connected_broker()
        order = _market_sell()
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_buy_deducts_cash(self):
        broker = await _connected_broker(capital=100_000.0)
        order = _market_buy(quantity=100)
        await broker.submit_order(order)
        # 100 shares * $150 = $15,000
        assert broker.cash == 100_000.0 - 150.0 * 100

    @pytest.mark.asyncio
    async def test_sell_adds_cash(self):
        broker = await _connected_broker(capital=100_000.0)
        order = _market_sell(quantity=100)
        await broker.submit_order(order)
        # 100 shares * $150 = $15,000
        assert broker.cash == 100_000.0 + 150.0 * 100

    @pytest.mark.asyncio
    async def test_assigns_broker_order_id(self):
        broker = await _connected_broker()
        order = _market_buy()
        result = await broker.submit_order(order)
        assert result.broker_order_id is not None
        assert result.broker_order_id.startswith("PAPER-")

    @pytest.mark.asyncio
    async def test_assigns_submitted_at(self):
        broker = await _connected_broker()
        order = _market_buy()
        result = await broker.submit_order(order)
        assert result.submitted_at is not None

    @pytest.mark.asyncio
    async def test_rejected_when_not_connected(self):
        broker = PaperBroker()
        broker.set_price("AAPL", 150.0)
        order = _market_buy()
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_rejected_when_no_price(self):
        broker = PaperBroker()
        await broker.connect()
        order = _market_buy()
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_creates_fill_record(self):
        broker = await _connected_broker()
        order = _market_buy()
        await broker.submit_order(order)
        assert len(broker.fills) == 1
        assert broker.fills[0].symbol == "AAPL"
        assert broker.fills[0].side == "BUY"


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------


class TestSlippage:
    @pytest.mark.asyncio
    async def test_buy_slippage_increases_price(self):
        broker = await _connected_broker(slippage=0.01)
        order = _market_buy()
        result = await broker.submit_order(order)
        # 150 * 1.01 = 151.5
        assert result.filled_avg_price == pytest.approx(151.5)

    @pytest.mark.asyncio
    async def test_sell_slippage_decreases_price(self):
        broker = await _connected_broker(slippage=0.01)
        order = _market_sell()
        result = await broker.submit_order(order)
        # 150 * 0.99 = 148.5
        assert result.filled_avg_price == pytest.approx(148.5)

    @pytest.mark.asyncio
    async def test_zero_slippage(self):
        broker = await _connected_broker(slippage=0.0)
        order = _market_buy()
        result = await broker.submit_order(order)
        assert result.filled_avg_price == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------


class TestCommission:
    @pytest.mark.asyncio
    async def test_commission_deducted_on_buy(self):
        broker = await _connected_broker(capital=100_000.0, commission=0.001)
        order = _market_buy(quantity=100)
        await broker.submit_order(order)
        # cost = 150 * 100 = 15000, commission = 15000 * 0.001 = 15
        assert broker.cash == pytest.approx(100_000.0 - 15_000.0 - 15.0)

    @pytest.mark.asyncio
    async def test_commission_deducted_on_sell(self):
        broker = await _connected_broker(capital=100_000.0, commission=0.001)
        order = _market_sell(quantity=100)
        await broker.submit_order(order)
        # proceeds = 150 * 100 = 15000, commission = 15
        assert broker.cash == pytest.approx(100_000.0 + 15_000.0 - 15.0)

    @pytest.mark.asyncio
    async def test_commission_in_fill_record(self):
        broker = await _connected_broker(commission=0.001)
        order = _market_buy(quantity=100)
        await broker.submit_order(order)
        assert broker.fills[0].commission == pytest.approx(15.0)

    @pytest.mark.asyncio
    async def test_zero_commission(self):
        broker = await _connected_broker(commission=0.0)
        order = _market_buy(quantity=100)
        await broker.submit_order(order)
        assert broker.fills[0].commission == 0.0


# ---------------------------------------------------------------------------
# Limit orders
# ---------------------------------------------------------------------------


class TestLimitOrders:
    @pytest.mark.asyncio
    async def test_buy_limit_fills_when_price_at_or_below(self):
        broker = await _connected_broker()
        # Price is 150, limit at 155 → should fill
        order = _limit_buy(limit_price=155.0)
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.FILLED
        assert result.filled_avg_price == 155.0

    @pytest.mark.asyncio
    async def test_buy_limit_no_fill_when_price_above(self):
        broker = await _connected_broker()
        # Price is 150, limit at 140 → should NOT fill
        order = _limit_buy(limit_price=140.0)
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_sell_limit_fills_when_price_at_or_above(self):
        broker = await _connected_broker()
        # Price is 150, limit at 145 → should fill
        order = _limit_sell(limit_price=145.0)
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.FILLED
        assert result.filled_avg_price == 145.0

    @pytest.mark.asyncio
    async def test_sell_limit_no_fill_when_price_below(self):
        broker = await _connected_broker()
        # Price is 150, limit at 160 → should NOT fill
        order = _limit_sell(limit_price=160.0)
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_limit_fills_at_limit_price(self):
        broker = await _connected_broker()
        order = _limit_buy(limit_price=155.0)
        result = await broker.submit_order(order)
        # Should fill at limit price, not market
        assert result.filled_avg_price == 155.0


# ---------------------------------------------------------------------------
# Cancel orders
# ---------------------------------------------------------------------------


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_submitted_order(self):
        broker = await _connected_broker()
        # Submit limit that won't fill
        order = _limit_buy(limit_price=140.0)
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.SUBMITTED

        cancelled = await broker.cancel_order(result.order_id)
        assert cancelled.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_filled_order_no_change(self):
        broker = await _connected_broker()
        order = _market_buy()
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.FILLED

        # Cancelling a filled order should return it unchanged
        cancelled = await broker.cancel_order(result.order_id)
        assert cancelled.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_cancel_unknown_order_raises(self):
        broker = await _connected_broker()
        with pytest.raises(ValueError, match="Unknown order"):
            await broker.cancel_order("nonexistent")


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------


class TestPositions:
    @pytest.mark.asyncio
    async def test_buy_creates_position(self):
        broker = await _connected_broker()
        await broker.submit_order(_market_buy(quantity=100))

        positions = await broker.get_positions()
        assert "AAPL" in positions
        assert positions["AAPL"]["quantity"] == 100

    @pytest.mark.asyncio
    async def test_sell_reduces_position(self):
        broker = await _connected_broker()
        await broker.submit_order(_market_buy(quantity=100))
        await broker.submit_order(_market_sell(quantity=60))

        positions = await broker.get_positions()
        assert positions["AAPL"]["quantity"] == 40

    @pytest.mark.asyncio
    async def test_sell_to_zero_clears_avg_cost(self):
        broker = await _connected_broker()
        await broker.submit_order(_market_buy(quantity=100))
        await broker.submit_order(_market_sell(quantity=100))

        positions = await broker.get_positions()
        assert positions["AAPL"]["quantity"] == 0
        assert positions["AAPL"]["avg_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_multiple_buys_accumulate(self):
        broker = await _connected_broker()
        await broker.submit_order(_market_buy(quantity=50))
        await broker.submit_order(
            BrokerOrder(
                order_id="test-005",
                symbol="AAPL",
                side="BUY",
                order_type="MARKET",
                quantity=50,
            )
        )

        positions = await broker.get_positions()
        assert positions["AAPL"]["quantity"] == 100

    @pytest.mark.asyncio
    async def test_positions_returns_copy(self):
        broker = await _connected_broker()
        await broker.submit_order(_market_buy(quantity=100))

        pos = await broker.get_positions()
        pos["AAPL"]["quantity"] = 999
        pos2 = await broker.get_positions()
        assert pos2["AAPL"]["quantity"] == 100

    @pytest.mark.asyncio
    async def test_multiple_symbols(self):
        broker = await _connected_broker()
        broker.set_price("MSFT", 300.0)

        await broker.submit_order(_market_buy("AAPL", 50))
        await broker.submit_order(
            BrokerOrder(
                order_id="test-006",
                symbol="MSFT",
                side="BUY",
                order_type="MARKET",
                quantity=30,
            )
        )

        positions = await broker.get_positions()
        assert positions["AAPL"]["quantity"] == 50
        assert positions["MSFT"]["quantity"] == 30


# ---------------------------------------------------------------------------
# Account info
# ---------------------------------------------------------------------------


class TestAccountInfo:
    @pytest.mark.asyncio
    async def test_initial_account(self):
        broker = PaperBroker(initial_capital=100_000.0)
        await broker.connect()
        info = await broker.get_account_info()
        assert info["cash"] == 100_000.0
        assert info["equity"] == 100_000.0
        assert info["initial_capital"] == 100_000.0

    @pytest.mark.asyncio
    async def test_after_buy(self):
        broker = await _connected_broker(capital=100_000.0)
        await broker.submit_order(_market_buy(quantity=100))

        info = await broker.get_account_info()
        assert info["cash"] == 100_000.0 - 150.0 * 100
        # equity = cash + market_value
        assert info["equity"] == pytest.approx(100_000.0)

    @pytest.mark.asyncio
    async def test_buying_power_equals_cash(self):
        broker = await _connected_broker(capital=100_000.0)
        info = await broker.get_account_info()
        assert info["buying_power"] == info["cash"]


# ---------------------------------------------------------------------------
# Fill callbacks
# ---------------------------------------------------------------------------


class TestFillCallbacks:
    @pytest.mark.asyncio
    async def test_callback_invoked_on_fill(self):
        broker = await _connected_broker()
        received: list[BrokerFill] = []
        broker.add_fill_callback(received.append)

        await broker.submit_order(_market_buy())
        assert len(received) == 1
        assert received[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        broker = await _connected_broker()
        a: list[BrokerFill] = []
        b: list[BrokerFill] = []
        broker.add_fill_callback(a.append)
        broker.add_fill_callback(b.append)

        await broker.submit_order(_market_buy())
        assert len(a) == 1
        assert len(b) == 1

    @pytest.mark.asyncio
    async def test_no_callback_on_rejected(self):
        broker = PaperBroker()
        received: list[BrokerFill] = []
        broker.add_fill_callback(received.append)

        # Not connected → rejected
        await broker.submit_order(_market_buy())
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_callback_on_limit_fill(self):
        broker = await _connected_broker()
        received: list[BrokerFill] = []
        broker.add_fill_callback(received.append)

        order = _limit_buy(limit_price=155.0)
        await broker.submit_order(order)
        assert len(received) == 1


# ---------------------------------------------------------------------------
# get_order_status
# ---------------------------------------------------------------------------


class TestGetOrderStatus:
    @pytest.mark.asyncio
    async def test_returns_order(self):
        broker = await _connected_broker()
        order = _market_buy()
        await broker.submit_order(order)
        status = await broker.get_order_status(order.order_id)
        assert status.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_unknown_order_raises(self):
        broker = await _connected_broker()
        with pytest.raises(ValueError, match="Unknown order"):
            await broker.get_order_status("nonexistent")

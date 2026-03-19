"""Unit tests for the ExecutionHandler class."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.events.event import FillEvent, OrderEvent, OrderSide, OrderType
from src.events.queue import EventQueue
from src.execution.execution_handler import ExecutionHandler


class TestExecutionHandlerInitialization:
    """Tests for ExecutionHandler initialization."""

    def test_default_initialization(self):
        """Test ExecutionHandler with default parameters."""
        handler = ExecutionHandler()

        assert handler.commission_pct == 0.001
        assert handler.slippage_pct == 0.0005
        assert handler.events is None
        assert handler.orders_processed == 0
        assert handler.total_commission == 0.0
        assert handler.total_slippage_cost == 0.0

    def test_custom_initialization(self):
        """Test ExecutionHandler with custom parameters."""
        handler = ExecutionHandler(commission_pct=0.002, slippage_pct=0.001)

        assert handler.commission_pct == 0.002
        assert handler.slippage_pct == 0.001

    def test_zero_commission(self):
        """Test ExecutionHandler with zero commission."""
        handler = ExecutionHandler(commission_pct=0.0)

        assert handler.commission_pct == 0.0

    def test_zero_slippage(self):
        """Test ExecutionHandler with zero slippage."""
        handler = ExecutionHandler(slippage_pct=0.0)

        assert handler.slippage_pct == 0.0


class TestSetEventQueue:
    """Tests for event queue setup."""

    def test_set_event_queue(self):
        """Test setting event queue."""
        handler = ExecutionHandler()
        events = EventQueue()

        handler.set_event_queue(events)

        assert handler.events is events


class TestSlippage:
    """Tests for slippage calculation."""

    def test_buy_slippage_increases_price(self):
        """Test that BUY orders have price increased by slippage."""
        handler = ExecutionHandler(slippage_pct=0.01)  # 1% slippage

        result = handler._apply_slippage(100.0, OrderSide.BUY)

        assert result == 101.0  # 100 * 1.01

    def test_sell_slippage_decreases_price(self):
        """Test that SELL orders have price decreased by slippage."""
        handler = ExecutionHandler(slippage_pct=0.01)  # 1% slippage

        result = handler._apply_slippage(100.0, OrderSide.SELL)

        assert result == 99.0  # 100 * 0.99

    def test_zero_slippage_no_change(self):
        """Test that zero slippage doesn't change price."""
        handler = ExecutionHandler(slippage_pct=0.0)

        buy_result = handler._apply_slippage(100.0, OrderSide.BUY)
        sell_result = handler._apply_slippage(100.0, OrderSide.SELL)

        assert buy_result == 100.0
        assert sell_result == 100.0

    def test_slippage_precision(self):
        """Test slippage calculation with small percentages."""
        handler = ExecutionHandler(slippage_pct=0.0005)  # 0.05%

        result = handler._apply_slippage(150.0, OrderSide.BUY)

        # 150 * 1.0005 = 150.075
        assert abs(result - 150.075) < 0.0001


class TestExecuteOrder:
    """Tests for order execution."""

    @pytest.fixture
    def handler_with_queue(self):
        """Create a handler with event queue."""
        handler = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0005)
        events = EventQueue()
        handler.set_event_queue(events)
        return handler, events

    @pytest.fixture
    def mock_data_handler(self):
        """Create a mock data handler."""
        mock = MagicMock()
        mock.get_latest_bars.return_value = pd.DataFrame(
            {"close": [100.0]}, index=[datetime.now()]
        )
        return mock

    def test_market_buy_order_executes(self, handler_with_queue, mock_data_handler):
        """Test that market BUY order executes and emits fill."""
        handler, events = handler_with_queue

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        handler.execute_order(order, mock_data_handler)

        assert not events.empty()
        fill = events.get()
        assert isinstance(fill, FillEvent)
        assert fill.symbol == "AAPL"
        assert fill.side == OrderSide.BUY
        assert fill.quantity == 100

    def test_market_sell_order_executes(self, handler_with_queue, mock_data_handler):
        """Test that market SELL order executes and emits fill."""
        handler, events = handler_with_queue

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=50,
        )

        handler.execute_order(order, mock_data_handler)

        fill = events.get()
        assert fill.side == OrderSide.SELL
        assert fill.quantity == 50

    def test_fill_price_includes_slippage(self, handler_with_queue, mock_data_handler):
        """Test that fill price includes slippage."""
        handler, events = handler_with_queue
        handler.slippage_pct = 0.01  # 1%

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        handler.execute_order(order, mock_data_handler)

        fill = events.get()
        # Base price 100, with 1% slippage for BUY = 101
        assert fill.fill_price == 101.0

    def test_commission_calculated(self, handler_with_queue, mock_data_handler):
        """Test that commission is calculated correctly."""
        handler, events = handler_with_queue
        handler.commission_pct = 0.01  # 1%
        handler.slippage_pct = 0.0  # No slippage for easier calculation

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        handler.execute_order(order, mock_data_handler)

        fill = events.get()
        # Trade value: 100 * 100 = 10000, commission: 10000 * 0.01 = 100
        assert fill.commission == 100.0

    def test_limit_order_treated_as_market(
        self, handler_with_queue, mock_data_handler, caplog
    ):
        """Test that LIMIT orders are treated as MARKET with warning."""
        handler, events = handler_with_queue

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=100,
            limit_price=95.0,
        )

        with caplog.at_level("WARNING"):
            handler.execute_order(order, mock_data_handler)

        assert "LIMIT orders not supported" in caplog.text
        assert not events.empty()  # Order still executed

    def test_no_data_rejects_order(self, handler_with_queue, caplog):
        """Test that order is rejected when no data available."""
        handler, events = handler_with_queue

        mock = MagicMock()
        mock.get_latest_bars.return_value = pd.DataFrame()  # Empty

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        with caplog.at_level("WARNING"):
            handler.execute_order(order, mock)

        assert "No data for AAPL" in caplog.text
        assert events.empty()  # No fill emitted

    def test_orders_processed_incremented(self, handler_with_queue, mock_data_handler):
        """Test that orders_processed counter increments."""
        handler, _ = handler_with_queue

        assert handler.orders_processed == 0

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        handler.execute_order(order, mock_data_handler)
        assert handler.orders_processed == 1

        handler.execute_order(order, mock_data_handler)
        assert handler.orders_processed == 2

    def test_total_commission_accumulated(self, handler_with_queue, mock_data_handler):
        """Test that total commission accumulates across orders."""
        handler, _ = handler_with_queue
        handler.commission_pct = 0.01
        handler.slippage_pct = 0.0

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        handler.execute_order(order, mock_data_handler)
        handler.execute_order(order, mock_data_handler)

        # Each order: 100 * 100 * 0.01 = 100 commission
        assert handler.total_commission == 200.0

    def test_slippage_cost_accumulated(self, handler_with_queue, mock_data_handler):
        """Test that slippage cost accumulates across orders."""
        handler, _ = handler_with_queue
        handler.slippage_pct = 0.01  # 1%

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        handler.execute_order(order, mock_data_handler)

        # Base price 100, fill price 101, slippage cost = 1 * 100 = 100
        assert handler.total_slippage_cost == 100.0

    def test_fill_timestamp_matches_order(self, handler_with_queue, mock_data_handler):
        """Test that fill timestamp matches order timestamp."""
        handler, events = handler_with_queue
        order_time = datetime(2023, 6, 15, 10, 30, 0)

        order = OrderEvent(
            timestamp=order_time,
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        handler.execute_order(order, mock_data_handler)

        fill = events.get()
        assert fill.timestamp == order_time


class TestGetStatistics:
    """Tests for execution statistics."""

    def test_initial_statistics(self):
        """Test initial statistics are zero."""
        handler = ExecutionHandler()

        stats = handler.get_statistics()

        assert stats["orders_processed"] == 0
        assert stats["total_commission"] == 0.0
        assert stats["total_slippage_cost"] == 0.0

    def test_statistics_after_execution(self):
        """Test statistics after order execution."""
        handler = ExecutionHandler(commission_pct=0.01, slippage_pct=0.01)
        events = EventQueue()
        handler.set_event_queue(events)

        mock = MagicMock()
        mock.get_latest_bars.return_value = pd.DataFrame(
            {"close": [100.0]}, index=[datetime.now()]
        )

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        handler.execute_order(order, mock)

        stats = handler.get_statistics()

        assert stats["orders_processed"] == 1
        # Fill price: 101 (100 * 1.01), commission: 101 * 100 * 0.01 = 101
        assert stats["total_commission"] == 101.0
        # Slippage cost: (101 - 100) * 100 = 100
        assert stats["total_slippage_cost"] == 100.0


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_statistics(self):
        """Test that reset clears all statistics."""
        handler = ExecutionHandler()
        handler.orders_processed = 10
        handler.total_commission = 500.0
        handler.total_slippage_cost = 100.0

        handler.reset()

        assert handler.orders_processed == 0
        assert handler.total_commission == 0.0
        assert handler.total_slippage_cost == 0.0


class TestExecutionCalculations:
    """Tests for execution calculation correctness."""

    def test_manual_calculation_verification(self):
        """Manual verification of execution calculations."""
        handler = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0005)
        events = EventQueue()
        handler.set_event_queue(events)

        mock = MagicMock()
        mock.get_latest_bars.return_value = pd.DataFrame(
            {"close": [150.0]}, index=[datetime.now()]
        )

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        handler.execute_order(order, mock)

        fill = events.get()

        # Expected fill price: 150 * 1.0005 = 150.075
        assert abs(fill.fill_price - 150.075) < 0.0001

        # Expected commission: 150.075 * 100 * 0.001 = 15.0075
        assert abs(fill.commission - 15.0075) < 0.0001

        # Expected slippage cost: (150.075 - 150) * 100 = 7.5
        assert abs(handler.total_slippage_cost - 7.5) < 0.0001

    def test_sell_order_calculation(self):
        """Manual verification of SELL order calculations."""
        handler = ExecutionHandler(commission_pct=0.001, slippage_pct=0.001)
        events = EventQueue()
        handler.set_event_queue(events)

        mock = MagicMock()
        mock.get_latest_bars.return_value = pd.DataFrame(
            {"close": [200.0]}, index=[datetime.now()]
        )

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=50,
        )

        handler.execute_order(order, mock)

        fill = events.get()

        # Expected fill price: 200 * 0.999 = 199.8 (worse for seller)
        assert abs(fill.fill_price - 199.8) < 0.0001

        # Expected commission: 199.8 * 50 * 0.001 = 9.99
        assert abs(fill.commission - 9.99) < 0.0001

        # Expected slippage cost: (200 - 199.8) * 50 = 10
        assert abs(handler.total_slippage_cost - 10.0) < 0.0001


class TestEdgeCases:
    """Tests for edge cases."""

    def test_non_order_event_ignored(self):
        """Test that non-OrderEvent is ignored."""
        handler = ExecutionHandler()
        events = EventQueue()
        handler.set_event_queue(events)

        # Pass a FillEvent instead of OrderEvent
        fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=1.5,
        )

        mock = MagicMock()
        handler.execute_order(fill, mock)

        assert events.empty()
        assert handler.orders_processed == 0

    def test_no_event_queue_no_crash(self):
        """Test that execution doesn't crash without event queue."""
        handler = ExecutionHandler()
        # Don't set event queue

        mock = MagicMock()
        mock.get_latest_bars.return_value = pd.DataFrame(
            {"close": [100.0]}, index=[datetime.now()]
        )

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        # Should not raise
        handler.execute_order(order, mock)

        # Statistics should still be updated
        assert handler.orders_processed == 1

    def test_very_small_order(self):
        """Test execution with very small order quantity."""
        handler = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0005)
        events = EventQueue()
        handler.set_event_queue(events)

        mock = MagicMock()
        mock.get_latest_bars.return_value = pd.DataFrame(
            {"close": [100.0]}, index=[datetime.now()]
        )

        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1,
        )

        handler.execute_order(order, mock)

        fill = events.get()
        assert fill.quantity == 1
        # Commission on 1 share at ~100: 0.1
        assert fill.commission < 1.0

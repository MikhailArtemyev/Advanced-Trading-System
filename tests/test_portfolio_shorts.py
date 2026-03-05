"""Unit tests for short selling in Portfolio."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.events.event import (
    FillEvent,
    OrderSide,
    SignalEvent,
    SignalType,
)
from src.events.queue import EventQueue
from src.portfolio.portfolio import Portfolio


@pytest.fixture
def portfolio():
    """Basic portfolio with one symbol."""
    return Portfolio(initial_capital=100000.0, symbols=["AAPL"])


@pytest.fixture
def portfolio_with_queue():
    """Portfolio wired to an event queue."""
    p = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
    events = EventQueue()
    p.set_event_queue(events)
    return p, events


@pytest.fixture
def mock_data_handler():
    """Mock data handler returning AAPL at $150."""
    handler = MagicMock()
    handler.get_latest_bars.return_value = pd.DataFrame(
        {"close": [150.0]}, index=[datetime.now()]
    )
    return handler


class TestShortOnFill:
    """Tests for on_fill() with short positions."""

    def test_sell_to_open_short(self, portfolio):
        """SELL fill when flat opens a short position (negative quantity)."""
        fill = FillEvent(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=150.0,
            commission=1.50,
        )
        portfolio.on_fill(fill)

        assert portfolio.positions["AAPL"].quantity == -100

    def test_short_position_avg_cost(self, portfolio):
        """avg_cost is set to fill_price when opening short."""
        fill = FillEvent(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=150.0,
            commission=1.50,
        )
        portfolio.on_fill(fill)

        assert portfolio.positions["AAPL"].avg_cost == 150.0

    def test_short_position_cash_credited(self, portfolio):
        """Cash increases by (qty * price - commission) on short sale."""
        initial_cash = portfolio.cash
        fill = FillEvent(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=150.0,
            commission=1.50,
        )
        portfolio.on_fill(fill)

        expected_cash = initial_cash + 100 * 150.0 - 1.50
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_buy_to_cover_short(self, portfolio):
        """BUY fill when short reduces (covers) the short position."""
        # Open short
        sell_fill = FillEvent(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=150.0,
            commission=1.50,
        )
        portfolio.on_fill(sell_fill)

        # Cover half
        buy_fill = FillEvent(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=50,
            fill_price=145.0,
            commission=1.00,
        )
        portfolio.on_fill(buy_fill)

        assert portfolio.positions["AAPL"].quantity == -50

    def test_buy_to_cover_pnl(self, portfolio):
        """P&L = (avg_cost - fill_price) * cover_qty - commission."""
        # Open short at 150
        sell_fill = FillEvent(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=150.0,
            commission=1.50,
        )
        portfolio.on_fill(sell_fill)

        # Cover at 145 (profit)
        buy_fill = FillEvent(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=145.0,
            commission=1.00,
        )
        portfolio.on_fill(buy_fill)

        # P&L = (150 - 145) * 100 - 1.00 = 499.00
        expected_pnl = (150.0 - 145.0) * 100 - 1.00
        assert portfolio.positions["AAPL"].realized_pnl == pytest.approx(expected_pnl)

    def test_buy_to_cover_full_close(self, portfolio):
        """Fully covering a short resets position to zero."""
        sell_fill = FillEvent(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=150.0,
            commission=1.50,
        )
        portfolio.on_fill(sell_fill)

        buy_fill = FillEvent(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=145.0,
            commission=1.00,
        )
        portfolio.on_fill(buy_fill)

        assert portfolio.positions["AAPL"].quantity == 0
        assert portfolio.positions["AAPL"].avg_cost == 0.0

    def test_add_to_short_updates_avg_cost(self, portfolio):
        """Adding to a short position updates weighted avg_cost."""
        # Short 100 at 150
        fill1 = FillEvent(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=150.0,
            commission=1.0,
        )
        portfolio.on_fill(fill1)

        # Short 100 more at 160
        fill2 = FillEvent(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=160.0,
            commission=1.0,
        )
        portfolio.on_fill(fill2)

        assert portfolio.positions["AAPL"].quantity == -200
        # Weighted avg: (100 * 150 + 100 * 160) / 200 = 155
        assert portfolio.positions["AAPL"].avg_cost == pytest.approx(155.0)

    def test_short_trade_recorded(self, portfolio):
        """Short sell is recorded in trade history."""
        fill = FillEvent(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=150.0,
            commission=1.50,
        )
        portfolio.on_fill(fill)

        assert len(portfolio.trades) == 1
        trade = portfolio.trades[0]
        assert trade.side == "SELL"
        assert trade.quantity == 100

    def test_cover_trade_records_pnl(self, portfolio):
        """BUY to cover records P&L in trade history."""
        sell_fill = FillEvent(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=150.0,
            commission=1.0,
        )
        portfolio.on_fill(sell_fill)

        buy_fill = FillEvent(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=140.0,
            commission=1.0,
        )
        portfolio.on_fill(buy_fill)

        # The cover trade should have positive P&L
        cover_trade = portfolio.trades[1]
        assert cover_trade.side == "BUY"
        assert cover_trade.pnl == pytest.approx((150.0 - 140.0) * 100 - 1.0)


class TestShortOnSignal:
    """Tests for _generate_order() with SHORT signals."""

    def test_short_signal_generates_sell_order(
        self, portfolio_with_queue, mock_data_handler
    ):
        """SHORT signal when flat generates SELL order."""
        portfolio, events = portfolio_with_queue

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.SHORT,
            strength=1.0,
        )
        portfolio.on_signal(signal, mock_data_handler)

        assert not events.empty()
        order = events.get()
        assert order.side == OrderSide.SELL
        assert order.quantity > 0

    def test_short_signal_with_existing_long_no_order(
        self, portfolio_with_queue, mock_data_handler
    ):
        """SHORT signal when already long returns None."""
        portfolio, events = portfolio_with_queue
        portfolio.positions["AAPL"].quantity = 100

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.SHORT,
            strength=1.0,
        )
        portfolio.on_signal(signal, mock_data_handler)

        assert events.empty()

    def test_short_signal_with_existing_short_no_order(
        self, portfolio_with_queue, mock_data_handler
    ):
        """SHORT signal when already short returns None."""
        portfolio, events = portfolio_with_queue
        portfolio.positions["AAPL"].quantity = -50

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.SHORT,
            strength=1.0,
        )
        portfolio.on_signal(signal, mock_data_handler)

        assert events.empty()

    def test_short_signal_uses_position_sizer(
        self, portfolio_with_queue, mock_data_handler
    ):
        """Quantity from SHORT signal matches position sizer output."""
        portfolio, events = portfolio_with_queue
        # Default 10% of 100000 / 150 = 66 shares
        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.SHORT,
            strength=1.0,
        )
        portfolio.on_signal(signal, mock_data_handler)

        order = events.get()
        assert order.quantity == 66  # 10% of 100k / 150


class TestShortExit:
    """Tests for EXIT signal with short positions."""

    def test_exit_short_generates_buy_order(
        self, portfolio_with_queue, mock_data_handler
    ):
        """EXIT signal when short generates BUY order."""
        portfolio, events = portfolio_with_queue
        portfolio.positions["AAPL"].quantity = -100
        portfolio.positions["AAPL"].avg_cost = 150.0

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            strength=1.0,
        )
        portfolio.on_signal(signal, mock_data_handler)

        assert not events.empty()
        order = events.get()
        assert order.side == OrderSide.BUY

    def test_exit_short_covers_full_position(
        self, portfolio_with_queue, mock_data_handler
    ):
        """EXIT order quantity == abs(position.quantity)."""
        portfolio, events = portfolio_with_queue
        portfolio.positions["AAPL"].quantity = -75
        portfolio.positions["AAPL"].avg_cost = 150.0

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            strength=1.0,
        )
        portfolio.on_signal(signal, mock_data_handler)

        order = events.get()
        assert order.quantity == 75


class TestShortEquity:
    """Tests for equity calculations with shorts."""

    def test_equity_with_short_position(self, portfolio):
        """Equity = cash + (negative position value)."""
        portfolio.positions["AAPL"].quantity = -100
        portfolio.positions["AAPL"].avg_cost = 150.0
        portfolio.current_prices["AAPL"] = 150.0

        # Cash unchanged (short proceeds not simulated in init)
        # Equity = 100000 + (-100 * 150) = 85000
        assert portfolio.get_equity() == 100000.0 + (-100 * 150.0)

    def test_equity_short_profit(self, portfolio):
        """Equity increases when short position's price drops."""
        # Simulate opening short: cash gets proceeds
        portfolio.cash = 100000.0 + 100 * 150.0 - 1.0  # After short sale
        portfolio.positions["AAPL"].quantity = -100
        portfolio.positions["AAPL"].avg_cost = 150.0
        portfolio.current_prices["AAPL"] = 140.0  # Price dropped

        equity = portfolio.get_equity()
        # cash + position_value = (100000 + 15000 - 1) + (-100 * 140) = 114999 - 14000 = 100999
        expected = portfolio.cash + (-100 * 140.0)
        assert equity == pytest.approx(expected)
        assert equity > 100000.0  # Profit

    def test_equity_short_loss(self, portfolio):
        """Equity decreases when short position's price rises."""
        portfolio.cash = 100000.0 + 100 * 150.0 - 1.0
        portfolio.positions["AAPL"].quantity = -100
        portfolio.positions["AAPL"].avg_cost = 150.0
        portfolio.current_prices["AAPL"] = 160.0  # Price rose

        equity = portfolio.get_equity()
        expected = portfolio.cash + (-100 * 160.0)
        assert equity == pytest.approx(expected)
        assert equity < 100000.0 + 100 * 150.0  # Lost money vs proceeds


class TestShortUpdateTimeindex:
    """Tests for update_timeindex() with short positions."""

    def test_short_unrealized_pnl_positive(self, portfolio):
        """Short with price decline shows positive unrealized P&L."""
        portfolio.positions["AAPL"].quantity = -100
        portfolio.positions["AAPL"].avg_cost = 150.0
        portfolio.current_prices["AAPL"] = 140.0

        portfolio.update_timeindex(datetime(2023, 1, 1))

        # unrealized = (140 - 150) * (-100) = 1000
        assert portfolio.positions["AAPL"].unrealized_pnl == pytest.approx(1000.0)

    def test_short_unrealized_pnl_negative(self, portfolio):
        """Short with price increase shows negative unrealized P&L."""
        portfolio.positions["AAPL"].quantity = -100
        portfolio.positions["AAPL"].avg_cost = 150.0
        portfolio.current_prices["AAPL"] = 160.0

        portfolio.update_timeindex(datetime(2023, 1, 1))

        # unrealized = (160 - 150) * (-100) = -1000
        assert portfolio.positions["AAPL"].unrealized_pnl == pytest.approx(-1000.0)

    def test_short_market_value_negative(self, portfolio):
        """Short position market_value is negative."""
        portfolio.positions["AAPL"].quantity = -100
        portfolio.positions["AAPL"].avg_cost = 150.0
        portfolio.current_prices["AAPL"] = 145.0

        portfolio.update_timeindex(datetime(2023, 1, 1))

        # market_value = -100 * 145 = -14500
        assert portfolio.positions["AAPL"].market_value == pytest.approx(-14500.0)

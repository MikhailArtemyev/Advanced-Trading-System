"""Unit tests for the Portfolio class."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.events.event import (
    FillEvent,
    OrderSide,
    OrderType,
    SignalEvent,
    SignalType,
)
from src.events.queue import EventQueue
from src.portfolio.portfolio import Portfolio, Position, Trade
from src.risk.position_sizer import (
    FixedFractionSizer,
    VolatilityBasedSizer,
)


class TestPosition:
    """Tests for the Position dataclass."""

    def test_position_creation_defaults(self):
        """Test Position creation with default values."""
        pos = Position(symbol="AAPL")

        assert pos.symbol == "AAPL"
        assert pos.quantity == 0
        assert pos.avg_cost == 0.0
        assert pos.market_value == 0.0
        assert pos.unrealized_pnl == 0.0
        assert pos.realized_pnl == 0.0

    def test_position_creation_with_values(self):
        """Test Position creation with custom values."""
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_cost=150.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0,
        )

        assert pos.symbol == "AAPL"
        assert pos.quantity == 100
        assert pos.avg_cost == 150.0
        assert pos.market_value == 15500.0
        assert pos.unrealized_pnl == 500.0
        assert pos.realized_pnl == 200.0


class TestTrade:
    """Tests for the Trade dataclass."""

    def test_trade_creation(self):
        """Test Trade creation."""
        timestamp = datetime(2023, 1, 15, 10, 30)
        trade = Trade(
            timestamp=timestamp,
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            commission=15.0,
            pnl=0.0,
        )

        assert trade.timestamp == timestamp
        assert trade.symbol == "AAPL"
        assert trade.side == "BUY"
        assert trade.quantity == 100
        assert trade.price == 150.0
        assert trade.commission == 15.0
        assert trade.pnl == 0.0

    def test_trade_with_pnl(self):
        """Test Trade creation with P&L."""
        trade = Trade(
            timestamp=datetime.now(),
            symbol="AAPL",
            side="SELL",
            quantity=100,
            price=160.0,
            commission=16.0,
            pnl=984.0,  # (160-150)*100 - 16
        )

        assert trade.pnl == 984.0


class TestPortfolioInitialization:
    """Tests for Portfolio initialization."""

    def test_portfolio_initialization_defaults(self):
        """Test Portfolio initialization with default values."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL", "MSFT"])

        assert portfolio.initial_capital == 100000.0
        assert portfolio.cash == 100000.0
        assert portfolio.commission_pct == 0.001
        assert portfolio.position_size_pct == 0.1
        assert "AAPL" in portfolio.positions
        assert "MSFT" in portfolio.positions
        assert len(portfolio.trades) == 0
        assert len(portfolio.equity_history) == 0

    def test_portfolio_initialization_custom_params(self):
        """Test Portfolio initialization with custom parameters."""
        portfolio = Portfolio(
            initial_capital=50000.0,
            symbols=["GOOGL"],
            commission_pct=0.002,
            position_size_pct=0.2,
        )

        assert portfolio.initial_capital == 50000.0
        assert portfolio.commission_pct == 0.002
        assert portfolio.position_size_pct == 0.2

    def test_positions_initialized_empty(self):
        """Test that positions are initialized with zero values."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        pos = portfolio.positions["AAPL"]
        assert pos.quantity == 0
        assert pos.avg_cost == 0.0


class TestPortfolioEventQueue:
    """Tests for Portfolio event queue integration."""

    def test_set_event_queue(self):
        """Test setting event queue."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        events = EventQueue()

        portfolio.set_event_queue(events)

        assert portfolio.events is events


class TestPortfolioOnSignal:
    """Tests for Portfolio signal processing."""

    @pytest.fixture
    def portfolio_with_queue(self):
        """Create a portfolio with event queue."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        events = EventQueue()
        portfolio.set_event_queue(events)
        return portfolio, events

    @pytest.fixture
    def mock_data_handler(self):
        """Create a mock data handler."""
        handler = MagicMock()
        # Return a DataFrame with close price
        handler.get_latest_bars.return_value = pd.DataFrame(
            {"close": [150.0]}, index=[datetime.now()]
        )
        return handler

    def test_long_signal_generates_buy_order(
        self, portfolio_with_queue, mock_data_handler
    ):
        """Test that LONG signal generates BUY order."""
        portfolio, events = portfolio_with_queue

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=1.0,
        )

        portfolio.on_signal(signal, mock_data_handler)

        assert not events.empty()
        order = events.get()
        assert order.side == OrderSide.BUY
        assert order.symbol == "AAPL"
        assert order.order_type == OrderType.MARKET

    def test_exit_signal_generates_sell_order(
        self, portfolio_with_queue, mock_data_handler
    ):
        """Test that EXIT signal generates SELL order when position exists."""
        portfolio, events = portfolio_with_queue

        # First, create a position
        portfolio.positions["AAPL"].quantity = 100
        portfolio.positions["AAPL"].avg_cost = 145.0

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            strength=1.0,
        )

        portfolio.on_signal(signal, mock_data_handler)

        assert not events.empty()
        order = events.get()
        assert order.side == OrderSide.SELL
        assert order.quantity == 100

    def test_exit_signal_no_position_no_order(
        self, portfolio_with_queue, mock_data_handler
    ):
        """Test that EXIT signal with no position generates no order."""
        portfolio, events = portfolio_with_queue

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            strength=1.0,
        )

        portfolio.on_signal(signal, mock_data_handler)

        assert events.empty()

    def test_short_signal_generates_sell_order(
        self, portfolio_with_queue, mock_data_handler
    ):
        """Test that SHORT signal generates SELL order when flat."""
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
        assert order.symbol == "AAPL"
        assert order.quantity > 0

    def test_position_sizing(self, portfolio_with_queue, mock_data_handler):
        """Test that position size is calculated based on equity percentage."""
        portfolio, events = portfolio_with_queue
        portfolio.position_size_pct = 0.1  # 10% of 100,000 = 10,000

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=1.0,
        )

        portfolio.on_signal(signal, mock_data_handler)

        order = events.get()
        # Expected: 10,000 / 150 = 66 shares
        assert order.quantity == 66

    def test_insufficient_cash_reduces_quantity(self):
        """Test that order quantity is reduced when cash is insufficient."""
        portfolio = Portfolio(initial_capital=1000.0, symbols=["AAPL"])
        events = EventQueue()
        portfolio.set_event_queue(events)
        portfolio.position_size_pct = 1.0  # Try to use 100% of equity

        handler = MagicMock()
        handler.get_latest_bars.return_value = pd.DataFrame(
            {"close": [150.0]}, index=[datetime.now()]
        )

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=1.0,
        )

        portfolio.on_signal(signal, handler)

        order = events.get()
        # With 1000 cash, commission 0.1%, max ~6 shares at $150
        assert order.quantity <= 6


class TestPortfolioOnFill:
    """Tests for Portfolio fill processing."""

    def test_buy_fill_updates_position(self):
        """Test that BUY fill updates position correctly."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )

        portfolio.on_fill(fill)

        pos = portfolio.positions["AAPL"]
        assert pos.quantity == 100
        assert pos.avg_cost == 150.0
        # Cash: 100000 - (100 * 150 + 15) = 84985
        assert portfolio.cash == 84985.0

    def test_buy_fill_updates_avg_cost(self):
        """Test that multiple buys update average cost correctly."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        # First buy: 100 @ 150
        fill1 = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill(fill1)

        # Second buy: 100 @ 160
        fill2 = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=160.0,
            commission=16.0,
        )
        portfolio.on_fill(fill2)

        pos = portfolio.positions["AAPL"]
        assert pos.quantity == 200
        # Average cost: (100*150 + 100*160) / 200 = 155
        assert pos.avg_cost == 155.0

    def test_sell_fill_calculates_pnl(self):
        """Test that SELL fill calculates P&L correctly."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        # First buy
        buy_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill(buy_fill)

        # Then sell at profit
        sell_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=160.0,
            commission=16.0,
        )
        portfolio.on_fill(sell_fill)

        pos = portfolio.positions["AAPL"]
        assert pos.quantity == 0
        # P&L: (160 - 150) * 100 - 16 = 984
        assert pos.realized_pnl == 984.0

    def test_sell_fill_updates_cash(self):
        """Test that SELL fill updates cash correctly."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        # Buy first
        buy_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill(buy_fill)
        cash_after_buy = portfolio.cash  # 84985

        # Sell
        sell_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=160.0,
            commission=16.0,
        )
        portfolio.on_fill(sell_fill)

        # Cash: 84985 + (100 * 160 - 16) = 84985 + 15984 = 100969
        assert portfolio.cash == cash_after_buy + (100 * 160 - 16)

    def test_fill_records_trade(self):
        """Test that fills are recorded in trade history."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )

        portfolio.on_fill(fill)

        assert len(portfolio.trades) == 1
        trade = portfolio.trades[0]
        assert trade.symbol == "AAPL"
        assert trade.side == "BUY"
        assert trade.quantity == 100
        assert trade.price == 150.0
        assert trade.commission == 15.0

    def test_partial_sell(self):
        """Test partial position sell."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        # Buy 100 shares
        buy_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill(buy_fill)

        # Sell 50 shares
        sell_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            fill_price=160.0,
            commission=8.0,
        )
        portfolio.on_fill(sell_fill)

        pos = portfolio.positions["AAPL"]
        assert pos.quantity == 50
        assert pos.avg_cost == 150.0  # Avg cost unchanged
        # P&L: (160 - 150) * 50 - 8 = 492
        assert pos.realized_pnl == 492.0


class TestPortfolioEquity:
    """Tests for Portfolio equity calculations."""

    def test_initial_equity(self):
        """Test initial equity equals initial capital."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        assert portfolio.get_equity() == 100000.0

    def test_equity_with_position(self):
        """Test equity calculation with open position."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        # Buy 100 shares at 150
        fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill(fill)

        # Set current price to 160
        portfolio.current_prices["AAPL"] = 160.0

        # Cash: 84985, Position value: 100 * 160 = 16000
        # Total equity: 84985 + 16000 = 100985
        assert portfolio.get_equity() == 100985.0

    def test_equity_with_price_decline(self):
        """Test equity calculation when price declines."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        # Buy 100 shares at 150
        fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill(fill)

        # Set current price to 140
        portfolio.current_prices["AAPL"] = 140.0

        # Cash: 84985, Position value: 100 * 140 = 14000
        # Total equity: 84985 + 14000 = 98985
        assert portfolio.get_equity() == 98985.0


class TestPortfolioUpdateTimeindex:
    """Tests for Portfolio time index updates."""

    def test_update_timeindex_records_equity(self):
        """Test that update_timeindex records equity history."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        timestamp = datetime(2023, 1, 15)

        portfolio.update_timeindex(timestamp)

        assert len(portfolio.equity_history) == 1
        assert portfolio.equity_history[0]["timestamp"] == timestamp
        assert portfolio.equity_history[0]["equity"] == 100000.0
        assert portfolio.equity_history[0]["cash"] == 100000.0

    def test_update_timeindex_updates_unrealized_pnl(self):
        """Test that update_timeindex updates unrealized P&L."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        # Buy shares
        fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill(fill)
        portfolio.current_prices["AAPL"] = 160.0

        portfolio.update_timeindex(datetime.now())

        pos = portfolio.positions["AAPL"]
        assert pos.market_value == 16000.0
        assert pos.unrealized_pnl == 1000.0  # (160 - 150) * 100


class TestPortfolioGetters:
    """Tests for Portfolio getter methods."""

    def test_get_positions(self):
        """Test get_positions returns position info."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        # Create a position
        fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill(fill)

        positions = portfolio.get_positions()

        assert "AAPL" in positions
        assert positions["AAPL"]["quantity"] == 100
        assert positions["AAPL"]["avg_cost"] == 150.0

    def test_get_trade_history(self):
        """Test get_trade_history returns copy of trades."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill(fill)

        history = portfolio.get_trade_history()

        assert len(history) == 1
        # Verify it's a copy
        history.clear()
        assert len(portfolio.trades) == 1

    def test_get_equity_curve(self):
        """Test get_equity_curve returns DataFrame."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        portfolio.update_timeindex(datetime(2023, 1, 1))
        portfolio.update_timeindex(datetime(2023, 1, 2))

        equity_curve = portfolio.get_equity_curve()

        assert isinstance(equity_curve, pd.DataFrame)
        assert len(equity_curve) == 2
        assert "timestamp" in equity_curve.columns
        assert "equity" in equity_curve.columns
        assert "cash" in equity_curve.columns


class TestPortfolioPnLCalculations:
    """Tests for P&L calculation correctness."""

    def test_pnl_manual_verification(self):
        """Manual verification test for P&L calculations."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        # Buy 100 @ 50 (cost = 5000 + 5 commission)
        buy_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=50.0,
            commission=5.0,
        )
        portfolio.on_fill(buy_fill)

        assert portfolio.positions["AAPL"].quantity == 100
        assert portfolio.positions["AAPL"].avg_cost == 50.0
        assert portfolio.cash == 100000 - (100 * 50 + 5)  # 94995

        # Sell 100 @ 55 (revenue = 5500 - 5.5 commission)
        portfolio.current_prices["AAPL"] = 55.0
        sell_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=55.0,
            commission=5.5,
        )
        portfolio.on_fill(sell_fill)

        assert portfolio.positions["AAPL"].quantity == 0
        # P&L = (55 - 50) * 100 - 5.5 = 494.5
        assert abs(portfolio.positions["AAPL"].realized_pnl - 494.5) < 0.01
        # Cash = 94995 + (100 * 55 - 5.5) = 94995 + 5494.5 = 100489.5
        assert abs(portfolio.cash - 100489.5) < 0.01

    def test_multiple_trades_pnl(self):
        """Test P&L with multiple trades."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])

        # Trade 1: Buy 100 @ 50, Sell @ 55
        portfolio.on_fill(
            FillEvent(
                timestamp=datetime.now(),
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                fill_price=50.0,
                commission=5.0,
            )
        )
        portfolio.on_fill(
            FillEvent(
                timestamp=datetime.now(),
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=100,
                fill_price=55.0,
                commission=5.5,
            )
        )

        # Trade 2: Buy 100 @ 55, Sell @ 50 (loss)
        portfolio.on_fill(
            FillEvent(
                timestamp=datetime.now(),
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                fill_price=55.0,
                commission=5.5,
            )
        )
        portfolio.on_fill(
            FillEvent(
                timestamp=datetime.now(),
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=100,
                fill_price=50.0,
                commission=5.0,
            )
        )

        # Trade 1 P&L: (55-50)*100 - 5.5 = 494.5
        # Trade 2 P&L: (50-55)*100 - 5.0 = -505
        # Total realized P&L: 494.5 - 505 = -10.5
        assert abs(portfolio.positions["AAPL"].realized_pnl - (-10.5)) < 0.01


class TestPortfolioWithCustomSizer:
    """Tests for Portfolio integration with custom PositionSizer."""

    @pytest.fixture
    def mock_data_handler(self):
        """Create a mock data handler returning close=150."""
        handler = MagicMock()
        handler.get_latest_bars.return_value = pd.DataFrame(
            {"close": [150.0]}, index=[datetime.now()]
        )
        return handler

    def test_default_sizer_matches_phase1(self, mock_data_handler):
        """Default sizer produces identical output to Phase 1 hardcoded logic."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        events = EventQueue()
        portfolio.set_event_queue(events)

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=1.0,
        )
        portfolio.on_signal(signal, mock_data_handler)

        order = events.get()
        # Phase 1: int(100000 * 0.10 / 150) = 66
        assert order.quantity == 66

    def test_custom_fixed_fraction_sizer(self, mock_data_handler):
        """Custom FixedFractionSizer(0.20) produces qty=133."""
        sizer = FixedFractionSizer(fraction=0.20)
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["AAPL"],
            position_sizer=sizer,
        )
        events = EventQueue()
        portfolio.set_event_queue(events)

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=1.0,
        )
        portfolio.on_signal(signal, mock_data_handler)

        order = events.get()
        # int(100000 * 0.20 / 150) = 133
        assert order.quantity == 133

    def test_volatility_sizer_integration(self):
        """VolatilityBasedSizer integration with OHLCV mock data."""
        handler = MagicMock()
        # For on_signal: get_latest_bars(symbol, 1) → close price
        # For ATR: get_latest_bars(symbol, 15) → OHLCV data
        ohlcv_data = pd.DataFrame(
            {
                "high": [152.0] * 15,
                "low": [148.0] * 15,
                "close": [150.0] * 15,
            }
        )

        def mock_get_latest_bars(symbol, n):
            if n == 1:
                return pd.DataFrame({"close": [150.0]}, index=[datetime.now()])
            return ohlcv_data.tail(n)

        handler.get_latest_bars = mock_get_latest_bars

        sizer = VolatilityBasedSizer(
            risk_fraction=0.02, atr_period=14, atr_multiplier=2.0
        )
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["AAPL"],
            position_sizer=sizer,
        )
        events = EventQueue()
        portfolio.set_event_queue(events)

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=1.0,
        )
        portfolio.on_signal(signal, handler)

        assert not events.empty()
        order = events.get()
        # ATR = 4.0 (high-low=4, no prev_close shift on first row)
        # risk_per_share = 4.0 * 2.0 = 8.0
        # max_risk = 100000 * 0.02 = 2000
        # quantity = int(2000 / 8.0) = 250
        assert order.quantity > 0
        assert order.side == OrderSide.BUY

    def test_exit_signal_ignores_sizer(self, mock_data_handler):
        """EXIT signal closes full position regardless of sizer."""
        sizer = FixedFractionSizer(fraction=0.05)  # Small fraction
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["AAPL"],
            position_sizer=sizer,
        )
        events = EventQueue()
        portfolio.set_event_queue(events)

        # Set up a position of 200 shares
        portfolio.positions["AAPL"].quantity = 200
        portfolio.positions["AAPL"].avg_cost = 145.0

        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            strength=1.0,
        )
        portfolio.on_signal(signal, mock_data_handler)

        order = events.get()
        assert order.side == OrderSide.SELL
        assert order.quantity == 200  # Full position closed

    def test_position_sizer_attribute(self):
        """Portfolio stores the position_sizer attribute correctly."""
        # Default
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        assert isinstance(portfolio.position_sizer, FixedFractionSizer)
        assert portfolio.position_sizer.fraction == 0.1

        # Custom
        sizer = FixedFractionSizer(fraction=0.25)
        portfolio2 = Portfolio(
            initial_capital=100000.0,
            symbols=["AAPL"],
            position_sizer=sizer,
        )
        assert portfolio2.position_sizer is sizer

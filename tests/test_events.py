"""Unit tests for event classes."""

from dataclasses import asdict
from datetime import datetime

from src.events.event import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalEvent,
    SignalType,
)


class TestMarketEvent:
    """Tests for MarketEvent class."""

    def test_create_market_event(self):
        """Test creating a MarketEvent."""
        timestamp = datetime(2023, 1, 15, 10, 30, 0)
        event = MarketEvent(timestamp=timestamp)

        assert event.event_type == EventType.MARKET
        assert event.timestamp == timestamp

    def test_market_event_type_is_fixed(self):
        """Test that event_type is always MARKET."""
        event = MarketEvent(timestamp=datetime.now())
        assert event.event_type == EventType.MARKET


class TestSignalEvent:
    """Tests for SignalEvent class."""

    def test_create_signal_event(self):
        """Test creating a SignalEvent."""
        timestamp = datetime(2023, 1, 15, 10, 30, 0)
        event = SignalEvent(
            timestamp=timestamp,
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=0.8,
        )

        assert event.event_type == EventType.SIGNAL
        assert event.timestamp == timestamp
        assert event.symbol == "AAPL"
        assert event.signal_type == SignalType.LONG
        assert event.strength == 0.8

    def test_signal_event_default_strength(self):
        """Test that default strength is 1.0."""
        event = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.LONG,
        )
        assert event.strength == 1.0

    def test_signal_event_all_signal_types(self):
        """Test creating signals with all signal types."""
        for signal_type in SignalType:
            event = SignalEvent(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=signal_type,
            )
            assert event.signal_type == signal_type


class TestOrderEvent:
    """Tests for OrderEvent class."""

    def test_create_market_order(self):
        """Test creating a market order."""
        timestamp = datetime(2023, 1, 15, 10, 30, 0)
        event = OrderEvent(
            timestamp=timestamp,
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        assert event.event_type == EventType.ORDER
        assert event.timestamp == timestamp
        assert event.symbol == "AAPL"
        assert event.order_type == OrderType.MARKET
        assert event.side == OrderSide.BUY
        assert event.quantity == 100
        assert event.limit_price is None

    def test_create_limit_order(self):
        """Test creating a limit order with price."""
        event = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=50,
            limit_price=150.50,
        )

        assert event.order_type == OrderType.LIMIT
        assert event.side == OrderSide.SELL
        assert event.quantity == 50
        assert event.limit_price == 150.50

    def test_order_event_all_sides(self):
        """Test creating orders with all sides."""
        for side in OrderSide:
            event = OrderEvent(
                timestamp=datetime.now(),
                symbol="AAPL",
                order_type=OrderType.MARKET,
                side=side,
                quantity=100,
            )
            assert event.side == side


class TestFillEvent:
    """Tests for FillEvent class."""

    def test_create_fill_event(self):
        """Test creating a FillEvent."""
        timestamp = datetime(2023, 1, 15, 10, 30, 0)
        event = FillEvent(
            timestamp=timestamp,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.25,
            commission=1.50,
        )

        assert event.event_type == EventType.FILL
        assert event.timestamp == timestamp
        assert event.symbol == "AAPL"
        assert event.side == OrderSide.BUY
        assert event.quantity == 100
        assert event.fill_price == 150.25
        assert event.commission == 1.50

    def test_fill_event_sell_side(self):
        """Test creating a sell fill."""
        event = FillEvent(
            timestamp=datetime.now(),
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=200,
            fill_price=300.00,
            commission=3.00,
        )

        assert event.side == OrderSide.SELL
        assert event.symbol == "MSFT"


class TestEventSerialization:
    """Tests for event serialization (dataclass compatibility)."""

    def test_market_event_to_dict(self):
        """Test MarketEvent can be converted to dict."""
        timestamp = datetime(2023, 1, 15, 10, 30, 0)
        event = MarketEvent(timestamp=timestamp)
        data = asdict(event)

        assert data["event_type"] == EventType.MARKET
        assert data["timestamp"] == timestamp

    def test_signal_event_to_dict(self):
        """Test SignalEvent can be converted to dict."""
        event = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=0.9,
        )
        data = asdict(event)

        assert data["symbol"] == "AAPL"
        assert data["signal_type"] == SignalType.LONG
        assert data["strength"] == 0.9

    def test_order_event_to_dict(self):
        """Test OrderEvent can be converted to dict."""
        event = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )
        data = asdict(event)

        assert data["symbol"] == "AAPL"
        assert data["order_type"] == OrderType.MARKET
        assert data["side"] == OrderSide.BUY
        assert data["quantity"] == 100

    def test_fill_event_to_dict(self):
        """Test FillEvent can be converted to dict."""
        event = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.25,
            commission=1.50,
        )
        data = asdict(event)

        assert data["fill_price"] == 150.25
        assert data["commission"] == 1.50

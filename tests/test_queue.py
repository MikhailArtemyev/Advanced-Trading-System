"""Unit tests for EventQueue class."""

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
from src.events.queue import EventQueue


class TestEventQueue:
    """Tests for EventQueue class."""

    def test_initial_state(self):
        """Test initial state of empty queue."""
        queue = EventQueue()
        assert queue.empty() is True
        assert queue.event_count == 0
        assert queue.qsize() == 0

    def test_put_single_event(self):
        """Test adding a single event to queue."""
        queue = EventQueue()
        event = MarketEvent(timestamp=datetime.now())

        queue.put(event)

        assert queue.empty() is False
        assert queue.event_count == 1
        assert queue.qsize() == 1

    def test_put_multiple_events(self):
        """Test adding multiple events to queue."""
        queue = EventQueue()

        for _ in range(5):
            event = MarketEvent(timestamp=datetime.now())
            queue.put(event)

        assert queue.event_count == 5
        assert queue.qsize() == 5

    def test_get_event(self):
        """Test getting an event from queue."""
        queue = EventQueue()
        timestamp = datetime.now()
        event = MarketEvent(timestamp=timestamp)

        queue.put(event)
        retrieved = queue.get()

        assert retrieved is not None
        assert retrieved.event_type == EventType.MARKET
        assert retrieved.timestamp == timestamp
        assert queue.empty() is True

    def test_get_empty_queue_returns_none(self):
        """Test that getting from empty queue returns None."""
        queue = EventQueue()
        result = queue.get(block=False)
        assert result is None

    def test_fifo_order(self):
        """Test that events are retrieved in FIFO order."""
        queue = EventQueue()
        timestamps = [datetime(2023, 1, i) for i in range(1, 4)]

        for ts in timestamps:
            queue.put(MarketEvent(timestamp=ts))

        for expected_ts in timestamps:
            event = queue.get()
            assert event.timestamp == expected_ts

    def test_different_event_types(self):
        """Test queue handles different event types."""
        queue = EventQueue()
        ts = datetime.now()

        # Add different event types
        queue.put(MarketEvent(timestamp=ts))
        queue.put(
            SignalEvent(
                timestamp=ts, symbol="AAPL", signal_type=SignalType.LONG, strength=0.8
            )
        )
        queue.put(
            OrderEvent(
                timestamp=ts,
                symbol="AAPL",
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=100,
            )
        )
        queue.put(
            FillEvent(
                timestamp=ts,
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                fill_price=150.0,
                commission=1.5,
            )
        )

        assert queue.event_count == 4

        # Verify order and types
        assert queue.get().event_type == EventType.MARKET
        assert queue.get().event_type == EventType.SIGNAL
        assert queue.get().event_type == EventType.ORDER
        assert queue.get().event_type == EventType.FILL

    def test_clear(self):
        """Test clearing the queue."""
        queue = EventQueue()

        for _ in range(10):
            queue.put(MarketEvent(timestamp=datetime.now()))

        assert queue.qsize() == 10

        queue.clear()

        assert queue.empty() is True
        assert queue.qsize() == 0
        # event_count should still reflect total events added
        assert queue.event_count == 10

    def test_clear_empty_queue(self):
        """Test clearing an already empty queue doesn't error."""
        queue = EventQueue()
        queue.clear()  # Should not raise
        assert queue.empty() is True

    def test_event_count_persists_after_get(self):
        """Test that event_count tracks total adds, not current size."""
        queue = EventQueue()

        queue.put(MarketEvent(timestamp=datetime.now()))
        queue.put(MarketEvent(timestamp=datetime.now()))
        queue.get()

        assert queue.event_count == 2  # Still 2, even though 1 removed
        assert queue.qsize() == 1


class TestEventQueueBlocking:
    """Tests for blocking behavior of EventQueue."""

    def test_get_with_timeout_empty_queue(self):
        """Test that get with timeout returns None when queue empty."""
        queue = EventQueue()
        result = queue.get(block=True, timeout=0.01)
        assert result is None

    def test_get_with_timeout_has_event(self):
        """Test that get with timeout returns event when available."""
        queue = EventQueue()
        event = MarketEvent(timestamp=datetime.now())
        queue.put(event)

        result = queue.get(block=True, timeout=0.1)
        assert result is not None
        assert result.event_type == EventType.MARKET

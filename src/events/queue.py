"""Event queue for the backtesting engine.

The EventQueue is the central communication mechanism between all
components in the event-driven architecture:

    DataHandler -> MarketEvent -> Strategy -> SignalEvent ->
    Portfolio -> OrderEvent -> ExecutionHandler -> FillEvent -> Portfolio
"""

from queue import Empty, Queue

from .event import Event


class EventQueue:
    """FIFO event queue for the backtesting engine.

    All components communicate through this queue:
    - DataHandler -> MarketEvent
    - Strategy -> SignalEvent
    - Portfolio -> OrderEvent
    - ExecutionHandler -> FillEvent

    Attributes:
        _queue: Internal Python Queue for thread-safe operations
        _event_count: Total number of events added to queue
    """

    def __init__(self) -> None:
        """Initialize an empty event queue."""
        self._queue: Queue[Event] = Queue()
        self._event_count: int = 0

    def put(self, event: Event) -> None:
        """Add an event to the queue.

        Args:
            event: The event to add
        """
        self._queue.put(event)
        self._event_count += 1

    def get(self, block: bool = False, timeout: float | None = None) -> Event | None:
        """Get the next event from the queue.

        Args:
            block: If True, wait for an event to be available
            timeout: Maximum seconds to wait if blocking

        Returns:
            The next event, or None if queue is empty (non-blocking mode)

        Raises:
            queue.Empty: If blocking and timeout exceeded
        """
        try:
            return self._queue.get(block=block, timeout=timeout)
        except Empty:
            return None

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            True if queue has no pending events
        """
        return self._queue.empty()

    @property
    def event_count(self) -> int:
        """Total number of events added to queue since creation."""
        return self._event_count

    def clear(self) -> None:
        """Remove all pending events from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def qsize(self) -> int:
        """Return approximate number of events in queue.

        Note: This is approximate due to thread-safety considerations.
        """
        return self._queue.qsize()

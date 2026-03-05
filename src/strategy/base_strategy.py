"""Abstract base class for trading strategies.

Strategies receive MarketEvents and may emit SignalEvents based on
their trading logic. All strategies must implement calculate_signals().

IMPORTANT: Strategies must only access market data through
data_handler.get_latest_bars() to prevent look-ahead bias.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from ..data.data_handler import DataHandler
from ..events.event import MarketEvent, SignalEvent, SignalType
from ..events.queue import EventQueue


class Strategy(ABC):
    """Abstract base class for trading strategies.

    Subclasses must implement:
        calculate_signals(): Core signal generation logic

    Strategies receive MarketEvents and may emit SignalEvents.

    Attributes:
        symbols: List of symbols this strategy trades
        parameters: Strategy-specific parameters
        events: Event queue for emitting signals
        current_positions: Tracked positions per symbol
    """

    def __init__(
        self,
        symbols: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            symbols: List of symbols to trade
            parameters: Strategy-specific parameters
        """
        self.symbols = symbols
        self.parameters = parameters or {}
        self.events: EventQueue | None = None

        # Track strategy state
        self.current_positions: dict[str, int] = dict.fromkeys(symbols, 0)

    def set_event_queue(self, events: EventQueue) -> None:
        """Set the event queue for emitting signals.

        Args:
            events: The event queue to use
        """
        self.events = events

    def on_market_data(
        self,
        event: MarketEvent,
        data_handler: DataHandler,
    ) -> None:
        """Handle incoming market data event.

        This method calls calculate_signals() and emits any resulting
        signals to the event queue. Subclasses should NOT override
        this method.

        Args:
            event: The market event
            data_handler: Access to market data
        """
        signals = self.calculate_signals(event.timestamp, data_handler)

        for signal in signals:
            if self.events is not None:
                self.events.put(signal)

    @abstractmethod
    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        """Generate trading signals based on current market data.

        MUST be implemented by subclasses.

        Args:
            timestamp: Current simulation time
            data_handler: Access to market data

        Returns:
            List of SignalEvent objects (may be empty)

        IMPORTANT: Only use data_handler.get_latest_bars() to access data.
        This ensures no look-ahead bias. Never access future data.
        """
        raise NotImplementedError

    def update_position(self, symbol: str, quantity: int) -> None:
        """Update tracked position for a symbol.

        Called by portfolio after fills to keep strategy in sync.

        Args:
            symbol: The symbol
            quantity: New position quantity
        """
        self.current_positions[symbol] = quantity

    def _create_signal(
        self,
        timestamp: datetime,
        symbol: str,
        signal_type: SignalType,
        strength: float = 1.0,
    ) -> SignalEvent:
        """Helper to create signal events.

        Args:
            timestamp: Signal timestamp
            symbol: The symbol
            signal_type: LONG, SHORT, or EXIT
            strength: Signal strength 0.0 to 1.0

        Returns:
            A new SignalEvent
        """
        return SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
        )

    def get_position(self, symbol: str) -> int:
        """Get current position for a symbol.

        Args:
            symbol: The symbol

        Returns:
            Current position quantity (0 if no position)
        """
        return self.current_positions.get(symbol, 0)

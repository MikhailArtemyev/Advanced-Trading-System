"""Main backtesting engine implementing event-driven simulation.

The BacktestEngine orchestrates the event-driven simulation:

1. DataHandler emits MarketEvent when new data is available
2. Strategy receives MarketEvent, may emit SignalEvent
3. Portfolio receives SignalEvent, may emit OrderEvent
4. ExecutionHandler receives OrderEvent, emits FillEvent
5. Portfolio receives FillEvent, updates positions

Loop continues until data is exhausted.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from ..data.data_handler import DataHandler
from ..events.event import (
    Event,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
)
from ..events.queue import EventQueue
from ..risk.risk_manager import RiskManager
from ..storage.backend import StorageBackend

if TYPE_CHECKING:
    import pandas as pd


class StrategyProtocol(Protocol):
    """Protocol defining the Strategy interface."""

    def set_event_queue(self, events: EventQueue) -> None:
        """Set the event queue for emitting signals."""
        ...

    def on_market_data(self, event: MarketEvent, data_handler: DataHandler) -> None:
        """Handle incoming market data event."""
        ...

    def update_position(self, symbol: str, quantity: int) -> None:
        """Update tracked position after a fill."""
        ...


class PortfolioProtocol(Protocol):
    """Protocol defining the Portfolio interface."""

    def set_event_queue(self, events: EventQueue) -> None:
        """Set the event queue for emitting orders."""
        ...

    def on_signal(self, event: Event, data_handler: DataHandler) -> None:
        """Handle incoming signal event."""
        ...

    def on_fill(self, event: Event) -> None:
        """Handle incoming fill event."""
        ...

    def update_timeindex(self, timestamp: datetime) -> None:
        """Update portfolio valuation at timestamp."""
        ...

    def get_equity(self) -> float:
        """Return current portfolio equity."""
        ...

    def get_positions(self) -> dict[str, Any]:
        """Return current positions."""
        ...

    def get_trade_history(self) -> list[Any]:
        """Return trade history."""
        ...


class ExecutionProtocol(Protocol):
    """Protocol defining the ExecutionHandler interface."""

    def set_event_queue(self, events: EventQueue) -> None:
        """Set the event queue for emitting fills."""
        ...

    def execute_order(self, event: Event, data_handler: DataHandler) -> None:
        """Execute an order event."""
        ...


class PerformanceTrackerProtocol(Protocol):
    """Protocol defining the PerformanceTracker interface."""

    def update(self, timestamp: datetime, equity: float) -> None:
        """Record equity value at timestamp."""
        ...

    def calculate_metrics(self) -> dict[str, Any]:
        """Calculate performance metrics."""
        ...

    def get_equity_curve(self) -> "pd.DataFrame":
        """Return equity curve as DataFrame."""
        ...


class BacktestEngine:
    """Main backtesting engine implementing event-driven simulation.

    Event Flow:
        1. DataHandler emits MarketEvent
        2. Strategy receives MarketEvent, may emit SignalEvent
        3. Portfolio receives SignalEvent, may emit OrderEvent
        4. ExecutionHandler receives OrderEvent, emits FillEvent
        5. Portfolio receives FillEvent, updates positions

    Attributes:
        data_handler: Source of market data
        strategy: Trading strategy generating signals
        portfolio: Portfolio managing positions and P&L
        execution_handler: Simulated order execution
        performance_tracker: Optional performance metrics tracker
        events: Central event queue
        bar_count: Number of bars processed
        event_count: Number of events processed
    """

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: StrategyProtocol,
        portfolio: PortfolioProtocol,
        execution_handler: ExecutionProtocol,
        performance_tracker: PerformanceTrackerProtocol | None = None,
        risk_manager: RiskManager | None = None,
        storage: StorageBackend | None = None,
    ) -> None:
        """Initialize the backtest engine.

        Args:
            data_handler: Data source for market data
            strategy: Trading strategy to test
            portfolio: Portfolio manager
            execution_handler: Order execution simulator
            performance_tracker: Optional performance tracker
            risk_manager: Optional pre-trade risk manager
            storage: Optional storage backend for persisting results
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.performance_tracker = performance_tracker
        self.risk_manager = risk_manager
        self.storage = storage

        self.events = EventQueue()

        # Wire up components to event queue
        self.strategy.set_event_queue(self.events)
        self.portfolio.set_event_queue(self.events)
        self.execution_handler.set_event_queue(self.events)

        # Statistics
        self.bar_count = 0
        self.event_count = 0
        self.rejected_orders = 0

    def run(self) -> dict[str, Any]:
        """Execute backtest simulation.

        Returns:
            Dictionary with performance results including:
            - bars_processed: Number of bars iterated
            - events_processed: Total events handled
            - final_equity: Ending portfolio value
            - positions: Final positions
            - trade_history: List of all trades
            - metrics: Performance metrics (if tracker provided)
            - equity_curve: Equity over time (if tracker provided)
        """
        print("Starting backtest...")
        start_time = datetime.now()

        # Main loop - iterate through each bar
        while self.data_handler.continue_backtest:
            # Advance to next bar
            self.data_handler.update_bars()
            self.bar_count += 1

            # Get current timestamp
            timestamp = self.data_handler.get_current_timestamp()

            # Emit market event
            market_event = MarketEvent(timestamp)
            self.events.put(market_event)

            # Process all events generated by this bar
            self._process_events()

            # Update portfolio valuation
            self.portfolio.update_timeindex(timestamp)

            # Update risk manager peak equity
            if self.risk_manager:
                self.risk_manager.update_peak_equity(self.portfolio.get_equity())

            # Track performance
            if self.performance_tracker:
                self.performance_tracker.update(timestamp, self.portfolio.get_equity())

            # Progress update every 500 bars
            if self.bar_count % 500 == 0:
                print(f"Processed {self.bar_count} bars...")

        # Finalize
        duration = (datetime.now() - start_time).total_seconds()

        print("\nBacktest complete:")
        print(f"  Bars processed: {self.bar_count}")
        print(f"  Events processed: {self.event_count}")
        print(f"  Duration: {duration:.2f} seconds")

        # Persist results to storage if configured
        if self.storage is not None:
            self._persist_to_storage()

        return self._generate_results()

    def _persist_to_storage(self) -> None:
        """Bulk-write backtest results to storage."""
        if self.storage is None:
            return

        session_id = self.storage.create_session(
            session_type="backtest",
            config={},
            initial_capital=getattr(self.portfolio, "initial_capital", 0.0),
        )

        # Bulk insert trades
        trades = self.portfolio.get_trade_history()
        if trades and hasattr(self.storage, "bulk_record_trades"):
            self.storage.bulk_record_trades(session_id, trades)

        # Bulk insert equity snapshots
        equity_history = getattr(self.portfolio, "equity_history", [])
        if equity_history and hasattr(self.storage, "bulk_record_equity"):
            self.storage.bulk_record_equity(session_id, equity_history)

        self.storage.end_session(session_id, self.portfolio.get_equity())

    def _process_events(self) -> None:
        """Process all events in queue until empty."""
        while not self.events.empty():
            event = self.events.get()
            if event is None:
                break

            self.event_count += 1
            self._handle_event(event)

    def _handle_event(self, event: Event) -> None:
        """Route event to appropriate handler.

        Args:
            event: The event to handle
        """
        if event.event_type == EventType.MARKET:
            # Event is guaranteed to be MarketEvent when event_type is MARKET
            market_event = MarketEvent(timestamp=event.timestamp)
            self.strategy.on_market_data(market_event, self.data_handler)

        elif event.event_type == EventType.SIGNAL:
            self.portfolio.on_signal(event, self.data_handler)

        elif event.event_type == EventType.ORDER:
            if self.risk_manager is not None and isinstance(event, OrderEvent):
                bars = self.data_handler.get_latest_bars(event.symbol, 1)
                current_price = float(bars["close"].iloc[-1]) if not bars.empty else 0.0
                result = self.risk_manager.check_order(
                    order=event,
                    equity=self.portfolio.get_equity(),
                    positions=self.portfolio.get_positions(),
                    current_price=current_price,
                )
                if not result.approved:
                    self.rejected_orders += 1
                    return
                if result.adjusted_quantity is not None:
                    event.quantity = result.adjusted_quantity

            self.execution_handler.execute_order(event, self.data_handler)

        elif event.event_type == EventType.FILL:
            self.portfolio.on_fill(event)

            # Update risk manager daily P&L on SELL fills
            if (
                self.risk_manager is not None
                and isinstance(event, FillEvent)
                and event.side == OrderSide.SELL
            ):
                trade_history = self.portfolio.get_trade_history()
                if trade_history:
                    latest = trade_history[-1]
                    if isinstance(latest, dict):
                        pnl = float(latest.get("pnl", 0.0))
                    else:
                        pnl = float(getattr(latest, "pnl", 0.0))
                    self.risk_manager.update_daily_pnl(pnl, event.timestamp)

            # Sync position back to strategy so it knows current holdings
            if isinstance(event, FillEvent) and hasattr(
                self.strategy, "update_position"
            ):
                positions = self.portfolio.get_positions()
                if event.symbol in positions:
                    pos = positions[event.symbol]
                    qty = pos["quantity"] if isinstance(pos, dict) else pos
                    self.strategy.update_position(event.symbol, qty)

    def _generate_results(self) -> dict[str, Any]:
        """Generate final backtest results.

        Returns:
            Dictionary containing all backtest results
        """
        results: dict[str, Any] = {
            "bars_processed": self.bar_count,
            "events_processed": self.event_count,
            "final_equity": self.portfolio.get_equity(),
            "positions": self.portfolio.get_positions(),
            "trade_history": self.portfolio.get_trade_history(),
            "rejected_orders": self.rejected_orders,
        }

        if self.performance_tracker:
            results["metrics"] = self.performance_tracker.calculate_metrics()
            results["equity_curve"] = self.performance_tracker.get_equity_curve()

        return results

    def reset(self) -> None:
        """Reset the engine for another run."""
        self.data_handler.reset()
        self.events.clear()
        self.bar_count = 0
        self.event_count = 0
        self.rejected_orders = 0
        if self.risk_manager:
            self.risk_manager.reset()

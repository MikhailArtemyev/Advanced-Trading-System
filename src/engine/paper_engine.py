"""PaperTradingEngine — real-time event loop for paper trading.

Mirrors BacktestEngine's event-driven architecture but runs
asynchronously against live data feeds. The core event cycle
is identical: MarketEvent -> Strategy -> SignalEvent -> Portfolio ->
OrderEvent -> Broker -> FillEvent -> Portfolio.

The engine runs continuously until stopped, processing events
as they arrive from the live data feed.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from src.broker.base_broker import BrokerFill
from src.broker.order_manager import OrderManager
from src.data.data_handler import DataHandler
from src.events.event import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    SignalEvent,
)
from src.events.queue import EventQueue
from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Async event-driven engine for paper trading.

    Reuses the same Strategy, Portfolio, and RiskManager components
    from backtesting. The key differences from BacktestEngine:
    - Runs asynchronously with asyncio
    - Receives bars from LiveDataHandler instead of replaying history
    - Submits orders through BrokerAdapter instead of simulating fills
    - Runs continuously until explicitly stopped

    Args:
        data_handler: LiveDataHandler providing real-time bars.
        strategy: Strategy instance (same as backtest).
        portfolio: Portfolio instance (same as backtest).
        order_manager: OrderManager wrapping a BrokerAdapter.
        risk_manager: Optional RiskManager for pre-trade checks.
        event_poll_interval: Seconds between event queue polls.
        bar_poll_interval: Seconds between checking for new bars.
    """

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Any,
        portfolio: Any,
        order_manager: OrderManager,
        risk_manager: RiskManager | None = None,
        event_poll_interval: float = 0.01,
        bar_poll_interval: float = 0.1,
    ) -> None:
        self._data_handler = data_handler
        self._strategy = strategy
        self._portfolio = portfolio
        self._order_manager = order_manager
        self._risk_manager = risk_manager
        self._event_poll_interval = event_poll_interval
        self._bar_poll_interval = bar_poll_interval

        self._events = EventQueue()
        self._running = False
        self._started_at: datetime | None = None

        # Statistics
        self.bars_processed = 0
        self.events_processed = 0
        self.orders_submitted = 0
        self.orders_rejected = 0
        self.fills_processed = 0

        # Wire up components
        self._strategy.set_event_queue(self._events)
        self._portfolio.set_event_queue(self._events)

    @property
    def is_running(self) -> bool:
        """Whether the engine is currently running."""
        return self._running

    @property
    def uptime_seconds(self) -> float:
        """Seconds since the engine was started."""
        if self._started_at is None:
            return 0.0
        return (datetime.now() - self._started_at).total_seconds()

    async def start(self) -> None:
        """Start the paper trading engine.

        Runs two concurrent tasks:
        1. Bar poller: checks for new bars and emits MarketEvents
        2. Event processor: processes events from the queue
        """
        self._running = True
        self._started_at = datetime.now()
        logger.info("Paper trading engine started")

        try:
            await asyncio.gather(
                self._bar_poll_loop(),
                self._event_process_loop(),
            )
        except asyncio.CancelledError:
            logger.info("Paper trading engine cancelled")
        finally:
            self._running = False
            logger.info(
                "Engine stopped. Bars=%d Events=%d Orders=%d Fills=%d",
                self.bars_processed,
                self.events_processed,
                self.orders_submitted,
                self.fills_processed,
            )

    async def stop(self) -> None:
        """Signal the engine to stop gracefully."""
        self._running = False
        logger.info("Stop signal sent to paper trading engine")

    async def _bar_poll_loop(self) -> None:
        """Poll for new bars and emit MarketEvents."""
        while self._running:
            if self._data_handler.update_bars():
                timestamp = self._data_handler.get_current_timestamp()
                event = MarketEvent(timestamp=timestamp)
                self._events.put(event)
                self.bars_processed += 1
            await asyncio.sleep(self._bar_poll_interval)

    async def _event_process_loop(self) -> None:
        """Process events from the queue."""
        while self._running:
            if self._events.empty():
                await asyncio.sleep(self._event_poll_interval)
                continue

            event = self._events.get()
            if event is None:
                continue

            self.events_processed += 1

            if isinstance(event, MarketEvent):
                self._handle_market(event)
            elif isinstance(event, SignalEvent):
                self._handle_signal(event)
            elif isinstance(event, OrderEvent):
                await self._handle_order(event)
            elif isinstance(event, FillEvent):
                self._handle_fill(event)

    def _handle_market(self, event: MarketEvent) -> None:
        """Dispatch market event to strategy."""
        self._strategy.on_market_data(event, self._data_handler)
        self._portfolio.update_timeindex(event.timestamp)

        # Update risk manager peak equity
        if self._risk_manager is not None:
            self._risk_manager.update_peak_equity(self._portfolio.get_equity())

    def _handle_signal(self, event: SignalEvent) -> None:
        """Dispatch signal event to portfolio for order generation."""
        self._portfolio.on_signal(event, self._data_handler)

    async def _handle_order(self, event: OrderEvent) -> None:
        """Check order with risk manager, then submit to broker."""
        if self._risk_manager is not None:
            bars = self._data_handler.get_latest_bars(event.symbol, 1)
            if bars.empty:
                logger.warning("No price for %s — rejecting order", event.symbol)
                self.orders_rejected += 1
                return

            current_price = float(bars["close"].iloc[-1])
            result = self._risk_manager.check_order(
                order=event,
                equity=self._portfolio.get_equity(),
                positions=self._portfolio.get_positions(),
                current_price=current_price,
            )
            if not result.approved:
                logger.warning(
                    "Order rejected by risk manager: %s",
                    result.rejection_reasons,
                )
                self.orders_rejected += 1
                return
            if result.adjusted_quantity is not None:
                event.quantity = result.adjusted_quantity

        # Submit to broker via order manager
        try:
            broker_order = await self._order_manager.submit_order_event(event)
            self.orders_submitted += 1
            logger.info(
                "Order submitted: %s %s %d %s",
                broker_order.side,
                broker_order.symbol,
                broker_order.quantity,
                broker_order.status.name,
            )

            # If filled immediately (paper broker market orders), emit FillEvent
            if broker_order.status.name == "FILLED":
                fill = BrokerFill(
                    order_id=broker_order.order_id,
                    symbol=broker_order.symbol,
                    side=broker_order.side,
                    quantity=broker_order.filled_quantity,
                    price=broker_order.filled_avg_price,
                    commission=abs(
                        broker_order.filled_avg_price
                        * broker_order.filled_quantity
                        * 0.001
                    ),
                    timestamp=broker_order.filled_at or datetime.now(),
                )
                fill_event = self._order_manager.on_fill(fill)
                self._events.put(fill_event)

        except Exception:
            logger.exception("Failed to submit order")
            self.orders_rejected += 1

    def _handle_fill(self, event: FillEvent) -> None:
        """Dispatch fill event to portfolio."""
        self._portfolio.on_fill(event)
        self.fills_processed += 1

        # Update risk manager daily P&L on SELL fills
        if self._risk_manager is not None and event.side == OrderSide.SELL:
            trade_history = self._portfolio.get_trade_history()
            if trade_history:
                latest = trade_history[-1]
                if isinstance(latest, dict):
                    pnl = float(latest.get("pnl", 0.0))
                else:
                    pnl = float(getattr(latest, "pnl", 0.0))
                self._risk_manager.update_daily_pnl(pnl, event.timestamp)

        # Sync position back to strategy
        if hasattr(self._strategy, "update_position"):
            positions = self._portfolio.get_positions()
            if event.symbol in positions:
                pos = positions[event.symbol]
                qty = pos["quantity"] if isinstance(pos, dict) else pos
                self._strategy.update_position(event.symbol, qty)

    def get_statistics(self) -> dict[str, Any]:
        """Return engine runtime statistics."""
        return {
            "is_running": self._running,
            "uptime_seconds": self.uptime_seconds,
            "bars_processed": self.bars_processed,
            "events_processed": self.events_processed,
            "orders_submitted": self.orders_submitted,
            "orders_rejected": self.orders_rejected,
            "fills_processed": self.fills_processed,
            "equity": self._portfolio.get_equity(),
            "positions": self._portfolio.get_positions(),
        }

    def get_state(self) -> dict[str, Any]:
        """Return full engine state for persistence.

        Returns a serializable dict suitable for StateManager.save_snapshot().
        """
        return {
            "statistics": self.get_statistics(),
            "cash": self._portfolio.cash,
            "initial_capital": self._portfolio.initial_capital,
            "trades": [
                {
                    "timestamp": str(t.timestamp),
                    "symbol": t.symbol,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "commission": t.commission,
                    "pnl": t.pnl,
                }
                for t in self._portfolio.trades
            ],
            "orders": {
                oid: {
                    "order_id": o.order_id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "order_type": o.order_type,
                    "quantity": o.quantity,
                    "status": o.status.name,
                    "filled_quantity": o.filled_quantity,
                    "filled_avg_price": o.filled_avg_price,
                }
                for oid, o in self._order_manager.all_orders.items()
            },
        }

    async def reconcile_positions(self) -> dict[str, Any]:
        """Compare Portfolio positions against PaperBroker positions.

        In paper mode these should always match, but this provides
        an auditable check for correctness.

        Returns:
            Reconciliation report with matched status and discrepancies.
        """
        engine_positions = self._portfolio.get_positions()
        broker_positions = await self._order_manager._broker.get_positions()

        discrepancies: list[dict[str, Any]] = []
        all_symbols = set(engine_positions.keys()) | set(broker_positions.keys())

        for symbol in sorted(all_symbols):
            engine_qty = 0
            broker_qty = 0

            if symbol in engine_positions:
                pos = engine_positions[symbol]
                engine_qty = pos["quantity"] if isinstance(pos, dict) else pos

            if symbol in broker_positions:
                bpos = broker_positions[symbol]
                broker_qty = (
                    int(bpos["quantity"]) if isinstance(bpos, dict) else int(bpos)
                )

            if engine_qty != broker_qty:
                discrepancies.append(
                    {
                        "symbol": symbol,
                        "engine_quantity": engine_qty,
                        "broker_quantity": broker_qty,
                        "difference": engine_qty - broker_qty,
                    }
                )

        return {
            "timestamp": datetime.now().isoformat(),
            "matched": len(discrepancies) == 0,
            "symbols_checked": len(all_symbols),
            "discrepancies": discrepancies,
        }

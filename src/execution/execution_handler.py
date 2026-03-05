"""Simulated execution handler for backtesting.

Handles order execution with configurable slippage and commission modeling.
"""

from ..data.data_handler import DataHandler
from ..events.event import Event, FillEvent, OrderEvent, OrderSide, OrderType
from ..events.queue import EventQueue


class ExecutionHandler:
    """Simulated execution handler for backtesting.

    Simulates order fills with configurable:
    - Commission (percentage of trade value)
    - Slippage (percentage of price)

    For MVP: All market orders fill at close price + slippage.
    Limit orders are not supported in MVP (treated as market orders).

    Attributes:
        commission_pct: Commission as percentage of trade value
        slippage_pct: Slippage as percentage of price
        events: Event queue for emitting fill events
        orders_processed: Count of orders executed
        total_commission: Cumulative commission paid
        total_slippage_cost: Cumulative slippage cost
    """

    def __init__(
        self,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        """Initialize the execution handler.

        Args:
            commission_pct: Commission as % of trade value (e.g., 0.001 = 0.1%)
            slippage_pct: Slippage as % of price (e.g., 0.0005 = 0.05%)
        """
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.events: EventQueue | None = None

        # Statistics
        self.orders_processed: int = 0
        self.total_commission: float = 0.0
        self.total_slippage_cost: float = 0.0

    def set_event_queue(self, events: EventQueue) -> None:
        """Set event queue for emitting fills.

        Args:
            events: The event queue to use for emitting FillEvents
        """
        self.events = events

    def execute_order(self, event: Event, data_handler: DataHandler) -> None:
        """Execute an order and emit FillEvent.

        For backtesting, we assume:
        - Market orders always fill
        - Fill at close price + slippage
        - No partial fills

        Args:
            event: The order event to execute
            data_handler: Data handler for getting current prices
        """
        if not isinstance(event, OrderEvent):
            return

        order: OrderEvent = event

        if order.order_type == OrderType.LIMIT:
            print("Warning: LIMIT orders not supported in MVP, treating as MARKET")

        # Get current price
        bars = data_handler.get_latest_bars(order.symbol, 1)
        if bars.empty:
            print(f"Warning: No data for {order.symbol}, order rejected")
            return

        base_price = float(bars["close"].iloc[-1])

        # Calculate fill price with slippage
        fill_price = self._apply_slippage(base_price, order.side)

        # Calculate commission
        trade_value = fill_price * order.quantity
        commission = trade_value * self.commission_pct

        # Track costs
        self.total_commission += commission
        slippage_cost = abs(fill_price - base_price) * order.quantity
        self.total_slippage_cost += slippage_cost

        # Create fill event
        fill = FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
        )

        self.orders_processed += 1

        if self.events:
            self.events.put(fill)

        print(
            f"[{order.timestamp}] FILL: {order.side.value} "
            f"{order.quantity} {order.symbol} @ {fill_price:.2f} "
            f"(commission: ${commission:.2f})"
        )

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to price.

        BUY orders: price increases (worse fill)
        SELL orders: price decreases (worse fill)

        Args:
            price: Base price before slippage
            side: Order side (BUY or SELL)

        Returns:
            Price adjusted for slippage
        """
        if side == OrderSide.BUY:
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)

    def get_statistics(self) -> dict[str, float | int]:
        """Return execution statistics.

        Returns:
            Dictionary with orders_processed, total_commission, total_slippage_cost
        """
        return {
            "orders_processed": self.orders_processed,
            "total_commission": self.total_commission,
            "total_slippage_cost": self.total_slippage_cost,
        }

    def reset(self) -> None:
        """Reset execution statistics."""
        self.orders_processed = 0
        self.total_commission = 0.0
        self.total_slippage_cost = 0.0

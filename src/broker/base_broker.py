"""Abstract broker adapter interface.

Defines the contract for all broker implementations (paper, Alpaca, IBKR).
The adapter translates between our internal OrderEvent/FillEvent types
and the broker's API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto


class OrderStatus(Enum):
    """Lifecycle states for a broker order."""

    PENDING = auto()
    SUBMITTED = auto()
    ACCEPTED = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


@dataclass
class BrokerOrder:
    """Internal representation of an order submitted to a broker.

    Attributes:
        order_id: Unique identifier (assigned by broker adapter).
        symbol: Instrument to trade.
        side: "BUY" or "SELL".
        order_type: "MARKET" or "LIMIT".
        quantity: Number of shares/units.
        limit_price: Limit price (None for market orders).
        status: Current lifecycle state.
        submitted_at: When the order was submitted.
        filled_at: When the order was fully filled (None if not yet).
        filled_quantity: Cumulative filled quantity.
        filled_avg_price: Volume-weighted average fill price.
        broker_order_id: External order ID from the broker (if any).
    """

    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: int
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    filled_quantity: int = 0
    filled_avg_price: float = 0.0
    broker_order_id: str | None = None


@dataclass(frozen=True)
class BrokerFill:
    """A fill notification from the broker.

    Attributes:
        order_id: Our internal order ID.
        symbol: Instrument filled.
        side: "BUY" or "SELL".
        quantity: Quantity filled in this notification.
        price: Execution price.
        commission: Commission charged.
        timestamp: Fill timestamp.
    """

    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    timestamp: datetime


class BrokerAdapter(ABC):
    """Abstract interface for broker interactions.

    All broker implementations (paper, live) must implement these methods.
    The adapter handles order submission, cancellation, and status queries.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the broker."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        ...

    @abstractmethod
    async def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        """Submit an order to the broker.

        Returns the order with updated status and broker_order_id.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel a pending/submitted order.

        Returns the order with updated status.
        """
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> BrokerOrder:
        """Query the current status of an order."""
        ...

    @abstractmethod
    async def get_positions(self) -> dict[str, dict[str, float]]:
        """Get current positions from the broker.

        Returns: {symbol: {quantity, avg_cost, market_value}}
        """
        ...

    @abstractmethod
    async def get_account_info(self) -> dict[str, float]:
        """Get account information (cash, equity, buying power).

        Returns: {cash, equity, buying_power}
        """
        ...

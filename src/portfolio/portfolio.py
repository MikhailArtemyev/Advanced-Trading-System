"""Portfolio management for the backtesting engine.

Handles position tracking, order generation, fill processing, and P&L calculation.

Event Flow:
    Portfolio receives SignalEvent -> generates OrderEvent
    Portfolio receives FillEvent -> updates positions
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from ..data.data_handler import DataHandler
from ..events.event import (
    Event,
    FillEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalEvent,
    SignalType,
)
from ..events.queue import EventQueue
from ..risk.position_sizer import FixedFractionSizer, PositionSizer
from .correlation import CorrelationTracker


@dataclass
class Position:
    """Represents a position in a single symbol.

    Attributes:
        symbol: The instrument symbol
        quantity: Number of shares held (positive for long, negative for short)
        avg_cost: Average cost per share
        market_value: Current market value of position
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss from closed trades
    """

    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Record of a completed trade.

    Attributes:
        timestamp: When the trade occurred
        symbol: The instrument symbol
        side: BUY or SELL
        quantity: Number of shares traded
        price: Fill price
        commission: Commission paid
        pnl: Realized P&L (for sells only)
    """

    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    pnl: float = 0.0


class Portfolio:
    """Portfolio management handling position tracking, order generation, and P&L.

    Receives SignalEvents and generates OrderEvents.
    Receives FillEvents and updates positions.

    Position Sizing:
        Uses fixed percentage of equity for each position.
        Default is 10% of equity per position.

    Attributes:
        initial_capital: Starting cash amount
        cash: Current cash available
        commission_pct: Commission as percentage of trade value
        position_size_pct: Target position size as percentage of equity
        positions: Dictionary of Position objects by symbol
        trades: List of completed trades
        equity_history: List of equity snapshots over time
        current_prices: Dictionary of latest prices by symbol
    """

    def __init__(
        self,
        initial_capital: float,
        symbols: list[str],
        commission_pct: float = 0.001,
        position_size_pct: float = 0.1,
        position_sizer: PositionSizer | None = None,
        correlation_tracker: CorrelationTracker | None = None,
    ) -> None:
        """Initialize the portfolio.

        Args:
            initial_capital: Starting cash amount
            symbols: List of tradeable symbols
            commission_pct: Commission as percentage of trade value (default 0.1%)
            position_size_pct: Target position size as % of equity (default 10%)
            position_sizer: Optional custom position sizer. If None, uses
                           FixedFractionSizer with the given position_size_pct.
            correlation_tracker: Optional correlation tracker for multi-asset
                                monitoring. If None, no correlation tracking.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_pct = commission_pct
        self.position_size_pct = position_size_pct
        self._custom_sizer = position_sizer is not None
        self.position_sizer = position_sizer or FixedFractionSizer(
            fraction=position_size_pct
        )

        self.events: EventQueue | None = None

        # Positions for each symbol
        self.positions: dict[str, Position] = {s: Position(symbol=s) for s in symbols}

        # Trade history
        self.trades: list[Trade] = []

        # Equity curve tracking
        self.equity_history: list[dict[str, Any]] = []

        # Current prices for valuation
        self.current_prices: dict[str, float] = {}

        # Optional correlation tracker
        self.correlation_tracker = correlation_tracker

    def set_event_queue(self, events: EventQueue) -> None:
        """Set event queue for emitting orders.

        Args:
            events: The event queue to use for emitting OrderEvents
        """
        self.events = events

    def on_signal(self, event: Event, data_handler: DataHandler) -> None:
        """Process signal event and generate orders.

        Position sizing uses fixed percentage of equity.

        Args:
            event: The signal event to process
            data_handler: Data handler for getting current prices
        """
        if not isinstance(event, SignalEvent):
            return

        signal: SignalEvent = event
        symbol = signal.symbol

        # Get current price
        bars = data_handler.get_latest_bars(symbol, 1)
        if bars.empty:
            return

        current_price = float(bars["close"].iloc[-1])
        self.current_prices[symbol] = current_price

        # Calculate order
        order = self._generate_order(signal, current_price, data_handler)

        if order and self.events:
            self.events.put(order)

    def _generate_order(
        self,
        signal: SignalEvent,
        current_price: float,
        data_handler: DataHandler | None = None,
    ) -> OrderEvent | None:
        """Generate order from signal.

        Args:
            signal: The signal event
            current_price: Current market price
            data_handler: Optional data handler for position sizer

        Returns:
            OrderEvent or None if order cannot be generated
        """
        symbol = signal.symbol
        position = self.positions[symbol]

        if signal.signal_type == SignalType.LONG:
            # Sync default sizer if position_size_pct was changed after init
            if (
                not self._custom_sizer
                and isinstance(self.position_sizer, FixedFractionSizer)
                and self.position_sizer.fraction != self.position_size_pct
            ):
                self.position_sizer = FixedFractionSizer(
                    fraction=self.position_size_pct
                )

            # Delegate position sizing to the sizer
            equity = self.get_equity()
            result = self.position_sizer.calculate_size(
                symbol=symbol,
                current_price=current_price,
                equity=equity,
                cash=self.cash,
                data_handler=data_handler,
                signal_strength=signal.strength,
                trades=self.trades,
            )
            quantity = result.quantity

            if quantity <= 0:
                return None

            # Check if we have enough cash
            cost = quantity * current_price * (1 + self.commission_pct)
            if cost > self.cash:
                quantity = int(self.cash / (current_price * (1 + self.commission_pct)))

            if quantity <= 0:
                return None

            return OrderEvent(
                timestamp=signal.timestamp,
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=quantity,
            )

        elif signal.signal_type == SignalType.EXIT:
            if position.quantity > 0:
                # Close long position
                return OrderEvent(
                    timestamp=signal.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                )
            elif position.quantity < 0:
                # Close short position (buy to cover)
                return OrderEvent(
                    timestamp=signal.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=abs(position.quantity),
                )
            else:
                return None

        elif signal.signal_type == SignalType.SHORT:
            # Only open short when flat (no existing position)
            if position.quantity != 0:
                return None

            # Sync default sizer if position_size_pct was changed after init
            if (
                not self._custom_sizer
                and isinstance(self.position_sizer, FixedFractionSizer)
                and self.position_sizer.fraction != self.position_size_pct
            ):
                self.position_sizer = FixedFractionSizer(
                    fraction=self.position_size_pct
                )

            equity = self.get_equity()
            result = self.position_sizer.calculate_size(
                symbol=symbol,
                current_price=current_price,
                equity=equity,
                cash=self.cash,
                data_handler=data_handler,
                signal_strength=signal.strength,
                trades=self.trades,
            )
            quantity = result.quantity

            if quantity <= 0:
                return None

            return OrderEvent(
                timestamp=signal.timestamp,
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=OrderSide.SELL,
                quantity=quantity,
            )

        return None

    def on_fill(self, event: Event) -> None:
        """Process fill event and update position.

        Args:
            event: The fill event to process
        """
        if not isinstance(event, FillEvent):
            return

        fill: FillEvent = event
        symbol = fill.symbol
        position = self.positions[symbol]
        pnl = 0.0

        if fill.side == OrderSide.BUY:
            if position.quantity < 0:
                # Covering a short position
                cover_qty = min(fill.quantity, abs(position.quantity))
                pnl = (position.avg_cost - fill.fill_price) * cover_qty
                pnl -= fill.commission
                position.realized_pnl += pnl
                position.quantity += fill.quantity
                self.cash -= fill.quantity * fill.fill_price + fill.commission

                if position.quantity > 0:
                    # Flipped from short to long
                    position.avg_cost = fill.fill_price
                elif position.quantity == 0:
                    position.avg_cost = 0.0
                # If still short, avg_cost stays the same
            else:
                # Adding to long or opening new long
                total_cost = (
                    position.quantity * position.avg_cost
                    + fill.quantity * fill.fill_price
                )
                new_quantity = position.quantity + fill.quantity
                position.avg_cost = total_cost / new_quantity if new_quantity > 0 else 0
                position.quantity = new_quantity
                self.cash -= fill.quantity * fill.fill_price + fill.commission

        elif fill.side == OrderSide.SELL:
            if position.quantity > 0:
                # Closing/reducing a long position
                pnl = (fill.fill_price - position.avg_cost) * fill.quantity
                pnl -= fill.commission
                position.realized_pnl += pnl
                position.quantity -= fill.quantity
                self.cash += fill.quantity * fill.fill_price - fill.commission

                if position.quantity == 0:
                    position.avg_cost = 0.0
            else:
                # Opening or adding to a short position
                if position.quantity == 0:
                    position.avg_cost = fill.fill_price
                else:
                    # Weighted average for adding to short
                    total_cost = (
                        abs(position.quantity) * position.avg_cost
                        + fill.quantity * fill.fill_price
                    )
                    new_abs_qty = abs(position.quantity) + fill.quantity
                    position.avg_cost = total_cost / new_abs_qty
                position.quantity -= fill.quantity
                self.cash += fill.quantity * fill.fill_price - fill.commission

        # Record trade
        trade = Trade(
            timestamp=fill.timestamp,
            symbol=symbol,
            side=fill.side.value,
            quantity=fill.quantity,
            price=fill.fill_price,
            commission=fill.commission,
            pnl=pnl,
        )
        self.trades.append(trade)

    def update_timeindex(self, timestamp: datetime) -> None:
        """Update portfolio valuation at current timestamp.

        Records equity snapshot for equity curve tracking.

        Args:
            timestamp: Current simulation timestamp
        """
        # Update unrealized P&L for all positions
        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in self.current_prices:
                current_price = self.current_prices[symbol]
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (
                    current_price - position.avg_cost
                ) * position.quantity
            else:
                position.market_value = 0.0
                position.unrealized_pnl = 0.0

        # Update correlation tracker with latest prices
        if self.correlation_tracker is not None:
            for symbol in self.current_prices:
                self.correlation_tracker.update(symbol, self.current_prices[symbol])

        equity = self.get_equity()

        self.equity_history.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "cash": self.cash,
                "positions_value": equity - self.cash,
                "num_positions": sum(
                    1 for p in self.positions.values() if p.quantity != 0
                ),
            }
        )

    def get_equity(self) -> float:
        """Calculate total portfolio equity.

        Returns:
            Total equity (cash + positions market value)
        """
        positions_value = sum(
            p.quantity * self.current_prices.get(p.symbol, p.avg_cost)
            for p in self.positions.values()
        )
        return self.cash + positions_value

    def get_positions(self) -> dict[str, Any]:
        """Return current positions.

        Returns:
            Dictionary of Position objects by symbol
        """
        return {
            symbol: {
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
            }
            for symbol, pos in self.positions.items()
        }

    def get_trade_history(self) -> list[Any]:
        """Return trade history.

        Returns:
            List of Trade objects
        """
        return self.trades.copy()

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Return aggregate portfolio statistics.

        Returns:
            Dictionary with long/short exposure, net/gross exposure,
            position counts, cash, and equity.
        """
        long_exposure = 0.0
        short_exposure = 0.0
        long_count = 0
        short_count = 0

        for pos in self.positions.values():
            if pos.quantity > 0:
                price = self.current_prices.get(pos.symbol, pos.avg_cost)
                long_exposure += pos.quantity * price
                long_count += 1
            elif pos.quantity < 0:
                price = self.current_prices.get(pos.symbol, pos.avg_cost)
                short_exposure += abs(pos.quantity) * price
                short_count += 1

        return {
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "net_exposure": long_exposure - short_exposure,
            "gross_exposure": long_exposure + short_exposure,
            "long_count": long_count,
            "short_count": short_count,
            "num_positions": long_count + short_count,
            "cash": self.cash,
            "equity": self.get_equity(),
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as DataFrame.

        Returns:
            DataFrame with timestamp, equity, cash, positions_value,
            and num_positions columns.
        """
        return pd.DataFrame(self.equity_history)

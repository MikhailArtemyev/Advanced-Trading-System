"""Pre-trade risk management for the backtesting engine.

Provides configurable risk checks that sit between the Portfolio
(which generates orders) and the ExecutionHandler (which fills them).

Risk checks performed (in priority order):
1. Circuit breaker (max drawdown halt)
2. Daily loss limit
3. Order rate limit
4. Open position count limit
5. Position concentration limit (can reduce quantity)
6. Total portfolio exposure limit (can reduce quantity)
7. Correlation check (placeholder for future use)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..events.event import OrderEvent, OrderSide


@dataclass(frozen=True)
class RiskLimits:
    """Configurable risk limits for pre-trade checks.

    Attributes:
        max_position_pct: Max single position as fraction of equity
        max_portfolio_exposure_pct: Max total exposure as fraction of equity
        max_daily_loss_pct: Max daily loss as fraction of equity
        max_drawdown_pct: Max drawdown from peak before trading halt
        max_open_positions: Max simultaneous open positions
        max_orders_per_day: Max orders per calendar day
    """

    max_position_pct: float = 0.10
    max_portfolio_exposure_pct: float = 1.0
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_open_positions: int = 20
    max_orders_per_day: int = 100


@dataclass
class RiskCheckResult:
    """Result of a pre-trade risk check.

    Attributes:
        approved: Whether the order passed all checks
        order: The original OrderEvent
        adjusted_quantity: New quantity if reduced, None if unchanged
        rejection_reasons: List of reasons the order was rejected
        warnings: List of non-blocking warnings
    """

    approved: bool
    order: OrderEvent
    adjusted_quantity: int | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RiskManager:
    """Pre-trade risk manager checking orders against configurable limits.

    Performs 7 checks in priority order. Checks 1-4 are hard rejects.
    Checks 5-6 can adjust order quantity downward. Check 7 is a placeholder.

    Only approved orders increment the daily order counter.

    Attributes:
        limits: Risk limit configuration
        daily_pnl: Realized P&L for the current trading day
        current_date: Date of the current trading day
        orders_today: Number of approved orders on current day
        peak_equity: Highest equity observed (for drawdown calculation)
        is_halted: True if max drawdown breached (circuit breaker)
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        """Initialize the risk manager.

        Args:
            limits: Risk limit configuration. Uses defaults if None.
        """
        self.limits = limits or RiskLimits()
        self.daily_pnl: float = 0.0
        self.current_date: date | None = None
        self.orders_today: int = 0
        self.peak_equity: float = 0.0
        self.is_halted: bool = False

    def check_order(
        self,
        order: OrderEvent,
        equity: float,
        positions: dict[str, Any],
        current_price: float,
    ) -> RiskCheckResult:
        """Check an order against all risk limits.

        Args:
            order: The order to validate
            equity: Current portfolio equity
            positions: Current positions from portfolio.get_positions()
            current_price: Current market price of the order's symbol

        Returns:
            RiskCheckResult with approval status and any adjustments
        """
        rejection_reasons: list[str] = []
        warnings: list[str] = []

        if equity <= 0:
            return RiskCheckResult(
                approved=False,
                order=order,
                rejection_reasons=["Portfolio equity is zero or negative"],
            )

        # Check 1: Circuit breaker (max drawdown halt)
        if self.is_halted:
            return RiskCheckResult(
                approved=False,
                order=order,
                rejection_reasons=["Trading halted: max drawdown breached"],
            )

        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - equity) / self.peak_equity
            if current_drawdown >= self.limits.max_drawdown_pct:
                self.is_halted = True
                return RiskCheckResult(
                    approved=False,
                    order=order,
                    rejection_reasons=[
                        f"Max drawdown breached: {current_drawdown:.2%} >= "
                        f"{self.limits.max_drawdown_pct:.2%}"
                    ],
                )

        # Check 2: Daily loss limit
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / equity
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                rejection_reasons.append(
                    f"Daily loss limit: {daily_loss_pct:.2%} >= "
                    f"{self.limits.max_daily_loss_pct:.2%}"
                )

        # Check 3: Order rate limit
        if self.orders_today >= self.limits.max_orders_per_day:
            rejection_reasons.append(
                f"Order rate limit: {self.orders_today} >= "
                f"{self.limits.max_orders_per_day}"
            )

        # Check 4: Open position count (BUY only)
        if order.side == OrderSide.BUY:
            open_count = self._count_open_positions(positions)
            # Check if this is a new position (no existing holding)
            existing_qty = self._get_position_quantity(positions, order.symbol)
            if existing_qty == 0 and open_count >= self.limits.max_open_positions:
                rejection_reasons.append(
                    f"Max open positions: {open_count} >= "
                    f"{self.limits.max_open_positions}"
                )

        # Early return if any hard rejections from checks 1-4
        if rejection_reasons:
            return RiskCheckResult(
                approved=False,
                order=order,
                rejection_reasons=rejection_reasons,
                warnings=warnings,
            )

        # Check 5: Position concentration (BUY only, can adjust)
        working_quantity = order.quantity
        if order.side == OrderSide.BUY and current_price > 0:
            existing_value = (
                self._get_position_quantity(positions, order.symbol) * current_price
            )
            proposed_value = existing_value + working_quantity * current_price
            max_value = equity * self.limits.max_position_pct

            if proposed_value > max_value:
                allowed_add = max_value - existing_value
                if allowed_add <= 0:
                    rejection_reasons.append(
                        f"Position concentration: {order.symbol} at limit"
                    )
                else:
                    new_qty = int(allowed_add / current_price)
                    if new_qty <= 0:
                        rejection_reasons.append(
                            "Position concentration: adjusted quantity is 0"
                        )
                    else:
                        warnings.append(
                            f"Position reduced from {order.quantity} to "
                            f"{new_qty} (concentration limit)"
                        )
                        working_quantity = new_qty

        # Check 6: Total portfolio exposure (BUY only, can adjust)
        if order.side == OrderSide.BUY and current_price > 0 and not rejection_reasons:
            total_exposure = self._get_total_exposure(positions)
            proposed_exposure = total_exposure + working_quantity * current_price
            max_exposure = equity * self.limits.max_portfolio_exposure_pct

            if proposed_exposure > max_exposure:
                headroom = max_exposure - total_exposure
                if headroom <= 0:
                    rejection_reasons.append(
                        "Portfolio exposure limit: no room for new exposure"
                    )
                else:
                    new_qty = int(headroom / current_price)
                    if new_qty <= 0:
                        rejection_reasons.append(
                            "Portfolio exposure limit: adjusted quantity is 0"
                        )
                    else:
                        warnings.append(
                            f"Exposure reduced to {new_qty} (exposure limit)"
                        )
                        working_quantity = new_qty

        # Check 7: Correlation check (placeholder)
        # Infrastructure for future use when correlation matrix is available.

        if rejection_reasons:
            return RiskCheckResult(
                approved=False,
                order=order,
                rejection_reasons=rejection_reasons,
                warnings=warnings,
            )

        # Order approved — track it
        self.orders_today += 1
        adjusted = working_quantity if working_quantity != order.quantity else None

        return RiskCheckResult(
            approved=True,
            order=order,
            adjusted_quantity=adjusted,
            warnings=warnings,
        )

    def update_daily_pnl(self, pnl: float, timestamp: datetime) -> None:
        """Update the daily realized P&L tracker.

        Handles day rollover: resets counters when a new day is detected.

        Args:
            pnl: Realized P&L from this fill
            timestamp: Timestamp of the fill event
        """
        ts_date = timestamp.date()
        if self.current_date is None or ts_date != self.current_date:
            self.daily_pnl = 0.0
            self.orders_today = 0
            self.current_date = ts_date
        self.daily_pnl += pnl

    def update_peak_equity(self, equity: float) -> None:
        """Update peak equity for drawdown tracking.

        Args:
            equity: Current portfolio equity
        """
        if equity > self.peak_equity:
            self.peak_equity = equity

    def reset_halt(self) -> None:
        """Manually reset the circuit breaker halt state."""
        self.is_halted = False

    def reset(self) -> None:
        """Reset all state for a new backtest run."""
        self.daily_pnl = 0.0
        self.current_date = None
        self.orders_today = 0
        self.peak_equity = 0.0
        self.is_halted = False

    def _count_open_positions(self, positions: dict[str, Any]) -> int:
        """Count positions with quantity > 0."""
        count = 0
        for p in positions.values():
            qty = self._extract_quantity(p)
            if qty > 0:
                count += 1
        return count

    def _get_position_quantity(self, positions: dict[str, Any], symbol: str) -> int:
        """Get quantity for a specific symbol from positions dict."""
        p = positions.get(symbol)
        if p is None:
            return 0
        return self._extract_quantity(p)

    def _get_total_exposure(self, positions: dict[str, Any]) -> float:
        """Sum market value of all positions."""
        total = 0.0
        for p in positions.values():
            if isinstance(p, dict):
                total += abs(p.get("market_value", 0.0))
            else:
                total += 0.0
        return total

    def _extract_quantity(self, position: Any) -> int:
        """Extract quantity from a position (dict or int)."""
        if isinstance(position, dict):
            return int(position.get("quantity", 0))
        if isinstance(position, int):
            return position
        return int(getattr(position, "quantity", 0))

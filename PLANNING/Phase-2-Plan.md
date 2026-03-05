# Phase 2: Portfolio Engine & Risk Management
## Step-by-Step Implementation Guide

**Duration:** 8 Weeks (Months 3-4)
**Goal:** Multi-asset portfolio management with proper position sizing, risk controls, and basic portfolio optimization

**Prerequisites:** Phase 1 complete — working event-driven backtester with single strategy capability

---

## Overview

```
Week 1-2: Position Sizing Algorithms & Risk Manager Foundation
Week 3-4: Enhanced Portfolio (Multi-Asset, Correlation Tracking)
Week 5-6: Portfolio Optimization (Mean-Variance, Risk Parity)
Week 7:   Walk-Forward Testing Framework & Multi-Asset Strategies
Week 8:   Integration Testing, Validation & Documentation
```

---

## What Changes From Phase 1

Phase 1 established the event-driven backbone. Phase 2 builds on it without breaking existing functionality:

| Component | Phase 1 State | Phase 2 Target |
|-----------|--------------|----------------|
| Portfolio | Fixed 10% position sizing, single-symbol focus | Dynamic sizing (Kelly, vol-based), multi-asset correlation |
| Risk | None | Pre-trade checks, position limits, drawdown controls |
| Optimization | None | Mean-variance, risk parity, configurable allocator |
| Strategy | SMA Crossover only | Multi-asset strategy support, short selling |
| Execution | Simple fill model | Limit order support, partial fill handling |
| Data | CSV only | yfinance integration for live downloads |
| Config | Basic YAML | Extended with risk/portfolio/optimization sections |
| Testing | 400+ tests | Target 600+ with risk and optimization coverage |

### New Project Structure (additions to Phase 1)

```
src/
├── ... (existing Phase 1 modules)
├── risk/
│   ├── __init__.py
│   ├── risk_manager.py        # Pre-trade checks, position limits
│   └── position_sizer.py      # Kelly, vol-based, fixed-fraction sizing
├── optimization/
│   ├── __init__.py
│   ├── base_optimizer.py       # Abstract optimizer interface
│   ├── mean_variance.py        # Mean-variance optimization
│   └── risk_parity.py          # Risk parity allocation
├── strategy/
│   ├── ... (existing)
│   └── multi_asset_sma.py      # Multi-asset SMA strategy
tests/
├── ... (existing Phase 1 tests)
├── test_risk_manager.py
├── test_position_sizer.py
├── test_optimizer.py
├── test_multi_asset.py
└── test_walk_forward.py
configs/
├── backtest_config.yaml        # Updated with new sections
└── risk_config.yaml            # Standalone risk configuration
```

---

## Week 1: Position Sizing Algorithms

### Step 1.1: Implement Position Sizer Module
**Time:** Day 1-3

**File: `src/risk/position_sizer.py`**
```python
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SizingResult:
    """Result of position sizing calculation"""
    symbol: str
    quantity: int
    target_value: float
    method: str
    risk_per_share: float = 0.0
    notes: str = ""


class PositionSizer(ABC):
    """Abstract base class for position sizing algorithms"""

    @abstractmethod
    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        equity: float,
        current_positions: Dict[str, int],
        market_data: Optional[pd.DataFrame] = None,
    ) -> SizingResult:
        """
        Calculate position size for a trade.

        Args:
            symbol: Instrument to trade
            current_price: Current market price
            equity: Total portfolio equity
            current_positions: Dict of symbol -> quantity
            market_data: Recent price history for volatility calc

        Returns:
            SizingResult with recommended quantity
        """
        raise NotImplementedError


class FixedFractionSizer(PositionSizer):
    """
    Fixed fraction of equity per position.

    This is the Phase 1 approach, extracted into its own class
    for consistency with the new sizing framework.
    """

    def __init__(self, fraction: float = 0.10):
        """
        Args:
            fraction: Fraction of equity to allocate (e.g., 0.10 = 10%)
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Fraction must be in (0, 1], got {fraction}")
        self.fraction = fraction

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        equity: float,
        current_positions: Dict[str, int],
        market_data: Optional[pd.DataFrame] = None,
    ) -> SizingResult:
        target_value = equity * self.fraction
        quantity = int(target_value / current_price)

        return SizingResult(
            symbol=symbol,
            quantity=max(quantity, 0),
            target_value=target_value,
            method="fixed_fraction",
            notes=f"fraction={self.fraction}",
        )


class VolatilityBasedSizer(PositionSizer):
    """
    Position sizing based on ATR (Average True Range).

    Size = (Equity * risk_pct) / (ATR * atr_multiplier)

    Larger positions in low-volatility environments,
    smaller positions in high-volatility environments.
    """

    def __init__(
        self,
        risk_pct: float = 0.01,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        """
        Args:
            risk_pct: Max risk per trade as fraction of equity (e.g., 0.01 = 1%)
            atr_period: Lookback for ATR calculation
            atr_multiplier: ATR multiplier for stop distance
        """
        self.risk_pct = risk_pct
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        equity: float,
        current_positions: Dict[str, int],
        market_data: Optional[pd.DataFrame] = None,
    ) -> SizingResult:
        if market_data is None or len(market_data) < self.atr_period:
            # Fallback to fixed fraction if insufficient data
            fallback_qty = int((equity * self.risk_pct * 10) / current_price)
            return SizingResult(
                symbol=symbol,
                quantity=max(fallback_qty, 0),
                target_value=equity * self.risk_pct * 10,
                method="volatility_based",
                notes="insufficient data, used fallback",
            )

        atr = self._calculate_atr(market_data)
        risk_per_share = atr * self.atr_multiplier
        dollar_risk = equity * self.risk_pct
        quantity = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0
        target_value = quantity * current_price

        return SizingResult(
            symbol=symbol,
            quantity=max(quantity, 0),
            target_value=target_value,
            method="volatility_based",
            risk_per_share=risk_per_share,
            notes=f"ATR={atr:.4f}, risk/share={risk_per_share:.4f}",
        )

    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.tail(self.atr_period).mean()


class KellyCriterionSizer(PositionSizer):
    """
    Fractional Kelly Criterion position sizing.

    Full Kelly: f* = (p * b - q) / b
    Where:
        p = win probability
        b = win/loss ratio
        q = 1 - p

    We use fractional Kelly (default 0.25) to reduce drawdowns.
    Full Kelly can experience 50-70% drawdowns.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        lookback_trades: int = 50,
        max_position_pct: float = 0.20,
        default_win_rate: float = 0.50,
        default_win_loss_ratio: float = 1.5,
    ):
        """
        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
            lookback_trades: Number of past trades for estimating win rate
            max_position_pct: Maximum position as % of equity (safety cap)
            default_win_rate: Win rate to use when no trade history available
            default_win_loss_ratio: Win/loss ratio when no trade history
        """
        self.kelly_fraction = kelly_fraction
        self.lookback_trades = lookback_trades
        self.max_position_pct = max_position_pct
        self.default_win_rate = default_win_rate
        self.default_win_loss_ratio = default_win_loss_ratio

        # Trade history will be fed externally
        self._trade_history: list = []

    def update_trade_history(self, trades: list) -> None:
        """Update with recent trades for Kelly estimation"""
        self._trade_history = trades

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        equity: float,
        current_positions: Dict[str, int],
        market_data: Optional[pd.DataFrame] = None,
    ) -> SizingResult:
        win_rate, win_loss_ratio = self._estimate_parameters()
        kelly_pct = self._kelly_formula(win_rate, win_loss_ratio)

        # Apply fractional Kelly
        position_pct = kelly_pct * self.kelly_fraction

        # Safety cap
        position_pct = min(position_pct, self.max_position_pct)
        position_pct = max(position_pct, 0.0)

        target_value = equity * position_pct
        quantity = int(target_value / current_price)

        return SizingResult(
            symbol=symbol,
            quantity=max(quantity, 0),
            target_value=target_value,
            method="kelly_criterion",
            notes=(
                f"kelly_f={kelly_pct:.4f}, "
                f"fractional={position_pct:.4f}, "
                f"win_rate={win_rate:.2f}, "
                f"wl_ratio={win_loss_ratio:.2f}"
            ),
        )

    def _estimate_parameters(self) -> tuple:
        """Estimate win rate and win/loss ratio from trade history"""
        recent = self._trade_history[-self.lookback_trades :]

        # Filter to closed trades (sells)
        closed = [t for t in recent if t.side == "SELL" and t.pnl != 0]

        if len(closed) < 10:
            return self.default_win_rate, self.default_win_loss_ratio

        winners = [t for t in closed if t.pnl > 0]
        losers = [t for t in closed if t.pnl < 0]

        win_rate = len(winners) / len(closed)

        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = abs(np.mean([t.pnl for t in losers])) if losers else 1

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else self.default_win_loss_ratio

        return win_rate, win_loss_ratio

    def _kelly_formula(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Kelly Criterion: f* = (p * b - q) / b
        """
        p = win_rate
        q = 1 - p
        b = win_loss_ratio

        if b <= 0:
            return 0.0

        kelly = (p * b - q) / b
        return max(kelly, 0.0)
```

**Tasks:**
- [ ] Create `src/risk/__init__.py`
- [ ] Implement `PositionSizer` abstract base class
- [ ] Implement `FixedFractionSizer` (refactor from Portfolio)
- [ ] Implement `VolatilityBasedSizer` with ATR calculation
- [ ] Implement `KellyCriterionSizer` with trade history estimation
- [ ] Write unit tests for each sizer with known inputs/outputs
- [ ] Test ATR calculation against manual computation
- [ ] Test Kelly formula edge cases (no history, all wins, all losses)

**Deliverable:** Three position sizing algorithms with full test coverage

---

### Step 1.2: Integrate Position Sizer into Portfolio
**Time:** Day 3-5

Refactor `Portfolio.on_signal()` to delegate sizing to the `PositionSizer` instead of hardcoded logic.

**Changes to `src/portfolio/portfolio.py`:**
```python
# Add to __init__:
from ..risk.position_sizer import PositionSizer, FixedFractionSizer

class Portfolio:
    def __init__(
        self,
        initial_capital: float,
        symbols: list,
        commission_pct: float = 0.001,
        position_sizer: Optional[PositionSizer] = None,
    ):
        # ... existing init ...
        self.position_sizer = position_sizer or FixedFractionSizer(fraction=0.10)

    def on_signal(self, signal: SignalEvent, data_handler: DataHandler) -> None:
        """Process signal event and generate orders using position sizer"""
        symbol = signal.symbol
        bars = data_handler.get_latest_bars(symbol, 1)
        if bars.empty:
            return

        current_price = bars["close"].iloc[-1]
        self.current_prices[symbol] = current_price

        order = self._generate_order(signal, current_price, data_handler)
        if order:
            self.events.put(order)

    def _generate_order(
        self, signal, current_price, data_handler
    ) -> Optional[OrderEvent]:
        """Generate order from signal using position sizer"""
        symbol = signal.symbol
        position = self.positions[symbol]

        if signal.signal_type == SignalType.LONG:
            # Get market data for vol-based sizing
            market_data = data_handler.get_latest_bars(symbol, 60)

            sizing = self.position_sizer.calculate_size(
                symbol=symbol,
                current_price=current_price,
                equity=self.get_equity(),
                current_positions={
                    s: p.quantity for s, p in self.positions.items()
                },
                market_data=market_data,
            )

            quantity = sizing.quantity

            # Cash check
            cost = quantity * current_price * (1 + self.commission_pct)
            if cost > self.cash:
                quantity = int(
                    self.cash / (current_price * (1 + self.commission_pct))
                )

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
            if position.quantity <= 0:
                return None
            return OrderEvent(
                timestamp=signal.timestamp,
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=OrderSide.SELL,
                quantity=position.quantity,
            )

        elif signal.signal_type == SignalType.SHORT:
            # Phase 2: Enable short selling
            market_data = data_handler.get_latest_bars(symbol, 60)
            sizing = self.position_sizer.calculate_size(
                symbol=symbol,
                current_price=current_price,
                equity=self.get_equity(),
                current_positions={
                    s: p.quantity for s, p in self.positions.items()
                },
                market_data=market_data,
            )
            quantity = sizing.quantity
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
```

**Tasks:**
- [ ] Refactor `Portfolio.__init__` to accept `PositionSizer`
- [ ] Refactor `Portfolio._generate_order` to use sizer
- [ ] Enable SHORT signal handling in Portfolio
- [ ] Update Portfolio to handle negative positions (short selling)
- [ ] Ensure backwards compatibility (default to `FixedFractionSizer`)
- [ ] Update all existing Portfolio tests to pass
- [ ] Add new tests for each sizing method via Portfolio

**Deliverable:** Portfolio with pluggable position sizing

---

## Week 2: Risk Manager

### Step 2.1: Implement Risk Manager
**Time:** Day 1-4

**File: `src/risk/risk_manager.py`**
```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from ..events.event import OrderEvent, OrderSide


@dataclass
class RiskLimits:
    """Configuration for risk limits"""
    max_position_pct: float = 0.10        # Max single position as % of equity
    max_portfolio_exposure_pct: float = 1.0  # Max total exposure (1.0 = 100%)
    max_sector_pct: float = 0.30          # Max sector exposure
    max_correlation_threshold: float = 0.70  # Block if correlated > this
    max_daily_loss_pct: float = 0.03      # Daily loss limit (3%)
    max_drawdown_pct: float = 0.15        # Max drawdown before halt (15%)
    max_open_positions: int = 20          # Max simultaneous positions
    max_orders_per_day: int = 100         # Rate limit


@dataclass
class RiskCheckResult:
    """Result of a risk check"""
    approved: bool
    order: Optional[OrderEvent]
    adjusted_quantity: Optional[int] = None
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RiskManager:
    """
    Pre-trade risk management.

    Checks all orders against configurable limits before execution.
    Can reject, reduce, or approve orders.

    Risk checks performed:
    1. Position concentration limit
    2. Total portfolio exposure limit
    3. Daily loss limit
    4. Maximum drawdown halt
    5. Open position count limit
    6. Order rate limit
    7. Correlation check (optional)
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()

        # State tracking
        self.daily_pnl: float = 0.0
        self.current_date: Optional[datetime] = None
        self.orders_today: int = 0
        self.peak_equity: float = 0.0
        self.is_halted: bool = False
        self.halt_reason: str = ""

        # Correlation matrix (updated externally)
        self.correlation_matrix: Optional[pd.DataFrame] = None

    def check_order(
        self,
        order: OrderEvent,
        equity: float,
        cash: float,
        positions: Dict[str, "Position"],
        current_prices: Dict[str, float],
    ) -> RiskCheckResult:
        """
        Run all risk checks on an order.

        Returns:
            RiskCheckResult with approval status and any adjustments
        """
        reasons = []
        warnings = []

        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Reset daily counters on new day
        self._update_daily_tracking(order.timestamp)

        # 0. Check if trading is halted
        if self.is_halted:
            reasons.append(f"Trading halted: {self.halt_reason}")
            return RiskCheckResult(
                approved=False,
                order=order,
                rejection_reasons=reasons,
            )

        # 1. Max drawdown check
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown >= self.limits.max_drawdown_pct:
                self.is_halted = True
                self.halt_reason = (
                    f"Max drawdown {drawdown:.1%} >= "
                    f"{self.limits.max_drawdown_pct:.1%}"
                )
                reasons.append(self.halt_reason)
                return RiskCheckResult(
                    approved=False,
                    order=order,
                    rejection_reasons=reasons,
                )

        # 2. Daily loss limit
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / equity if equity > 0 else 0
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                reasons.append(
                    f"Daily loss limit: {daily_loss_pct:.1%} >= "
                    f"{self.limits.max_daily_loss_pct:.1%}"
                )

        # 3. Order rate limit
        if self.orders_today >= self.limits.max_orders_per_day:
            reasons.append(
                f"Order rate limit: {self.orders_today} >= "
                f"{self.limits.max_orders_per_day}"
            )

        # 4. Open position count (only for new positions)
        if order.side == OrderSide.BUY:
            open_count = sum(
                1 for p in positions.values() if p.quantity > 0
            )
            is_new_position = positions.get(
                order.symbol
            ) is None or positions[order.symbol].quantity == 0

            if is_new_position and open_count >= self.limits.max_open_positions:
                reasons.append(
                    f"Max open positions: {open_count} >= "
                    f"{self.limits.max_open_positions}"
                )

        # 5. Position concentration limit
        adjusted_qty = order.quantity
        if order.side == OrderSide.BUY and equity > 0:
            order_value = order.quantity * current_prices.get(
                order.symbol, 0
            )
            existing_value = (
                positions[order.symbol].quantity
                * current_prices.get(order.symbol, 0)
                if order.symbol in positions
                else 0
            )
            total_position_value = order_value + existing_value
            position_pct = total_position_value / equity

            if position_pct > self.limits.max_position_pct:
                max_value = equity * self.limits.max_position_pct - existing_value
                price = current_prices.get(order.symbol, 1)
                adjusted_qty = max(int(max_value / price), 0)
                if adjusted_qty == 0:
                    reasons.append(
                        f"Position limit: {position_pct:.1%} > "
                        f"{self.limits.max_position_pct:.1%}"
                    )
                else:
                    warnings.append(
                        f"Position reduced from {order.quantity} to "
                        f"{adjusted_qty} (concentration limit)"
                    )

        # 6. Total exposure limit
        if order.side == OrderSide.BUY and equity > 0:
            total_exposure = sum(
                abs(p.quantity) * current_prices.get(p.symbol, 0)
                for p in positions.values()
            )
            new_exposure = adjusted_qty * current_prices.get(
                order.symbol, 0
            )
            exposure_pct = (total_exposure + new_exposure) / equity

            if exposure_pct > self.limits.max_portfolio_exposure_pct:
                headroom = (
                    equity * self.limits.max_portfolio_exposure_pct
                    - total_exposure
                )
                price = current_prices.get(order.symbol, 1)
                adjusted_qty = max(int(headroom / price), 0)
                if adjusted_qty == 0:
                    reasons.append(
                        f"Exposure limit: {exposure_pct:.1%} > "
                        f"{self.limits.max_portfolio_exposure_pct:.1%}"
                    )
                else:
                    warnings.append(
                        f"Position reduced to {adjusted_qty} "
                        f"(exposure limit)"
                    )

        # 7. Correlation check
        if (
            self.correlation_matrix is not None
            and order.side == OrderSide.BUY
            and order.symbol in self.correlation_matrix.columns
        ):
            corr_warning = self._check_correlation(
                order.symbol, positions, current_prices
            )
            if corr_warning:
                warnings.append(corr_warning)

        # Determine result
        if reasons:
            return RiskCheckResult(
                approved=False,
                order=order,
                rejection_reasons=reasons,
                warnings=warnings,
            )

        self.orders_today += 1

        return RiskCheckResult(
            approved=True,
            order=order,
            adjusted_quantity=adjusted_qty if adjusted_qty != order.quantity else None,
            warnings=warnings,
        )

    def update_daily_pnl(self, pnl_change: float) -> None:
        """Update daily P&L tracker (called after each fill)"""
        self.daily_pnl += pnl_change

    def update_correlation_matrix(self, matrix: pd.DataFrame) -> None:
        """Update correlation matrix for correlation checks"""
        self.correlation_matrix = matrix

    def reset_halt(self) -> None:
        """Manually reset trading halt (requires human decision)"""
        self.is_halted = False
        self.halt_reason = ""

    def _update_daily_tracking(self, timestamp: datetime) -> None:
        """Reset daily counters on new trading day"""
        current = timestamp.date() if hasattr(timestamp, "date") else timestamp
        if self.current_date is None or current != self.current_date:
            self.current_date = current
            self.daily_pnl = 0.0
            self.orders_today = 0

    def _check_correlation(
        self,
        symbol: str,
        positions: Dict[str, "Position"],
        current_prices: Dict[str, float],
    ) -> Optional[str]:
        """Check if new position is highly correlated with existing"""
        corr = self.correlation_matrix
        held = [
            s for s, p in positions.items()
            if p.quantity != 0 and s in corr.columns and s != symbol
        ]

        high_corr = []
        for held_symbol in held:
            c = abs(corr.loc[symbol, held_symbol])
            if c >= self.limits.max_correlation_threshold:
                high_corr.append(f"{held_symbol}({c:.2f})")

        if high_corr:
            return (
                f"High correlation with: {', '.join(high_corr)} "
                f"(threshold={self.limits.max_correlation_threshold})"
            )
        return None
```

**Tasks:**
- [ ] Implement `RiskLimits` dataclass with sensible defaults
- [ ] Implement `RiskCheckResult` dataclass
- [ ] Implement `RiskManager` with all 7 risk checks
- [ ] Test position concentration limit with order reduction
- [ ] Test daily loss limit halts new trades
- [ ] Test max drawdown triggers halt
- [ ] Test order rate limiting
- [ ] Test correlation warnings
- [ ] Test daily counter reset on new day
- [ ] Test halt/reset mechanism

**Deliverable:** Complete pre-trade risk manager

---

### Step 2.2: Wire Risk Manager into Backtest Engine
**Time:** Day 4-5

The risk manager sits between the Portfolio (which generates orders) and the Execution Handler. Modify the event flow:

**Previous flow:**
```
Signal → Portfolio → OrderEvent → ExecutionHandler → Fill
```

**New flow:**
```
Signal → Portfolio → OrderEvent → RiskManager → (Approved?) → ExecutionHandler → Fill
```

**Changes to `src/backtest/engine.py`:**
```python
class BacktestEngine:
    def __init__(
        self,
        data_handler,
        strategy,
        portfolio,
        execution_handler,
        risk_manager=None,          # NEW
        performance_tracker=None,
    ):
        # ... existing ...
        self.risk_manager = risk_manager
        self.rejected_orders = 0     # NEW: track rejections

    def _handle_event(self, event) -> None:
        if event.event_type == EventType.MARKET:
            self.strategy.on_market_data(event, self.data_handler)

        elif event.event_type == EventType.SIGNAL:
            self.portfolio.on_signal(event, self.data_handler)

        elif event.event_type == EventType.ORDER:
            # NEW: Risk check before execution
            if self.risk_manager:
                result = self.risk_manager.check_order(
                    order=event,
                    equity=self.portfolio.get_equity(),
                    cash=self.portfolio.cash,
                    positions=self.portfolio.positions,
                    current_prices=self.portfolio.current_prices,
                )
                if not result.approved:
                    self.rejected_orders += 1
                    return
                if result.adjusted_quantity is not None:
                    event.quantity = result.adjusted_quantity

            self.execution_handler.execute_order(event, self.data_handler)

        elif event.event_type == EventType.FILL:
            self.portfolio.on_fill(event)
            self.strategy.update_position(
                event.symbol,
                self.portfolio.positions[event.symbol].quantity
            )
            # NEW: Update risk manager daily P&L
            if self.risk_manager and event.side == OrderSide.SELL:
                trade = self.portfolio.trades[-1] if self.portfolio.trades else None
                if trade:
                    self.risk_manager.update_daily_pnl(trade.pnl)
```

**Tasks:**
- [ ] Add optional `risk_manager` parameter to `BacktestEngine`
- [ ] Insert risk check in `_handle_event` for ORDER events
- [ ] Handle order quantity adjustment from risk manager
- [ ] Track rejected orders in results
- [ ] Update daily P&L in risk manager after fills
- [ ] Ensure engine works without risk manager (backwards compatible)
- [ ] Write integration tests with risk manager active

**Deliverable:** Risk-aware backtest engine

---

## Week 3: Enhanced Portfolio — Multi-Asset Support

### Step 3.1: Rolling Correlation Tracker
**Time:** Day 1-2

**File: `src/portfolio/correlation.py`** (new file)
```python
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class CorrelationTracker:
    """
    Tracks rolling correlations between assets.

    Uses rolling-window correlations to avoid look-ahead bias.
    Updates incrementally as new data arrives.
    """

    def __init__(self, window: int = 60, min_periods: int = 30):
        """
        Args:
            window: Rolling window size in bars (default: 60 trading days)
            min_periods: Minimum observations for valid correlation
        """
        self.window = window
        self.min_periods = min_periods
        self._returns_history: Dict[str, List[float]] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def update(self, symbol: str, price: float) -> None:
        """
        Record a price observation for a symbol.
        Call once per bar per symbol.
        """
        if symbol not in self._returns_history:
            self._returns_history[symbol] = [price]
            return

        prev_price = self._returns_history[symbol][-1]
        if prev_price > 0:
            ret = (price - prev_price) / prev_price
        else:
            ret = 0.0

        self._returns_history[symbol].append(price)

    def calculate_matrix(self) -> Optional[pd.DataFrame]:
        """
        Calculate rolling correlation matrix from recent returns.

        Returns:
            Correlation DataFrame or None if insufficient data
        """
        symbols = list(self._returns_history.keys())
        if len(symbols) < 2:
            return None

        # Build returns DataFrame
        returns_data = {}
        for symbol in symbols:
            prices = self._returns_history[symbol]
            if len(prices) < self.min_periods:
                continue
            # Calculate returns from prices
            price_series = pd.Series(prices)
            returns_data[symbol] = price_series.pct_change().dropna()

        if len(returns_data) < 2:
            return None

        df = pd.DataFrame(returns_data)

        # Use only the rolling window
        df = df.tail(self.window)

        if len(df) < self.min_periods:
            return None

        self._correlation_matrix = df.corr()
        return self._correlation_matrix

    def get_correlation(self, symbol_a: str, symbol_b: str) -> Optional[float]:
        """Get pairwise correlation between two symbols"""
        if self._correlation_matrix is None:
            return None
        if (
            symbol_a not in self._correlation_matrix.columns
            or symbol_b not in self._correlation_matrix.columns
        ):
            return None
        return self._correlation_matrix.loc[symbol_a, symbol_b]

    def get_matrix(self) -> Optional[pd.DataFrame]:
        """Return the most recently calculated correlation matrix"""
        return self._correlation_matrix
```

**Tasks:**
- [ ] Implement `CorrelationTracker` class
- [ ] Test with known correlated data (should show high correlation)
- [ ] Test with uncorrelated data
- [ ] Test with insufficient data (returns None)
- [ ] Test rolling window respects window size
- [ ] Verify no look-ahead bias in correlation calculation

**Deliverable:** Rolling correlation tracker

---

### Step 3.2: Enhanced Multi-Asset Portfolio
**Time:** Day 2-5

Upgrade the `Portfolio` class to properly handle multi-asset scenarios:

**Key changes to `src/portfolio/portfolio.py`:**

```python
# Enhanced Portfolio additions:

class Portfolio:
    def __init__(
        self,
        initial_capital: float,
        symbols: list,
        commission_pct: float = 0.001,
        position_sizer: Optional[PositionSizer] = None,
        risk_manager: Optional["RiskManager"] = None,
    ):
        # ... existing init ...
        self.correlation_tracker = CorrelationTracker()
        self.risk_manager = risk_manager

    def update_timeindex(self, timestamp: datetime) -> None:
        """Update portfolio valuation and correlations"""
        # Update current prices and correlation tracker
        for symbol in self.positions:
            if symbol in self.current_prices:
                self.correlation_tracker.update(
                    symbol, self.current_prices[symbol]
                )

        # Recalculate correlations periodically
        corr_matrix = self.correlation_tracker.calculate_matrix()
        if corr_matrix is not None and self.risk_manager:
            self.risk_manager.update_correlation_matrix(corr_matrix)

        # Update unrealized P&L for all positions
        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in self.current_prices:
                price = self.current_prices[symbol]
                position.market_value = position.quantity * price
                position.unrealized_pnl = (
                    (price - position.avg_cost) * position.quantity
                )

        equity = self.get_equity()
        self.equity_history.append({
            "timestamp": timestamp,
            "equity": equity,
            "cash": self.cash,
            "positions_value": equity - self.cash,
            "num_positions": sum(
                1 for p in self.positions.values() if p.quantity != 0
            ),
        })

    def get_portfolio_summary(self) -> Dict:
        """Return a summary of portfolio state"""
        equity = self.get_equity()
        positions_value = equity - self.cash

        long_exposure = sum(
            p.quantity * self.current_prices.get(p.symbol, 0)
            for p in self.positions.values()
            if p.quantity > 0
        )
        short_exposure = sum(
            abs(p.quantity) * self.current_prices.get(p.symbol, 0)
            for p in self.positions.values()
            if p.quantity < 0
        )

        return {
            "equity": equity,
            "cash": self.cash,
            "positions_value": positions_value,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "net_exposure": long_exposure - short_exposure,
            "gross_exposure": long_exposure + short_exposure,
            "num_long": sum(1 for p in self.positions.values() if p.quantity > 0),
            "num_short": sum(1 for p in self.positions.values() if p.quantity < 0),
            "total_realized_pnl": sum(
                p.realized_pnl for p in self.positions.values()
            ),
            "total_unrealized_pnl": sum(
                p.unrealized_pnl for p in self.positions.values()
            ),
        }
```

**Tasks:**
- [ ] Add `CorrelationTracker` to Portfolio
- [ ] Update `update_timeindex` to track correlations and unrealized P&L
- [ ] Add `get_portfolio_summary` method
- [ ] Handle short positions in `on_fill` (negative quantities, margin tracking)
- [ ] Track long/short exposure separately
- [ ] Add net/gross exposure calculations
- [ ] Update equity curve to include positions_value and num_positions
- [ ] Write tests for multi-asset P&L accuracy
- [ ] Test short selling flow end-to-end
- [ ] Test correlation tracking updates with portfolio

**Deliverable:** Multi-asset portfolio with correlation awareness

---

## Week 4: Extended Configuration & Data Source

### Step 4.1: Extended Configuration Schema
**Time:** Day 1-2

**Updated `src/config.py`:**
```python
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict
import yaml


class DataConfig(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    data_source: str = "csv"
    data_path: Optional[str] = None


class ExecutionConfig(BaseModel):
    initial_capital: float = 100000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005


class StrategyConfig(BaseModel):
    name: str
    parameters: Dict = {}


class SizingConfig(BaseModel):
    method: str = "fixed_fraction"  # fixed_fraction, volatility, kelly
    parameters: Dict = {}


class RiskConfig(BaseModel):
    max_position_pct: float = 0.10
    max_portfolio_exposure_pct: float = 1.0
    max_sector_pct: float = 0.30
    max_correlation_threshold: float = 0.70
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_open_positions: int = 20
    max_orders_per_day: int = 100

    @field_validator("max_position_pct", "max_daily_loss_pct", "max_drawdown_pct")
    @classmethod
    def validate_percentage(cls, v):
        if not 0 < v <= 1.0:
            raise ValueError(f"Must be between 0 and 1, got {v}")
        return v


class OptimizationConfig(BaseModel):
    method: str = "none"  # none, mean_variance, risk_parity
    rebalance_frequency: int = 20  # bars between rebalances
    parameters: Dict = {}


class BacktestConfig(BaseModel):
    data: DataConfig
    execution: ExecutionConfig
    strategy: StrategyConfig
    sizing: SizingConfig = SizingConfig()
    risk: RiskConfig = RiskConfig()
    optimization: OptimizationConfig = OptimizationConfig()


def load_config(path: str) -> BacktestConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return BacktestConfig(**data)
```

**Updated `configs/backtest_config.yaml`:**
```yaml
data:
  symbols:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - TSLA
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  data_source: "csv"
  data_path: "data/sample"

execution:
  initial_capital: 100000.0
  commission_pct: 0.001
  slippage_pct: 0.0005

strategy:
  name: "sma_crossover"
  parameters:
    short_window: 20
    long_window: 50

sizing:
  method: "volatility"
  parameters:
    risk_pct: 0.01
    atr_period: 14
    atr_multiplier: 2.0

risk:
  max_position_pct: 0.10
  max_portfolio_exposure_pct: 1.0
  max_daily_loss_pct: 0.03
  max_drawdown_pct: 0.15
  max_open_positions: 10
  max_orders_per_day: 50

optimization:
  method: "none"
  rebalance_frequency: 20
```

**Tasks:**
- [ ] Add `SizingConfig`, `RiskConfig`, `OptimizationConfig` to config
- [ ] Add Pydantic validators for risk parameters
- [ ] Maintain backwards compatibility (defaults for new sections)
- [ ] Update sample config file
- [ ] Write tests for new config validation
- [ ] Test loading configs with and without new sections

**Deliverable:** Extended configuration system

---

### Step 4.2: yfinance Data Handler
**Time:** Day 2-4

**File: `src/data/yfinance_handler.py`** (new file)
```python
from typing import List
from pathlib import Path
import pandas as pd

from .data_handler import HistoricalCSVDataHandler


class YFinanceDataHandler(HistoricalCSVDataHandler):
    """
    Data handler that downloads data from Yahoo Finance,
    caches it as CSV, then operates identically to CSV handler.

    This is NOT a live data handler — it downloads historical
    data once and replays it like the CSV handler.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        cache_dir: str = "data/cache",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Download any missing data
        self._download_missing(symbols, start_date, end_date)

        # Initialize parent CSV handler with cache directory
        super().__init__(
            data_path=str(self.cache_dir),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

    def _download_missing(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> None:
        """Download data for symbols not already cached"""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for YFinanceDataHandler. "
                "Install with: pip install yfinance"
            )

        for symbol in symbols:
            cache_file = self.cache_dir / f"{symbol}.csv"

            if cache_file.exists():
                # Check if cached data covers the requested range
                cached = pd.read_csv(cache_file, parse_dates=["date"])
                cached_start = cached["date"].min()
                cached_end = cached["date"].max()

                if (
                    cached_start <= pd.Timestamp(start_date)
                    and cached_end >= pd.Timestamp(end_date)
                ):
                    continue

            print(f"Downloading {symbol} from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            df = df.reset_index()
            df.columns = df.columns.str.lower()
            df = df.rename(columns={"date": "date"})
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df.to_csv(cache_file, index=False)

            print(f"  Cached {len(df)} bars to {cache_file}")
```

**Tasks:**
- [ ] Implement `YFinanceDataHandler`
- [ ] Add caching logic (download once, reuse)
- [ ] Handle missing/unavailable symbols gracefully
- [ ] Test with real symbol download
- [ ] Test cache hit (no re-download)
- [ ] Update `scripts/run_backtest.py` to support yfinance data source

**Deliverable:** yfinance data handler with caching

---

## Week 5: Portfolio Optimization

### Step 5.1: Mean-Variance Optimizer
**Time:** Day 1-3

**File: `src/optimization/base_optimizer.py`**
```python
from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class AllocationResult:
    """Result of portfolio optimization"""
    weights: Dict[str, float]  # symbol -> weight (sums to 1.0)
    method: str
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    notes: str = ""


class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers"""

    @abstractmethod
    def optimize(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
    ) -> AllocationResult:
        """
        Calculate optimal portfolio weights.

        Args:
            symbols: List of symbols to allocate across
            returns: DataFrame of historical returns (columns = symbols)
            current_weights: Current portfolio weights

        Returns:
            AllocationResult with target weights
        """
        raise NotImplementedError
```

**File: `src/optimization/mean_variance.py`**
```python
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from .base_optimizer import PortfolioOptimizer, AllocationResult


class MeanVarianceOptimizer(PortfolioOptimizer):
    """
    Classical Markowitz mean-variance optimization.

    Finds the portfolio weights that maximize the Sharpe ratio
    (or minimize variance for a target return).

    Uses Ledoit-Wolf shrinkage on the covariance matrix
    for numerical stability.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        target: str = "max_sharpe",  # max_sharpe, min_variance, target_return
        target_return: Optional[float] = None,
        max_weight: float = 0.30,
        min_weight: float = 0.0,
        shrinkage: bool = True,
    ):
        self.risk_free_rate = risk_free_rate
        self.target = target
        self.target_return = target_return
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.shrinkage = shrinkage

    def optimize(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
    ) -> AllocationResult:
        n = len(symbols)
        if n == 0:
            return AllocationResult(weights={}, method="mean_variance")

        # Filter returns to requested symbols
        returns = returns[symbols].dropna()

        if len(returns) < 30:
            # Insufficient data — equal weight fallback
            equal = 1.0 / n
            return AllocationResult(
                weights={s: equal for s in symbols},
                method="mean_variance",
                notes="insufficient data, equal weight fallback",
            )

        mu = returns.mean().values * 252  # Annualize
        if self.shrinkage:
            cov = self._ledoit_wolf_shrinkage(returns.values) * 252
        else:
            cov = returns.cov().values * 252

        # Optimization
        if self.target == "max_sharpe":
            weights = self._max_sharpe(mu, cov, n)
        elif self.target == "min_variance":
            weights = self._min_variance(cov, n)
        else:
            weights = self._max_sharpe(mu, cov, n)

        # Calculate portfolio metrics
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (
            (port_return - self.risk_free_rate) / port_vol
            if port_vol > 0
            else 0
        )

        return AllocationResult(
            weights={s: w for s, w in zip(symbols, weights)},
            method="mean_variance",
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
        )

    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """Find maximum Sharpe ratio portfolio"""

        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_weight, self.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(
            neg_sharpe, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        return result.x if result.success else np.ones(n) / n

    def _min_variance(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Find minimum variance portfolio"""

        def portfolio_vol(w):
            return np.sqrt(w @ cov @ w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_weight, self.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(
            portfolio_vol, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        return result.x if result.success else np.ones(n) / n

    def _ledoit_wolf_shrinkage(self, returns: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf shrinkage estimator for covariance matrix.
        Shrinks toward scaled identity matrix for stability.
        """
        t, n = returns.shape
        sample_cov = np.cov(returns, rowvar=False)
        mu_target = np.trace(sample_cov) / n
        target = mu_target * np.eye(n)

        # Estimate optimal shrinkage intensity
        delta = sample_cov - target
        sum_sq = np.sum(delta ** 2) / n

        # Simplified shrinkage (Ledoit-Wolf 2004)
        x = returns - returns.mean(axis=0)
        sum_sq_x = 0
        for i in range(t):
            xi = np.outer(x[i], x[i])
            sum_sq_x += np.sum((xi - sample_cov) ** 2) / n
        sum_sq_x /= t ** 2

        shrinkage_intensity = max(0, min(1, sum_sq_x / sum_sq)) if sum_sq > 0 else 0

        return shrinkage_intensity * target + (1 - shrinkage_intensity) * sample_cov
```

**Tasks:**
- [ ] Create `src/optimization/__init__.py`
- [ ] Implement `PortfolioOptimizer` abstract base class
- [ ] Implement `MeanVarianceOptimizer` with max-Sharpe and min-variance targets
- [ ] Implement Ledoit-Wolf covariance shrinkage
- [ ] Test against known optimal portfolio (2-asset case with analytical solution)
- [ ] Test weight constraints (min/max weight, sum to 1)
- [ ] Test with insufficient data (equal weight fallback)
- [ ] Test shrinkage improves stability vs. sample covariance

**Deliverable:** Mean-variance portfolio optimizer

---

### Step 5.2: Risk Parity Optimizer
**Time:** Day 3-5

**File: `src/optimization/risk_parity.py`**
```python
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from .base_optimizer import PortfolioOptimizer, AllocationResult


class RiskParityOptimizer(PortfolioOptimizer):
    """
    Risk Parity (Equal Risk Contribution) optimization.

    Each asset contributes equally to total portfolio risk.
    Does not require return estimates (only covariance).

    More robust than mean-variance in practice because:
    - No return forecasting needed
    - Naturally diversified
    - Lower sensitivity to estimation error
    """

    def __init__(
        self,
        max_weight: float = 0.40,
        min_weight: float = 0.01,
    ):
        self.max_weight = max_weight
        self.min_weight = min_weight

    def optimize(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
    ) -> AllocationResult:
        n = len(symbols)
        if n == 0:
            return AllocationResult(weights={}, method="risk_parity")

        returns = returns[symbols].dropna()

        if len(returns) < 30:
            equal = 1.0 / n
            return AllocationResult(
                weights={s: equal for s in symbols},
                method="risk_parity",
                notes="insufficient data, equal weight fallback",
            )

        cov = returns.cov().values * 252  # Annualize

        weights = self._risk_parity_weights(cov, n)

        # Calculate metrics
        port_vol = np.sqrt(weights @ cov @ weights)
        mu = returns.mean().values * 252
        port_return = weights @ mu

        return AllocationResult(
            weights={s: w for s, w in zip(symbols, weights)},
            method="risk_parity",
            expected_return=port_return,
            expected_volatility=port_vol,
        )

    def _risk_parity_weights(self, cov: np.ndarray, n: int) -> np.ndarray:
        """
        Find weights where each asset contributes equally to portfolio risk.

        Risk contribution of asset i = w_i * (cov @ w)_i / sigma_portfolio
        Target: all risk contributions equal to 1/n of total risk.
        """
        target_risk = 1.0 / n

        def risk_budget_objective(w):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol == 0:
                return 0
            marginal_risk = cov @ w
            risk_contrib = w * marginal_risk / port_vol
            # Minimize sum of squared deviations from target
            return np.sum((risk_contrib - target_risk * port_vol) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_weight, self.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(
            risk_budget_objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        return result.x if result.success else np.ones(n) / n
```

**Tasks:**
- [ ] Implement `RiskParityOptimizer`
- [ ] Verify equal risk contribution using known covariance matrices
- [ ] Test that high-vol assets get lower weight
- [ ] Test that all weights sum to 1.0
- [ ] Test with 2-asset (analytical solution) and 5-asset cases
- [ ] Compare with mean-variance output on same data

**Deliverable:** Risk parity optimizer

---

## Week 6: Walk-Forward Framework & Multi-Asset Strategy

### Step 6.1: Walk-Forward Testing Framework
**Time:** Day 1-3

**File: `src/backtest/walk_forward.py`** (new file)
```python
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

from .engine import BacktestEngine


@dataclass
class WalkForwardWindow:
    """A single train/test window"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_metrics: Optional[Dict] = None
    test_metrics: Optional[Dict] = None


@dataclass
class WalkForwardResult:
    """Results of walk-forward analysis"""
    windows: List[WalkForwardWindow]
    aggregate_metrics: Dict
    walk_forward_efficiency: float
    parameter_stability: Dict


class WalkForwardRunner:
    """
    Walk-forward validation framework.

    Splits data into rolling train/test windows.
    Runs backtest on each window independently.
    Aggregates out-of-sample results.

    This is critical for detecting overfitting:
    - WFE > 50% suggests strategy has genuine edge
    - WFE < 30% suggests overfitting
    """

    def __init__(
        self,
        train_days: int = 504,   # ~2 years
        test_days: int = 126,    # ~6 months
        step_days: int = 63,     # ~3 months between windows
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

    def generate_windows(
        self, start_date: str, end_date: str
    ) -> List[WalkForwardWindow]:
        """Generate train/test window pairs"""
        dates = pd.bdate_range(start=start_date, end=end_date)
        windows = []
        window_id = 0

        i = 0
        while i + self.train_days + self.test_days <= len(dates):
            train_start = dates[i].strftime("%Y-%m-%d")
            train_end = dates[i + self.train_days - 1].strftime("%Y-%m-%d")
            test_start = dates[i + self.train_days].strftime("%Y-%m-%d")
            test_end = dates[
                min(i + self.train_days + self.test_days - 1, len(dates) - 1)
            ].strftime("%Y-%m-%d")

            windows.append(
                WalkForwardWindow(
                    window_id=window_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            window_id += 1
            i += self.step_days

        return windows

    def run(
        self,
        windows: List[WalkForwardWindow],
        engine_factory: Callable[[str, str], BacktestEngine],
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        Args:
            windows: List of train/test windows
            engine_factory: Function that creates a BacktestEngine
                            given (start_date, end_date)
        """
        for window in windows:
            print(f"\n--- Window {window.window_id} ---")

            # Run train period
            print(f"  Train: {window.train_start} to {window.train_end}")
            train_engine = engine_factory(
                window.train_start, window.train_end
            )
            train_results = train_engine.run()
            window.train_metrics = train_results.get("metrics", {})

            # Run test period
            print(f"  Test:  {window.test_start} to {window.test_end}")
            test_engine = engine_factory(
                window.test_start, window.test_end
            )
            test_results = test_engine.run()
            window.test_metrics = test_results.get("metrics", {})

        return self._aggregate_results(windows)

    def _aggregate_results(
        self, windows: List[WalkForwardWindow]
    ) -> WalkForwardResult:
        """Aggregate walk-forward results across all windows"""
        is_returns = []
        oos_returns = []
        is_sharpes = []
        oos_sharpes = []

        for w in windows:
            if w.train_metrics and w.test_metrics:
                is_ret = w.train_metrics.get("annualized_return_pct", 0)
                oos_ret = w.test_metrics.get("annualized_return_pct", 0)
                is_returns.append(is_ret)
                oos_returns.append(oos_ret)
                is_sharpes.append(
                    w.train_metrics.get("sharpe_ratio", 0)
                )
                oos_sharpes.append(
                    w.test_metrics.get("sharpe_ratio", 0)
                )

        # Walk-Forward Efficiency
        avg_is_return = np.mean(is_returns) if is_returns else 0
        avg_oos_return = np.mean(oos_returns) if oos_returns else 0
        wfe = (
            (avg_oos_return / avg_is_return * 100)
            if avg_is_return != 0
            else 0
        )

        # Parameter stability (CV of Sharpe across windows)
        sharpe_cv = (
            np.std(oos_sharpes) / np.mean(oos_sharpes)
            if oos_sharpes and np.mean(oos_sharpes) != 0
            else float("inf")
        )

        aggregate = {
            "num_windows": len(windows),
            "avg_is_return": avg_is_return,
            "avg_oos_return": avg_oos_return,
            "avg_is_sharpe": np.mean(is_sharpes) if is_sharpes else 0,
            "avg_oos_sharpe": np.mean(oos_sharpes) if oos_sharpes else 0,
            "oos_sharpe_std": np.std(oos_sharpes) if oos_sharpes else 0,
            "pct_profitable_windows": (
                sum(1 for r in oos_returns if r > 0) / len(oos_returns) * 100
                if oos_returns
                else 0
            ),
        }

        return WalkForwardResult(
            windows=windows,
            aggregate_metrics=aggregate,
            walk_forward_efficiency=wfe,
            parameter_stability={"sharpe_cv": sharpe_cv},
        )
```

**Tasks:**
- [ ] Implement `WalkForwardRunner` with configurable window sizes
- [ ] Implement window generation logic
- [ ] Implement results aggregation (WFE, Sharpe CV)
- [ ] Test window generation produces correct date ranges
- [ ] Test with simple backtest factory function
- [ ] Verify WFE calculation against manual computation
- [ ] Add formatted reporting for walk-forward results

**Deliverable:** Walk-forward testing framework

---

### Step 6.2: Multi-Asset SMA Strategy
**Time:** Day 3-5

**File: `src/strategy/multi_asset_sma.py`** (new file)
```python
from typing import Dict, Any, List, Optional
import pandas as pd

from .base_strategy import Strategy
from ..events.event import SignalEvent, SignalType
from ..data.data_handler import DataHandler
from ..optimization.base_optimizer import PortfolioOptimizer, AllocationResult


class MultiAssetSMAStrategy(Strategy):
    """
    Multi-asset SMA crossover with portfolio optimization.

    Generates signals for multiple assets simultaneously.
    Optionally uses a PortfolioOptimizer to adjust allocations
    at a configurable rebalance frequency.

    Parameters:
        short_window: Short SMA period (default: 20)
        long_window: Long SMA period (default: 50)
        rebalance_frequency: Bars between rebalances (default: 20)
    """

    def __init__(
        self,
        symbols: list,
        parameters: Dict[str, Any] = None,
        optimizer: Optional[PortfolioOptimizer] = None,
    ):
        super().__init__(symbols, parameters)

        self.short_window = self.parameters.get("short_window", 20)
        self.long_window = self.parameters.get("long_window", 50)
        self.rebalance_frequency = self.parameters.get(
            "rebalance_frequency", 20
        )

        self.optimizer = optimizer
        self.target_weights: Dict[str, float] = {
            s: 1.0 / len(symbols) for s in symbols
        }

        self.prev_short_sma: Dict[str, float] = {}
        self.prev_long_sma: Dict[str, float] = {}
        self.bars_since_rebalance: int = 0

    def calculate_signals(
        self, timestamp, data_handler: DataHandler
    ) -> List[SignalEvent]:
        """Generate signals across all assets"""
        signals = []

        # Rebalance weights periodically
        self.bars_since_rebalance += 1
        if (
            self.optimizer
            and self.bars_since_rebalance >= self.rebalance_frequency
        ):
            self._rebalance_weights(data_handler)
            self.bars_since_rebalance = 0

        # Generate signals per symbol
        for symbol in self.symbols:
            signal = self._check_crossover(timestamp, symbol, data_handler)
            if signal:
                signals.append(signal)

        return signals

    def _check_crossover(
        self, timestamp, symbol: str, data_handler: DataHandler
    ) -> Optional[SignalEvent]:
        """Check for SMA crossover on a single symbol"""
        bars = data_handler.get_latest_bars(symbol, self.long_window + 1)

        if len(bars) < self.long_window:
            return None

        close_prices = bars["close"]
        short_sma = close_prices.tail(self.short_window).mean()
        long_sma = close_prices.tail(self.long_window).mean()

        prev_short = self.prev_short_sma.get(symbol)
        prev_long = self.prev_long_sma.get(symbol)

        self.prev_short_sma[symbol] = short_sma
        self.prev_long_sma[symbol] = long_sma

        if prev_short is None or prev_long is None:
            return None

        # Signal strength based on target weight from optimizer
        strength = self.target_weights.get(symbol, 1.0 / len(self.symbols))

        if prev_short <= prev_long and short_sma > long_sma:
            if self.current_positions.get(symbol, 0) <= 0:
                return self._create_signal(
                    timestamp, symbol, SignalType.LONG, strength=strength
                )

        elif prev_short >= prev_long and short_sma < long_sma:
            if self.current_positions.get(symbol, 0) > 0:
                return self._create_signal(
                    timestamp, symbol, SignalType.EXIT, strength=strength
                )

        return None

    def _rebalance_weights(self, data_handler: DataHandler) -> None:
        """Recalculate target weights using optimizer"""
        # Collect returns for all symbols
        returns_data = {}
        for symbol in self.symbols:
            bars = data_handler.get_latest_bars(symbol, 252)
            if len(bars) >= 60:
                returns_data[symbol] = bars["close"].pct_change().dropna()

        if len(returns_data) < 2:
            return

        returns_df = pd.DataFrame(returns_data).dropna()

        result = self.optimizer.optimize(
            symbols=list(returns_data.keys()),
            returns=returns_df,
            current_weights=self.target_weights,
        )

        self.target_weights = result.weights
```

**Tasks:**
- [ ] Implement `MultiAssetSMAStrategy`
- [ ] Integrate optional `PortfolioOptimizer` for weight rebalancing
- [ ] Signal strength tied to optimizer weight
- [ ] Test multi-symbol signal generation
- [ ] Test rebalance frequency
- [ ] Test with and without optimizer
- [ ] Run full backtest with 5 assets

**Deliverable:** Multi-asset strategy with optimization

---

## Week 7: Integration & Updated Run Script

### Step 7.1: Updated Run Script
**Time:** Day 1-3

**Updated `scripts/run_backtest.py`:**

Wire the new components together based on configuration:

```python
#!/usr/bin/env python
"""Run backtest with Phase 2 components"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.data_handler import HistoricalCSVDataHandler
from src.data.yfinance_handler import YFinanceDataHandler
from src.strategy.sma_crossover import SMACrossoverStrategy
from src.strategy.multi_asset_sma import MultiAssetSMAStrategy
from src.portfolio.portfolio import Portfolio
from src.execution.execution_handler import ExecutionHandler
from src.performance.metrics import PerformanceTracker
from src.performance.visualization import create_full_report
from src.backtest.engine import BacktestEngine
from src.risk.position_sizer import (
    FixedFractionSizer, VolatilityBasedSizer, KellyCriterionSizer
)
from src.risk.risk_manager import RiskManager, RiskLimits
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer


def build_position_sizer(config):
    """Create position sizer from config"""
    method = config.sizing.method
    params = config.sizing.parameters

    if method == "fixed_fraction":
        return FixedFractionSizer(fraction=params.get("fraction", 0.10))
    elif method == "volatility":
        return VolatilityBasedSizer(
            risk_pct=params.get("risk_pct", 0.01),
            atr_period=params.get("atr_period", 14),
            atr_multiplier=params.get("atr_multiplier", 2.0),
        )
    elif method == "kelly":
        return KellyCriterionSizer(
            kelly_fraction=params.get("kelly_fraction", 0.25),
            max_position_pct=params.get("max_position_pct", 0.20),
        )
    else:
        return FixedFractionSizer()


def build_risk_manager(config):
    """Create risk manager from config"""
    limits = RiskLimits(
        max_position_pct=config.risk.max_position_pct,
        max_portfolio_exposure_pct=config.risk.max_portfolio_exposure_pct,
        max_daily_loss_pct=config.risk.max_daily_loss_pct,
        max_drawdown_pct=config.risk.max_drawdown_pct,
        max_open_positions=config.risk.max_open_positions,
        max_orders_per_day=config.risk.max_orders_per_day,
    )
    return RiskManager(limits=limits)


def build_optimizer(config):
    """Create portfolio optimizer from config"""
    method = config.optimization.method
    params = config.optimization.parameters

    if method == "mean_variance":
        return MeanVarianceOptimizer(
            target=params.get("target", "max_sharpe"),
            max_weight=params.get("max_weight", 0.30),
        )
    elif method == "risk_parity":
        return RiskParityOptimizer(
            max_weight=params.get("max_weight", 0.40),
        )
    return None


def build_data_handler(config):
    """Create data handler from config"""
    if config.data.data_source == "yfinance":
        return YFinanceDataHandler(
            symbols=config.data.symbols,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            cache_dir=config.data.data_path or "data/cache",
        )
    else:
        return HistoricalCSVDataHandler(
            data_path=config.data.data_path,
            symbols=config.data.symbols,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
        )


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    config = load_config(args.config)
    symbols = config.data.symbols

    # Build components
    data_handler = build_data_handler(config)
    position_sizer = build_position_sizer(config)
    risk_manager = build_risk_manager(config)
    optimizer = build_optimizer(config)

    # Choose strategy
    if len(symbols) > 1 and optimizer:
        strategy = MultiAssetSMAStrategy(
            symbols=symbols,
            parameters=config.strategy.parameters,
            optimizer=optimizer,
        )
    else:
        strategy = SMACrossoverStrategy(
            symbols=symbols,
            parameters=config.strategy.parameters,
        )

    portfolio = Portfolio(
        initial_capital=config.execution.initial_capital,
        symbols=symbols,
        commission_pct=config.execution.commission_pct,
        position_sizer=position_sizer,
    )

    execution_handler = ExecutionHandler(
        commission_pct=config.execution.commission_pct,
        slippage_pct=config.execution.slippage_pct,
    )

    performance_tracker = PerformanceTracker(
        initial_capital=config.execution.initial_capital
    )

    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        risk_manager=risk_manager,
        performance_tracker=performance_tracker,
    )

    results = engine.run()

    # Report
    performance_tracker.print_report(trades=results["trade_history"])
    equity_curve = performance_tracker.get_equity_curve()
    create_full_report(equity_curve, output_dir=args.output)

    # Risk summary
    print(f"\nRisk Summary:")
    print(f"  Orders Rejected: {engine.rejected_orders}")
    print(f"  Trading Halted:  {risk_manager.is_halted}")
    if risk_manager.is_halted:
        print(f"  Halt Reason:     {risk_manager.halt_reason}")

    summary = portfolio.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"  Long Positions:  {summary['num_long']}")
    print(f"  Short Positions: {summary['num_short']}")
    print(f"  Gross Exposure:  ${summary['gross_exposure']:,.2f}")
    print(f"  Net Exposure:    ${summary['net_exposure']:,.2f}")

    return results


if __name__ == "__main__":
    main()
```

**Tasks:**
- [ ] Implement component factory functions
- [ ] Wire all Phase 2 components together
- [ ] Add risk summary to output
- [ ] Add portfolio summary to output
- [ ] Test with each sizing method
- [ ] Test with each optimizer
- [ ] Test with and without risk manager
- [ ] Run full multi-asset backtest

**Deliverable:** Updated run script supporting all Phase 2 features

---

### Step 7.2: Enhanced Performance Metrics
**Time:** Day 3-5

Add Phase 2 metrics to the performance report:

**Additions to `src/performance/metrics.py`:**

```python
# Add these methods to PerformanceTracker:

def calculate_extended_metrics(
    self, trades: list, portfolio_summary: dict
) -> Dict:
    """Phase 2 extended metrics"""
    base = self.calculate_metrics()

    base["risk_metrics"] = {
        "avg_position_count": self._avg_position_count(),
        "max_position_count": self._max_position_count(),
        "gross_exposure_avg": self._avg_gross_exposure(),
        "turnover": self._calculate_turnover(trades),
    }

    if trades:
        base["trade_metrics"] = self.calculate_trade_metrics(trades)

    return base

def _calculate_turnover(self, trades: list) -> float:
    """Annual portfolio turnover rate"""
    if not trades or not self.equity_curve:
        return 0.0
    total_traded = sum(t.quantity * t.price for t in trades)
    avg_equity = np.mean([e["equity"] for e in self.equity_curve])
    days = len(self.equity_curve)
    if avg_equity == 0 or days == 0:
        return 0.0
    return (total_traded / avg_equity) * (252 / days)
```

**Tasks:**
- [ ] Add turnover calculation
- [ ] Add position count tracking
- [ ] Add exposure tracking to equity curve
- [ ] Update `print_report` for Phase 2 metrics
- [ ] Test extended metrics calculations

**Deliverable:** Extended performance reporting

---

## Week 8: Testing, Validation & Documentation

### Step 8.1: Comprehensive Test Suite
**Time:** Day 1-3

**New test files to create:**

| Test File | Coverage |
|-----------|----------|
| `tests/test_position_sizer.py` | All 3 sizers with known inputs/outputs |
| `tests/test_risk_manager.py` | All 7 risk checks, halt/reset, daily reset |
| `tests/test_optimizer.py` | Mean-variance and risk parity with analytical solutions |
| `tests/test_correlation.py` | Rolling correlation accuracy |
| `tests/test_multi_asset.py` | Multi-asset backtest end-to-end |
| `tests/test_walk_forward.py` | Window generation, WFE calculation |

**Key test scenarios:**
- [ ] Kelly sizer with known win rate → verify formula output
- [ ] Vol sizer with known ATR → verify position size
- [ ] Risk manager rejects over-concentrated position
- [ ] Risk manager halts on drawdown breach
- [ ] Risk manager reduces position to stay within limits
- [ ] Mean-variance with 2 assets → compare to analytical
- [ ] Risk parity with equal-vol assets → equal weights
- [ ] Walk-forward WFE with known IS/OOS returns
- [ ] Multi-asset backtest processes all symbols
- [ ] Short selling P&L is calculated correctly
- [ ] Backwards compatibility: Phase 1 config still works

**Deliverable:** 200+ new tests, total suite 600+

---

### Step 8.2: Validation & Documentation
**Time:** Day 3-5

**Validation runs:**
- [ ] Single asset backtest (AAPL) — compare Phase 1 vs Phase 2 results
- [ ] Multi-asset backtest (5 symbols) with each optimizer
- [ ] Run with tight risk limits — verify orders get rejected
- [ ] Run with vol-based sizing — verify position sizes vary
- [ ] Walk-forward analysis on SMA crossover — calculate WFE
- [ ] Compare mean-variance vs risk parity allocations

**Documentation:**
- [ ] Update README with Phase 2 features
- [ ] Document all configuration options
- [ ] Add example configs for each sizing/optimization method
- [ ] Update NOTES.md with Phase 2 architecture

**Deliverable:** Validated system with documentation

---

## Success Criteria Checklist

### Functional Requirements
- [ ] Position sizing: Kelly, vol-based, and fixed-fraction all produce correct sizes
- [ ] Risk manager: All 7 checks fire correctly
- [ ] Risk manager: Drawdown halt stops all trading
- [ ] Multi-asset: Backtest 5+ assets simultaneously
- [ ] Correlation: Rolling correlation tracks correctly (no look-ahead)
- [ ] Mean-variance: Produces feasible weight allocations
- [ ] Risk parity: Equal risk contribution verified
- [ ] Walk-forward: WFE calculated across multiple windows
- [ ] Short selling: Negative positions handled correctly in P&L

### Code Quality
- [ ] All tests pass (600+ total)
- [ ] Backwards compatible with Phase 1 configurations
- [ ] Type hints on all new public methods
- [ ] Docstrings on all new classes and methods
- [ ] Code passes linting (ruff, black, mypy)

### Performance
- [ ] 5-asset, 4-year backtest completes in < 30 seconds
- [ ] Portfolio optimization runs in < 1 second per rebalance

---

## Risk Mitigation

| Risk | Mitigation | Status |
|------|------------|--------|
| Optimization instability | Ledoit-Wolf shrinkage, equal-weight fallback | ☐ |
| Kelly over-leveraging | Fractional Kelly (0.25x), hard position cap | ☐ |
| Correlation look-ahead | Rolling window only, min_periods enforced | ☐ |
| Risk manager too aggressive | Configurable limits, warnings before rejections | ☐ |
| Breaking Phase 1 | All existing tests must pass, default params match Phase 1 | ☐ |
| Walk-forward bias | Strict train/test separation, no parameter leakage | ☐ |

---

## Next Steps (Phase 3 Preview)

After completing Phase 2:
1. Feature engineering pipeline (technical indicators, statistical features)
2. ML model integration (XGBoost/LightGBM for signal generation)
3. CPCV implementation for proper ML validation
4. Deflated Sharpe Ratio calculation
5. MLflow experiment tracking
6. Regime detection (HMM)

---

*Phase 2 Development Plan v1.0*

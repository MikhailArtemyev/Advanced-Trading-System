"""Position sizing algorithms for risk management.

Provides multiple position sizing strategies:
- FixedFractionSizer: Fixed percentage of equity (Phase 1 extraction)
- VolatilityBasedSizer: ATR-based dynamic sizing
- KellyCriterionSizer: Fractional Kelly with trade history

All sizers implement the PositionSizer abstract base class and return
SizingResult dataclass instances.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..data.data_handler import DataHandler


@dataclass(frozen=True)
class SizingResult:
    """Result of a position sizing calculation.

    Attributes:
        quantity: Number of shares to trade (always >= 0)
        target_value: Dollar value the sizer targeted
        method: Name of the sizing method used
        metadata: Optional extra info (e.g., ATR value, Kelly fraction)
    """

    quantity: int
    target_value: float
    method: str
    metadata: dict[str, float] = field(default_factory=dict)


class PositionSizer(ABC):
    """Abstract base class for position sizing algorithms.

    All position sizers must implement calculate_size() which determines
    how many shares to trade given the current portfolio state and market data.
    """

    @abstractmethod
    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        equity: float,
        cash: float,
        data_handler: DataHandler | None = None,
        signal_strength: float = 1.0,
        trades: list[Any] | None = None,
    ) -> SizingResult:
        """Calculate the number of shares to trade.

        Args:
            symbol: The instrument symbol
            current_price: Current market price of the instrument
            equity: Total portfolio equity (cash + positions)
            cash: Available cash in the portfolio
            data_handler: Optional data handler for accessing market history
                         (required by VolatilityBasedSizer)
            signal_strength: Signal strength from 0.0 to 1.0
            trades: Optional list of past Trade objects
                   (required by KellyCriterionSizer)

        Returns:
            SizingResult with the calculated quantity and metadata
        """
        raise NotImplementedError


class FixedFractionSizer(PositionSizer):
    """Fixed fraction position sizing.

    Allocates a fixed percentage of equity to each position.
    This extracts the Phase 1 hardcoded logic into a standalone class.

    Attributes:
        fraction: Target allocation as fraction of equity (e.g., 0.10 for 10%)
    """

    def __init__(self, fraction: float = 0.10) -> None:
        """Initialize the fixed fraction sizer.

        Args:
            fraction: Target position size as fraction of equity (default 10%)

        Raises:
            ValueError: If fraction is not between 0 and 1
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
        self.fraction = fraction

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        equity: float,
        cash: float,
        data_handler: DataHandler | None = None,
        signal_strength: float = 1.0,
        trades: list[Any] | None = None,
    ) -> SizingResult:
        """Calculate position size as fixed fraction of equity.

        Logic (identical to Phase 1):
            target_value = equity * fraction
            quantity = int(target_value / current_price)
        """
        target_value = equity * self.fraction

        if current_price <= 0:
            return SizingResult(
                quantity=0, target_value=target_value, method="fixed_fraction"
            )

        quantity = int(target_value / current_price)

        return SizingResult(
            quantity=max(quantity, 0),
            target_value=target_value,
            method="fixed_fraction",
            metadata={"fraction": self.fraction},
        )


class VolatilityBasedSizer(PositionSizer):
    """Volatility-based position sizing using Average True Range (ATR).

    Sizes positions inversely proportional to volatility:
    - High volatility -> smaller positions
    - Low volatility -> larger positions

    Formula:
        risk_per_share = ATR * atr_multiplier
        max_risk = equity * risk_fraction
        quantity = int(max_risk / risk_per_share)

    Attributes:
        risk_fraction: Maximum fraction of equity to risk per trade
        atr_period: Lookback period for ATR calculation
        atr_multiplier: Multiplier on ATR for risk-per-share (default 2.0)
        max_position_fraction: Hard cap on position as fraction of equity
    """

    def __init__(
        self,
        risk_fraction: float = 0.02,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        max_position_fraction: float = 0.20,
    ) -> None:
        """Initialize the volatility-based sizer.

        Args:
            risk_fraction: Max risk per trade as fraction of equity (default 2%)
            atr_period: Lookback for ATR calculation (default 14)
            atr_multiplier: ATR multiplier for stop distance (default 2.0)
            max_position_fraction: Hard cap on position as fraction of equity

        Raises:
            ValueError: If parameters are out of valid range
        """
        if not 0.0 < risk_fraction <= 1.0:
            raise ValueError(
                f"risk_fraction must be between 0 and 1, got {risk_fraction}"
            )
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be > 0, got {atr_multiplier}")

        self.risk_fraction = risk_fraction
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.max_position_fraction = max_position_fraction

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        equity: float,
        cash: float,
        data_handler: DataHandler | None = None,
        signal_strength: float = 1.0,
        trades: list[Any] | None = None,
    ) -> SizingResult:
        """Calculate position size based on ATR volatility.

        Falls back to risk_fraction as a fixed fraction if data_handler
        is not provided or has insufficient data.
        """
        if current_price <= 0:
            return SizingResult(
                quantity=0, target_value=0.0, method="volatility_fallback"
            )

        if data_handler is None:
            target_value = equity * self.risk_fraction
            quantity = int(target_value / current_price)
            return SizingResult(
                quantity=max(quantity, 0),
                target_value=target_value,
                method="volatility_fallback",
            )

        atr = self._calculate_atr(symbol, data_handler)

        if atr <= 0:
            target_value = equity * self.risk_fraction
            quantity = int(target_value / current_price)
            return SizingResult(
                quantity=max(quantity, 0),
                target_value=target_value,
                method="volatility_fallback",
                metadata={"atr": 0.0},
            )

        risk_per_share = atr * self.atr_multiplier
        max_risk = equity * self.risk_fraction
        quantity = int(max_risk / risk_per_share)

        # Apply hard cap
        max_quantity = int((equity * self.max_position_fraction) / current_price)
        quantity = min(quantity, max_quantity)

        target_value = quantity * current_price

        return SizingResult(
            quantity=max(quantity, 0),
            target_value=target_value,
            method="volatility_atr",
            metadata={
                "atr": atr,
                "risk_per_share": risk_per_share,
                "max_risk": max_risk,
            },
        )

    def _calculate_atr(self, symbol: str, data_handler: DataHandler) -> float:
        """Calculate Average True Range.

        ATR = mean of True Range over atr_period bars.
        True Range = max(high - low, |high - prev_close|, |low - prev_close|)

        Args:
            symbol: The instrument symbol
            data_handler: Data handler for accessing OHLCV data

        Returns:
            ATR value, or 0.0 if insufficient data
        """
        bars = data_handler.get_latest_bars(symbol, self.atr_period + 1)

        if len(bars) < 2:
            return 0.0

        high = bars["high"]
        low = bars["low"]
        close = bars["close"]

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        true_range = true_range.dropna()

        if len(true_range) == 0:
            return 0.0

        atr_value: float = float(true_range.tail(self.atr_period).mean())
        return atr_value


class KellyCriterionSizer(PositionSizer):
    """Kelly Criterion position sizing using trade history.

    Uses the Kelly formula to determine optimal bet size:
        Kelly % = W - (1 - W) / R
    Where:
        W = win rate (probability of winning)
        R = win/loss ratio (average win / average loss)

    Uses fractional Kelly (half-Kelly by default) for safety,
    since full Kelly can experience 50-70% drawdowns.

    Attributes:
        kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly)
        min_trades: Minimum number of trades before Kelly is applied
        max_position_fraction: Hard cap on position as fraction of equity
        default_fraction: Fallback fraction when insufficient trade history
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        min_trades: int = 20,
        max_position_fraction: float = 0.25,
        default_fraction: float = 0.10,
    ) -> None:
        """Initialize the Kelly Criterion sizer.

        Args:
            kelly_fraction: Fraction of full Kelly to use (default 0.5)
            min_trades: Minimum trades for Kelly estimation (default 20)
            max_position_fraction: Hard cap on position size (default 25%)
            default_fraction: Fallback when insufficient history (default 10%)

        Raises:
            ValueError: If parameters are out of valid range
        """
        if not 0.0 < kelly_fraction <= 1.0:
            raise ValueError(
                f"kelly_fraction must be between 0 and 1, got {kelly_fraction}"
            )
        if min_trades < 1:
            raise ValueError(f"min_trades must be >= 1, got {min_trades}")

        self.kelly_fraction = kelly_fraction
        self.min_trades = min_trades
        self.max_position_fraction = max_position_fraction
        self.default_fraction = default_fraction

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        equity: float,
        cash: float,
        data_handler: DataHandler | None = None,
        signal_strength: float = 1.0,
        trades: list[Any] | None = None,
    ) -> SizingResult:
        """Calculate position size using fractional Kelly Criterion.

        Uses trade history to estimate win rate and win/loss ratio,
        then applies the Kelly formula scaled by kelly_fraction and
        signal_strength.
        """
        if current_price <= 0:
            return SizingResult(quantity=0, target_value=0.0, method="kelly_criterion")

        raw_kelly = self._calculate_kelly_pct(trades)

        # Apply fractional Kelly and signal strength
        adjusted_fraction = raw_kelly * self.kelly_fraction * signal_strength

        # Clamp to valid range
        adjusted_fraction = min(adjusted_fraction, self.max_position_fraction)
        adjusted_fraction = max(adjusted_fraction, 0.0)

        target_value = equity * adjusted_fraction
        quantity = int(target_value / current_price)

        return SizingResult(
            quantity=max(quantity, 0),
            target_value=target_value,
            method="kelly_criterion",
            metadata={
                "raw_kelly_pct": raw_kelly,
                "adjusted_fraction": adjusted_fraction,
                "kelly_fraction_multiplier": self.kelly_fraction,
                "signal_strength": signal_strength,
            },
        )

    def _calculate_kelly_pct(self, trades: list[Any] | None) -> float:
        """Calculate the full Kelly percentage from trade history.

        Kelly % = W - (1 - W) / R
        Where W = win rate, R = avg_win / avg_loss

        Args:
            trades: List of Trade objects with pnl and side attributes

        Returns:
            Kelly percentage (can be negative if strategy is losing)
        """
        if trades is None or len(trades) == 0:
            return self.default_fraction

        # Filter to completed round-trip trades (sells with P&L)
        sell_trades = [t for t in trades if t.side == "SELL"]

        if len(sell_trades) < self.min_trades:
            return self.default_fraction

        winners = [t for t in sell_trades if t.pnl > 0]
        losers = [t for t in sell_trades if t.pnl < 0]

        if not losers:
            return self.max_position_fraction

        if not winners:
            return 0.0

        win_rate = len(winners) / len(sell_trades)
        avg_win = sum(t.pnl for t in winners) / len(winners)
        avg_loss = abs(sum(t.pnl for t in losers) / len(losers))

        if avg_loss == 0:
            return self.max_position_fraction

        win_loss_ratio = avg_win / avg_loss

        # Kelly formula: W - (1 - W) / R
        kelly: float = win_rate - (1 - win_rate) / win_loss_ratio
        return kelly

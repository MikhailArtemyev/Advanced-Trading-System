"""Unit tests for the RiskManager, RiskLimits, and RiskCheckResult classes."""

from datetime import datetime
from typing import Any

import pytest

from src.events.event import OrderEvent, OrderSide, OrderType
from src.risk.risk_manager import RiskCheckResult, RiskLimits, RiskManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_order():
    """Factory for creating OrderEvent instances."""

    def _make(
        symbol: str = "AAPL",
        side: OrderSide = OrderSide.BUY,
        quantity: int = 100,
        timestamp: datetime | None = None,
    ) -> OrderEvent:
        return OrderEvent(
            timestamp=timestamp or datetime(2023, 6, 15, 10, 0),
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
        )

    return _make


@pytest.fixture
def empty_positions() -> dict[str, Any]:
    """No positions."""
    return {}


@pytest.fixture
def aapl_position() -> dict[str, Any]:
    """Single AAPL position: 50 shares at $150."""
    return {
        "AAPL": {
            "quantity": 50,
            "avg_cost": 150.0,
            "market_value": 7500.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
        }
    }


# ---------------------------------------------------------------------------
# TestRiskLimits
# ---------------------------------------------------------------------------


class TestRiskLimits:
    """Tests for the RiskLimits dataclass."""

    def test_default_values(self):
        limits = RiskLimits()
        assert limits.max_position_pct == 0.10
        assert limits.max_portfolio_exposure_pct == 1.0
        assert limits.max_daily_loss_pct == 0.03
        assert limits.max_drawdown_pct == 0.15
        assert limits.max_open_positions == 20
        assert limits.max_orders_per_day == 100

    def test_custom_values(self):
        limits = RiskLimits(
            max_position_pct=0.20,
            max_drawdown_pct=0.10,
            max_open_positions=5,
        )
        assert limits.max_position_pct == 0.20
        assert limits.max_drawdown_pct == 0.10
        assert limits.max_open_positions == 5

    def test_is_frozen(self):
        limits = RiskLimits()
        with pytest.raises(AttributeError):
            limits.max_position_pct = 0.50  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestRiskCheckResult
# ---------------------------------------------------------------------------


class TestRiskCheckResult:
    """Tests for the RiskCheckResult dataclass."""

    def test_approved_result(self, make_order):
        order = make_order()
        result = RiskCheckResult(approved=True, order=order)
        assert result.approved is True
        assert result.order is order
        assert result.adjusted_quantity is None
        assert result.rejection_reasons == []
        assert result.warnings == []

    def test_rejected_result(self, make_order):
        order = make_order()
        result = RiskCheckResult(
            approved=False,
            order=order,
            rejection_reasons=["Too risky"],
        )
        assert result.approved is False
        assert result.rejection_reasons == ["Too risky"]

    def test_multiple_rejection_reasons(self, make_order):
        order = make_order()
        result = RiskCheckResult(
            approved=False,
            order=order,
            rejection_reasons=["Reason 1", "Reason 2"],
        )
        assert len(result.rejection_reasons) == 2

    def test_warnings_with_approval(self, make_order):
        order = make_order()
        result = RiskCheckResult(
            approved=True,
            order=order,
            adjusted_quantity=50,
            warnings=["Quantity reduced"],
        )
        assert result.approved is True
        assert result.adjusted_quantity == 50
        assert len(result.warnings) == 1


# ---------------------------------------------------------------------------
# TestRiskManagerInit
# ---------------------------------------------------------------------------


class TestRiskManagerInit:
    """Tests for RiskManager initialization."""

    def test_default_limits(self):
        rm = RiskManager()
        assert rm.limits.max_position_pct == 0.10

    def test_custom_limits(self):
        limits = RiskLimits(max_drawdown_pct=0.05)
        rm = RiskManager(limits=limits)
        assert rm.limits.max_drawdown_pct == 0.05

    def test_initial_state(self):
        rm = RiskManager()
        assert rm.daily_pnl == 0.0
        assert rm.current_date is None
        assert rm.orders_today == 0
        assert rm.peak_equity == 0.0
        assert rm.is_halted is False


# ---------------------------------------------------------------------------
# TestRiskManagerDrawdownHalt
# ---------------------------------------------------------------------------


class TestRiskManagerDrawdownHalt:
    """Tests for the circuit breaker (max drawdown halt)."""

    def test_halt_on_max_drawdown(self, make_order, empty_positions):
        rm = RiskManager(limits=RiskLimits(max_drawdown_pct=0.10))
        rm.peak_equity = 100000.0

        # Equity dropped 15% from peak
        result = rm.check_order(
            make_order(),
            equity=85000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is False
        assert rm.is_halted is True
        assert "drawdown" in result.rejection_reasons[0].lower()

    def test_halted_rejects_all_orders(self, make_order, empty_positions):
        rm = RiskManager()
        rm.is_halted = True

        result = rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is False
        assert "halted" in result.rejection_reasons[0].lower()

    def test_halted_rejects_sell_orders(self, make_order, empty_positions):
        rm = RiskManager()
        rm.is_halted = True

        result = rm.check_order(
            make_order(side=OrderSide.SELL),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is False

    def test_reset_halt_allows_trading(self, make_order, empty_positions):
        rm = RiskManager()
        rm.is_halted = True

        rm.reset_halt()

        result = rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True

    def test_drawdown_just_below_threshold(self, make_order, empty_positions):
        rm = RiskManager(limits=RiskLimits(max_drawdown_pct=0.15))
        rm.peak_equity = 100000.0

        # 14% drawdown — just below 15% threshold
        result = rm.check_order(
            make_order(),
            equity=86000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True
        assert rm.is_halted is False

    def test_no_halt_when_peak_is_zero(self, make_order, empty_positions):
        """First order before peak is set should not trigger drawdown."""
        rm = RiskManager()
        # peak_equity is 0.0 (initial state)

        result = rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True


# ---------------------------------------------------------------------------
# TestRiskManagerDailyLossLimit
# ---------------------------------------------------------------------------


class TestRiskManagerDailyLossLimit:
    """Tests for the daily loss limit check."""

    def test_daily_loss_rejects_order(self, make_order, empty_positions):
        rm = RiskManager(limits=RiskLimits(max_daily_loss_pct=0.03))
        rm.daily_pnl = -4000.0  # -4% of 100k equity

        result = rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is False
        assert "daily loss" in result.rejection_reasons[0].lower()

    def test_daily_loss_just_below_threshold(self, make_order, empty_positions):
        rm = RiskManager(limits=RiskLimits(max_daily_loss_pct=0.03))
        rm.daily_pnl = -2500.0  # -2.5% of 100k

        result = rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True

    def test_daily_loss_resets_on_new_day(self, make_order, empty_positions):
        rm = RiskManager(limits=RiskLimits(max_daily_loss_pct=0.03))
        rm.daily_pnl = -5000.0
        rm.current_date = datetime(2023, 6, 14).date()

        # New day via update_daily_pnl resets counters
        rm.update_daily_pnl(0.0, datetime(2023, 6, 15, 9, 30))
        assert rm.daily_pnl == 0.0

        result = rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True

    def test_positive_pnl_does_not_trigger(self, make_order, empty_positions):
        rm = RiskManager(limits=RiskLimits(max_daily_loss_pct=0.03))
        rm.daily_pnl = 5000.0  # Positive P&L

        result = rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True


# ---------------------------------------------------------------------------
# TestRiskManagerOrderRateLimit
# ---------------------------------------------------------------------------


class TestRiskManagerOrderRateLimit:
    """Tests for the order rate limit check."""

    def test_rate_limit_rejects_after_max(self, make_order, empty_positions):
        rm = RiskManager(limits=RiskLimits(max_orders_per_day=3))
        rm.orders_today = 3

        result = rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is False
        assert "rate limit" in result.rejection_reasons[0].lower()

    def test_rate_limit_resets_on_new_day(self, make_order, empty_positions):
        rm = RiskManager(limits=RiskLimits(max_orders_per_day=3))
        rm.orders_today = 3
        rm.current_date = datetime(2023, 6, 14).date()

        # New day resets via update_daily_pnl
        rm.update_daily_pnl(0.0, datetime(2023, 6, 15, 9, 30))
        assert rm.orders_today == 0

        result = rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True

    def test_rejected_orders_dont_count(self, make_order, empty_positions):
        """Rejected orders should not increment the daily counter."""
        rm = RiskManager(limits=RiskLimits(max_orders_per_day=100))
        rm.is_halted = True  # Force rejection

        initial_count = rm.orders_today
        rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert rm.orders_today == initial_count


# ---------------------------------------------------------------------------
# TestRiskManagerOpenPositionLimit
# ---------------------------------------------------------------------------


class TestRiskManagerOpenPositionLimit:
    """Tests for the open position count limit."""

    def test_max_positions_rejects_new_buy(self, make_order):
        rm = RiskManager(limits=RiskLimits(max_open_positions=1))

        positions: dict[str, Any] = {
            "MSFT": {
                "quantity": 50,
                "avg_cost": 300.0,
                "market_value": 15000.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            }
        }

        # Try to buy AAPL (new position) when already at limit
        result = rm.check_order(
            make_order(symbol="AAPL"),
            equity=100000.0,
            positions=positions,
            current_price=150.0,
        )

        assert result.approved is False
        assert "open positions" in result.rejection_reasons[0].lower()

    def test_max_positions_allows_sell(self, make_order):
        rm = RiskManager(limits=RiskLimits(max_open_positions=1))

        positions: dict[str, Any] = {
            "AAPL": {
                "quantity": 50,
                "avg_cost": 150.0,
                "market_value": 7500.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            }
        }

        result = rm.check_order(
            make_order(side=OrderSide.SELL),
            equity=100000.0,
            positions=positions,
            current_price=150.0,
        )

        assert result.approved is True

    def test_adding_to_existing_position_allowed(self, make_order, aapl_position):
        """Adding to an existing position should not count as new."""
        rm = RiskManager(limits=RiskLimits(max_open_positions=1, max_position_pct=1.0))

        # AAPL already open, buying more AAPL
        result = rm.check_order(
            make_order(symbol="AAPL"),
            equity=100000.0,
            positions=aapl_position,
            current_price=150.0,
        )

        assert result.approved is True

    def test_below_limit_allows_buy(self, make_order, empty_positions):
        rm = RiskManager(limits=RiskLimits(max_open_positions=5))

        result = rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True


# ---------------------------------------------------------------------------
# TestRiskManagerPositionConcentration
# ---------------------------------------------------------------------------


class TestRiskManagerPositionConcentration:
    """Tests for position concentration limit (check 5)."""

    def test_reduces_quantity_when_over_limit(self, make_order, empty_positions):
        # Max 10% of 100k = $10,000. At $150/share, max 66 shares.
        rm = RiskManager(limits=RiskLimits(max_position_pct=0.10))

        result = rm.check_order(
            make_order(quantity=200),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True
        assert result.adjusted_quantity is not None
        assert result.adjusted_quantity == 66  # int(10000 / 150)
        assert len(result.warnings) > 0

    def test_rejects_when_existing_at_limit(self, make_order):
        rm = RiskManager(limits=RiskLimits(max_position_pct=0.10))

        # Already holding $10,000 worth (at limit)
        positions: dict[str, Any] = {
            "AAPL": {
                "quantity": 66,
                "avg_cost": 150.0,
                "market_value": 9900.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            }
        }

        result = rm.check_order(
            make_order(quantity=10),
            equity=100000.0,
            positions=positions,
            current_price=150.0,
        )

        assert result.approved is False
        assert "concentration" in result.rejection_reasons[0].lower()

    def test_no_adjustment_when_under_limit(self, make_order, empty_positions):
        rm = RiskManager(limits=RiskLimits(max_position_pct=0.50))

        result = rm.check_order(
            make_order(quantity=100),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True
        assert result.adjusted_quantity is None

    def test_sell_not_checked(self, make_order, aapl_position):
        """SELL orders bypass concentration checks."""
        rm = RiskManager(limits=RiskLimits(max_position_pct=0.01))

        result = rm.check_order(
            make_order(side=OrderSide.SELL, quantity=50),
            equity=100000.0,
            positions=aapl_position,
            current_price=150.0,
        )

        assert result.approved is True


# ---------------------------------------------------------------------------
# TestRiskManagerExposureLimit
# ---------------------------------------------------------------------------


class TestRiskManagerExposureLimit:
    """Tests for total portfolio exposure limit (check 6)."""

    def test_reduces_quantity_when_over_exposure(self, make_order):
        # Max 50% exposure on 100k = $50,000
        rm = RiskManager(
            limits=RiskLimits(
                max_portfolio_exposure_pct=0.50,
                max_position_pct=1.0,  # No concentration limit
            )
        )

        positions: dict[str, Any] = {
            "MSFT": {
                "quantity": 100,
                "avg_cost": 300.0,
                "market_value": 30000.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            }
        }

        # Want 200 AAPL at $150 = $30,000, but only $20,000 headroom
        result = rm.check_order(
            make_order(symbol="AAPL", quantity=200),
            equity=100000.0,
            positions=positions,
            current_price=150.0,
        )

        assert result.approved is True
        assert result.adjusted_quantity is not None
        # int(20000 / 150) = 133
        assert result.adjusted_quantity == 133

    def test_rejects_when_fully_exposed(self, make_order):
        rm = RiskManager(
            limits=RiskLimits(
                max_portfolio_exposure_pct=0.50,
                max_position_pct=1.0,
            )
        )

        positions: dict[str, Any] = {
            "MSFT": {
                "quantity": 200,
                "avg_cost": 300.0,
                "market_value": 60000.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            }
        }

        result = rm.check_order(
            make_order(symbol="AAPL", quantity=100),
            equity=100000.0,
            positions=positions,
            current_price=150.0,
        )

        assert result.approved is False
        assert "exposure" in result.rejection_reasons[0].lower()

    def test_no_adjustment_when_under_limit(self, make_order, empty_positions):
        rm = RiskManager(
            limits=RiskLimits(
                max_portfolio_exposure_pct=1.0,
                max_position_pct=1.0,
            )
        )

        result = rm.check_order(
            make_order(quantity=100),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True
        assert result.adjusted_quantity is None


# ---------------------------------------------------------------------------
# TestRiskManagerUpdateDailyPnl
# ---------------------------------------------------------------------------


class TestRiskManagerUpdateDailyPnl:
    """Tests for daily P&L tracking."""

    def test_accumulates_pnl(self):
        rm = RiskManager()
        ts = datetime(2023, 6, 15, 10, 0)

        rm.update_daily_pnl(100.0, ts)
        rm.update_daily_pnl(-50.0, ts)

        assert rm.daily_pnl == 50.0

    def test_resets_on_new_day(self):
        rm = RiskManager()
        rm.update_daily_pnl(-500.0, datetime(2023, 6, 14, 15, 0))
        assert rm.daily_pnl == -500.0

        # New day
        rm.update_daily_pnl(100.0, datetime(2023, 6, 15, 9, 30))
        assert rm.daily_pnl == 100.0  # Reset then added 100

    def test_first_call_sets_date(self):
        rm = RiskManager()
        assert rm.current_date is None

        rm.update_daily_pnl(50.0, datetime(2023, 6, 15, 10, 0))
        assert rm.current_date == datetime(2023, 6, 15).date()


# ---------------------------------------------------------------------------
# TestRiskManagerUpdatePeakEquity
# ---------------------------------------------------------------------------


class TestRiskManagerUpdatePeakEquity:
    """Tests for peak equity tracking."""

    def test_tracks_new_highs(self):
        rm = RiskManager()
        rm.update_peak_equity(100000.0)
        assert rm.peak_equity == 100000.0

        rm.update_peak_equity(105000.0)
        assert rm.peak_equity == 105000.0

    def test_ignores_lower_values(self):
        rm = RiskManager()
        rm.update_peak_equity(100000.0)
        rm.update_peak_equity(95000.0)
        assert rm.peak_equity == 100000.0


# ---------------------------------------------------------------------------
# TestRiskManagerReset
# ---------------------------------------------------------------------------


class TestRiskManagerReset:
    """Tests for full state reset."""

    def test_resets_all_state(self):
        rm = RiskManager()
        rm.daily_pnl = -500.0
        rm.current_date = datetime(2023, 6, 15).date()
        rm.orders_today = 10
        rm.peak_equity = 100000.0
        rm.is_halted = True

        rm.reset()

        assert rm.daily_pnl == 0.0
        assert rm.current_date is None
        assert rm.orders_today == 0
        assert rm.peak_equity == 0.0
        assert rm.is_halted is False


# ---------------------------------------------------------------------------
# TestRiskManagerMultipleChecks
# ---------------------------------------------------------------------------


class TestRiskManagerMultipleChecks:
    """Tests for interaction between multiple risk checks."""

    def test_drawdown_takes_priority(self, make_order, empty_positions):
        """Drawdown halt should reject before daily loss is even checked."""
        rm = RiskManager(
            limits=RiskLimits(max_drawdown_pct=0.10, max_daily_loss_pct=0.03)
        )
        rm.peak_equity = 100000.0
        rm.daily_pnl = -5000.0  # Also over daily limit

        result = rm.check_order(
            make_order(),
            equity=85000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is False
        # Only drawdown reason, daily loss not checked after halt
        assert len(result.rejection_reasons) == 1
        assert "drawdown" in result.rejection_reasons[0].lower()

    def test_concentration_then_exposure_adjusts(self, make_order):
        """Both concentration and exposure can progressively reduce qty."""
        rm = RiskManager(
            limits=RiskLimits(
                max_position_pct=0.20,  # $20k max per position
                max_portfolio_exposure_pct=0.15,  # $15k max total
            )
        )

        result = rm.check_order(
            make_order(quantity=200),
            equity=100000.0,
            positions={},
            current_price=150.0,
        )

        assert result.approved is True
        assert result.adjusted_quantity is not None
        # Concentration allows int(20000/150) = 133
        # Exposure allows int(15000/150) = 100
        # Exposure is more restrictive, so final = 100
        assert result.adjusted_quantity == 100

    def test_all_checks_pass_normal_order(self, make_order, empty_positions):
        """A normal, reasonable order should pass all checks."""
        rm = RiskManager()
        rm.peak_equity = 100000.0

        result = rm.check_order(
            make_order(quantity=10),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is True
        assert result.adjusted_quantity is None
        assert result.rejection_reasons == []
        assert result.warnings == []

    def test_sell_bypasses_concentration_and_exposure(self, make_order):
        """SELL orders should pass even with tight limits."""
        rm = RiskManager(
            limits=RiskLimits(
                max_position_pct=0.01,
                max_portfolio_exposure_pct=0.01,
            )
        )

        positions: dict[str, Any] = {
            "AAPL": {
                "quantity": 100,
                "avg_cost": 150.0,
                "market_value": 15000.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            }
        }

        result = rm.check_order(
            make_order(side=OrderSide.SELL, quantity=100),
            equity=100000.0,
            positions=positions,
            current_price=150.0,
        )

        assert result.approved is True

    def test_zero_equity_rejects(self, make_order, empty_positions):
        rm = RiskManager()

        result = rm.check_order(
            make_order(),
            equity=0.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert result.approved is False
        assert "zero or negative" in result.rejection_reasons[0].lower()

    def test_approved_orders_increment_counter(self, make_order, empty_positions):
        rm = RiskManager()
        assert rm.orders_today == 0

        rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert rm.orders_today == 1

        rm.check_order(
            make_order(),
            equity=100000.0,
            positions=empty_positions,
            current_price=150.0,
        )

        assert rm.orders_today == 2

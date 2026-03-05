"""Edge case tests across multiple modules.

Tests boundary conditions, degenerate inputs, and error paths for:
- Position sizing (zero equity, zero price, extreme signal strength)
- Risk manager (multi-check interactions, reset, daily rollover)
- Optimizers (single asset, short history, degenerate data)
- Correlation tracker (single symbol, identical prices)
- MultiAssetSMA (single symbol, missing data)
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.events.event import OrderEvent, OrderSide, OrderType
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer
from src.portfolio.correlation import CorrelationTracker
from src.risk.position_sizer import (
    FixedFractionSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
)
from src.risk.risk_manager import RiskLimits, RiskManager
from src.strategy.multi_asset_sma import MultiAssetSMAStrategy

# ---------------------------------------------------------------------------
# Position Sizing Edge Cases
# ---------------------------------------------------------------------------


class TestFixedFractionEdgeCases:
    def test_zero_price(self):
        sizer = FixedFractionSizer(fraction=0.10)
        result = sizer.calculate_size("AAPL", 0.0, 100_000.0, 50_000.0)
        assert result.quantity == 0

    def test_negative_price(self):
        sizer = FixedFractionSizer(fraction=0.10)
        result = sizer.calculate_size("AAPL", -10.0, 100_000.0, 50_000.0)
        assert result.quantity == 0

    def test_very_small_equity(self):
        sizer = FixedFractionSizer(fraction=0.10)
        result = sizer.calculate_size("AAPL", 150.0, 1.0, 1.0)
        assert result.quantity == 0  # 0.10 / 150 < 1

    def test_fraction_1_0(self):
        sizer = FixedFractionSizer(fraction=1.0)
        result = sizer.calculate_size("AAPL", 100.0, 10_000.0, 10_000.0)
        assert result.quantity == 100  # 10000 / 100


class TestVolatilitySizerEdgeCases:
    def test_zero_price(self):
        sizer = VolatilityBasedSizer()
        result = sizer.calculate_size("AAPL", 0.0, 100_000.0, 50_000.0)
        assert result.quantity == 0
        assert result.method == "volatility_fallback"

    def test_no_data_handler(self):
        sizer = VolatilityBasedSizer()
        result = sizer.calculate_size(
            "AAPL", 150.0, 100_000.0, 50_000.0, data_handler=None
        )
        assert result.method == "volatility_fallback"
        assert result.quantity > 0

    def test_insufficient_bars(self):
        sizer = VolatilityBasedSizer(atr_period=14)
        dh = MagicMock()
        dh.get_latest_bars.return_value = pd.DataFrame(
            {"high": [101], "low": [99], "close": [100]},
            index=pd.DatetimeIndex([datetime(2020, 1, 2)]),
        )
        result = sizer.calculate_size(
            "AAPL", 150.0, 100_000.0, 50_000.0, data_handler=dh
        )
        # Insufficient data for ATR -> fallback
        assert result.method == "volatility_fallback"


class TestKellySizerEdgeCases:
    def test_zero_price(self):
        sizer = KellyCriterionSizer()
        result = sizer.calculate_size("AAPL", 0.0, 100_000.0, 50_000.0)
        assert result.quantity == 0

    def test_signal_strength_zero(self):
        sizer = KellyCriterionSizer(min_trades=1, default_fraction=0.10)
        result = sizer.calculate_size(
            "AAPL", 100.0, 100_000.0, 50_000.0, signal_strength=0.0
        )
        assert result.quantity == 0

    def test_signal_strength_greater_than_one(self):
        sizer = KellyCriterionSizer(min_trades=1, default_fraction=0.10)
        result = sizer.calculate_size(
            "AAPL", 100.0, 100_000.0, 50_000.0, signal_strength=2.0
        )
        # Should still work, just scaled higher (clamped by max_position)
        assert result.quantity >= 0

    def test_all_losing_trades(self):
        sizer = KellyCriterionSizer(min_trades=1)
        trades = [
            SimpleNamespace(side="SELL", pnl=-100),
            SimpleNamespace(side="SELL", pnl=-200),
        ]
        result = sizer.calculate_size("AAPL", 100.0, 100_000.0, 50_000.0, trades=trades)
        # Kelly should be 0 or negative -> quantity = 0
        assert result.quantity == 0

    def test_no_trades_uses_default(self):
        sizer = KellyCriterionSizer(default_fraction=0.15)
        result = sizer.calculate_size("AAPL", 100.0, 100_000.0, 50_000.0)
        # Should use default_fraction = 0.15
        expected_qty = int(100_000 * 0.15 * 0.5 / 100)  # kelly_fraction=0.5
        assert result.quantity == expected_qty


# ---------------------------------------------------------------------------
# Risk Manager Edge Cases
# ---------------------------------------------------------------------------


def _make_order(symbol="AAPL", side=OrderSide.BUY, quantity=100):
    return OrderEvent(
        timestamp=datetime(2020, 6, 15),
        symbol=symbol,
        order_type=OrderType.MARKET,
        side=side,
        quantity=quantity,
    )


class TestRiskManagerEdgeCases:
    def test_all_checks_pass(self):
        rm = RiskManager(limits=RiskLimits())
        rm.update_peak_equity(100_000.0)
        result = rm.check_order(
            _make_order(quantity=10),
            equity=100_000.0,
            positions={},
            current_price=100.0,
        )
        assert result.approved

    def test_multiple_checks_fail(self):
        """Daily loss + order rate should both trigger rejection."""
        rm = RiskManager(
            limits=RiskLimits(max_daily_loss_pct=0.01, max_orders_per_day=1)
        )
        rm.update_peak_equity(100_000.0)
        rm.daily_pnl = -2000.0  # 2% loss
        rm.orders_today = 5  # over limit
        result = rm.check_order(
            _make_order(quantity=10),
            equity=100_000.0,
            positions={},
            current_price=100.0,
        )
        assert not result.approved
        assert len(result.rejection_reasons) >= 2

    def test_reset_after_halt(self):
        rm = RiskManager()
        rm.is_halted = True
        result = rm.check_order(
            _make_order(), equity=100_000.0, positions={}, current_price=100.0
        )
        assert not result.approved

        rm.reset_halt()
        rm.update_peak_equity(100_000.0)
        result = rm.check_order(
            _make_order(quantity=10),
            equity=100_000.0,
            positions={},
            current_price=100.0,
        )
        assert result.approved

    def test_daily_reset_on_new_day(self):
        rm = RiskManager()
        rm.update_daily_pnl(-500.0, datetime(2020, 6, 15, 10, 0))
        assert rm.daily_pnl == -500.0
        assert rm.orders_today == 0

        # New day should reset
        rm.update_daily_pnl(100.0, datetime(2020, 6, 16, 10, 0))
        assert rm.daily_pnl == 100.0  # reset to 0 + 100

    def test_zero_equity_rejects(self):
        rm = RiskManager()
        result = rm.check_order(
            _make_order(), equity=0.0, positions={}, current_price=100.0
        )
        assert not result.approved

    def test_sell_bypasses_concentration_check(self):
        rm = RiskManager(limits=RiskLimits(max_position_pct=0.01))
        rm.update_peak_equity(100_000.0)
        positions = {"AAPL": {"quantity": 100, "market_value": 15_000.0}}
        result = rm.check_order(
            _make_order(side=OrderSide.SELL, quantity=100),
            equity=100_000.0,
            positions=positions,
            current_price=150.0,
        )
        assert result.approved

    def test_full_reset(self):
        rm = RiskManager()
        rm.is_halted = True
        rm.daily_pnl = -5000.0
        rm.peak_equity = 200_000.0
        rm.orders_today = 50
        rm.current_date = datetime(2020, 1, 1).date()
        rm.reset()
        assert not rm.is_halted
        assert rm.daily_pnl == 0.0
        assert rm.peak_equity == 0.0
        assert rm.orders_today == 0
        assert rm.current_date is None


# ---------------------------------------------------------------------------
# Optimizer Edge Cases
# ---------------------------------------------------------------------------


class TestOptimizerEdgeCases:
    def test_mv_single_asset(self):
        opt = MeanVarianceOptimizer(target="max_sharpe")
        returns = pd.DataFrame(
            {"AAPL": np.random.randn(100) * 0.01},
            index=pd.bdate_range("2020-01-02", periods=100),
        )
        result = opt.optimize(["AAPL"], returns, {"AAPL": 1.0})
        assert abs(result.weights["AAPL"] - 1.0) < 0.01

    def test_rp_single_asset(self):
        opt = RiskParityOptimizer()
        returns = pd.DataFrame(
            {"AAPL": np.random.randn(100) * 0.01},
            index=pd.bdate_range("2020-01-02", periods=100),
        )
        result = opt.optimize(["AAPL"], returns, {"AAPL": 1.0})
        assert abs(result.weights["AAPL"] - 1.0) < 0.01

    def test_mv_very_short_history(self):
        opt = MeanVarianceOptimizer()
        returns = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.01, -0.01]},
            index=pd.bdate_range("2020-01-02", periods=2),
        )
        result = opt.optimize(["A", "B"], returns, {"A": 0.5, "B": 0.5})
        # Should fallback to equal weights
        assert abs(result.weights["A"] - 0.5) < 0.1
        assert abs(result.weights["B"] - 0.5) < 0.1

    def test_rp_very_short_history(self):
        opt = RiskParityOptimizer()
        returns = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.01, -0.01]},
            index=pd.bdate_range("2020-01-02", periods=2),
        )
        result = opt.optimize(["A", "B"], returns, {"A": 0.5, "B": 0.5})
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_mv_all_zero_returns(self):
        opt = MeanVarianceOptimizer()
        returns = pd.DataFrame(
            {"A": np.zeros(50), "B": np.zeros(50)},
            index=pd.bdate_range("2020-01-02", periods=50),
        )
        result = opt.optimize(["A", "B"], returns, {"A": 0.5, "B": 0.5})
        # Should produce valid weights
        assert abs(sum(result.weights.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Correlation Tracker Edge Cases
# ---------------------------------------------------------------------------


class TestCorrelationEdgeCases:
    def test_single_symbol(self):
        ct = CorrelationTracker(window=60, min_periods=5)
        for i in range(10):
            ct.update("AAPL", 100.0 + i)
        matrix = ct.calculate_matrix()
        assert matrix.shape == (1, 1)
        assert matrix.loc["AAPL", "AAPL"] == 1.0

    def test_identical_prices(self):
        ct = CorrelationTracker(window=60, min_periods=5)
        for _i in range(30):
            ct.update("A", 100.0)
            ct.update("B", 100.0)
        corr = ct.get_correlation("A", "B")
        # Zero variance -> returns 0.0
        assert corr == 0.0

    def test_min_periods_not_met(self):
        ct = CorrelationTracker(window=60, min_periods=30)
        for i in range(10):
            ct.update("A", 100.0 + i)
            ct.update("B", 200.0 - i)
        corr = ct.get_correlation("A", "B")
        assert corr is None

    def test_no_symbols(self):
        ct = CorrelationTracker()
        matrix = ct.calculate_matrix()
        assert matrix.empty

    def test_same_symbol_correlation(self):
        ct = CorrelationTracker()
        assert ct.get_correlation("X", "X") == 1.0

    def test_unknown_symbol(self):
        ct = CorrelationTracker()
        assert ct.get_correlation("X", "Y") is None

    def test_window_trimming(self):
        ct = CorrelationTracker(window=10, min_periods=5)
        for i in range(20):
            ct.update("A", 100.0 + i)
        assert len(ct._prices["A"]) == 10


# ---------------------------------------------------------------------------
# MultiAssetSMA Edge Cases
# ---------------------------------------------------------------------------


class TestMultiAssetSMAEdgeCases:
    def test_single_symbol(self):
        strat = MultiAssetSMAStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 5, "long_window": 10},
        )
        assert strat.target_weights == {"AAPL": 1.0}

    def test_empty_symbols(self):
        strat = MultiAssetSMAStrategy(
            symbols=[],
            parameters={"short_window": 5, "long_window": 10},
        )
        assert strat.target_weights == {}

    def test_invalid_windows(self):
        with pytest.raises(ValueError, match="short_window must be less"):
            MultiAssetSMAStrategy(
                symbols=["AAPL"],
                parameters={"short_window": 50, "long_window": 10},
            )

    def test_rebalance_frequency_validation(self):
        with pytest.raises(ValueError, match="rebalance_frequency must be"):
            MultiAssetSMAStrategy(
                symbols=["AAPL"],
                parameters={
                    "short_window": 5,
                    "long_window": 10,
                    "rebalance_frequency": 0,
                },
            )

    def test_get_sma_values_before_data(self):
        strat = MultiAssetSMAStrategy(
            symbols=["AAPL"], parameters={"short_window": 5, "long_window": 10}
        )
        short, long_ = strat.get_sma_values("AAPL")
        assert short is None
        assert long_ is None

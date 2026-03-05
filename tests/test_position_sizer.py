"""Unit tests for position sizing algorithms."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.portfolio.portfolio import Trade
from src.risk.position_sizer import (
    FixedFractionSizer,
    KellyCriterionSizer,
    PositionSizer,
    SizingResult,
    VolatilityBasedSizer,
)


class TestSizingResult:
    """Tests for the SizingResult dataclass."""

    def test_sizing_result_creation(self):
        """Test basic SizingResult creation."""
        result = SizingResult(quantity=100, target_value=15000.0, method="test")

        assert result.quantity == 100
        assert result.target_value == 15000.0
        assert result.method == "test"

    def test_sizing_result_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        result = SizingResult(quantity=50, target_value=7500.0, method="test")

        assert result.metadata == {}

    def test_sizing_result_with_metadata(self):
        """Test SizingResult with custom metadata."""
        result = SizingResult(
            quantity=50,
            target_value=7500.0,
            method="test",
            metadata={"atr": 3.5, "risk": 0.02},
        )

        assert result.metadata["atr"] == 3.5
        assert result.metadata["risk"] == 0.02

    def test_sizing_result_is_frozen(self):
        """Test that SizingResult is immutable."""
        result = SizingResult(quantity=100, target_value=15000.0, method="test")

        with pytest.raises(AttributeError):
            result.quantity = 200  # type: ignore[misc]


class TestFixedFractionSizer:
    """Tests for the FixedFractionSizer class."""

    def test_default_fraction(self):
        """Test default fraction is 0.10."""
        sizer = FixedFractionSizer()
        assert sizer.fraction == 0.10

    def test_custom_fraction(self):
        """Test custom fraction value."""
        sizer = FixedFractionSizer(fraction=0.20)
        assert sizer.fraction == 0.20

    def test_invalid_fraction_zero(self):
        """Test ValueError for zero fraction."""
        with pytest.raises(ValueError, match="fraction must be between 0 and 1"):
            FixedFractionSizer(fraction=0.0)

    def test_invalid_fraction_negative(self):
        """Test ValueError for negative fraction."""
        with pytest.raises(ValueError, match="fraction must be between 0 and 1"):
            FixedFractionSizer(fraction=-0.1)

    def test_invalid_fraction_above_one(self):
        """Test ValueError for fraction > 1."""
        with pytest.raises(ValueError, match="fraction must be between 0 and 1"):
            FixedFractionSizer(fraction=1.5)

    def test_fraction_exactly_one(self):
        """Test that fraction=1.0 is valid."""
        sizer = FixedFractionSizer(fraction=1.0)
        assert sizer.fraction == 1.0

    def test_basic_calculation(self):
        """Test basic position size calculation."""
        sizer = FixedFractionSizer(fraction=0.10)
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
        )

        # int(100000 * 0.10 / 150) = int(66.666) = 66
        assert result.quantity == 66

    def test_matches_phase1_logic(self):
        """Verify FixedFractionSizer produces identical results to Phase 1."""
        sizer = FixedFractionSizer(fraction=0.10)
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
        )

        # Phase 1 logic: int(100000 * 0.10 / 150) = 66
        phase1_quantity = int(100000.0 * 0.10 / 150.0)
        assert result.quantity == phase1_quantity

    def test_returns_sizing_result(self):
        """Test return type is SizingResult."""
        sizer = FixedFractionSizer()
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
        )

        assert isinstance(result, SizingResult)

    def test_method_name(self):
        """Test that method is 'fixed_fraction'."""
        sizer = FixedFractionSizer()
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
        )

        assert result.method == "fixed_fraction"

    def test_zero_price_returns_zero(self):
        """Test that zero price returns quantity 0."""
        sizer = FixedFractionSizer()
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=0.0,
            equity=100000.0,
            cash=100000.0,
        )

        assert result.quantity == 0

    def test_expensive_stock_returns_zero(self):
        """Test when stock price exceeds target value."""
        sizer = FixedFractionSizer(fraction=0.01)
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=2000.0,
            equity=100000.0,
            cash=100000.0,
        )

        # int(100000 * 0.01 / 2000) = int(0.5) = 0
        assert result.quantity == 0

    def test_ignores_data_handler(self):
        """Test that data_handler=None works fine."""
        sizer = FixedFractionSizer()
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            data_handler=None,
        )

        assert result.quantity == 66

    def test_metadata_contains_fraction(self):
        """Test that metadata includes the fraction used."""
        sizer = FixedFractionSizer(fraction=0.15)
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=100.0,
            equity=50000.0,
            cash=50000.0,
        )

        assert result.metadata["fraction"] == 0.15

    def test_different_equity_values(self):
        """Test sizing with various equity levels."""
        sizer = FixedFractionSizer(fraction=0.10)

        result_small = sizer.calculate_size(
            symbol="AAPL", current_price=100.0, equity=10000.0, cash=10000.0
        )
        result_large = sizer.calculate_size(
            symbol="AAPL", current_price=100.0, equity=1000000.0, cash=1000000.0
        )

        # int(10000 * 0.10 / 100) = 10
        assert result_small.quantity == 10
        # int(1000000 * 0.10 / 100) = 1000
        assert result_large.quantity == 1000


class TestVolatilityBasedSizer:
    """Tests for the VolatilityBasedSizer class."""

    @pytest.fixture
    def mock_data_handler_stable(self):
        """Create mock data handler with stable OHLCV data (known ATR=3.0)."""
        handler = MagicMock()
        dates = pd.date_range("2023-01-01", periods=20, freq="B")
        data = pd.DataFrame(
            {
                "open": [150.0] * 20,
                "high": [152.0] * 20,
                "low": [149.0] * 20,
                "close": [151.0] * 20,
                "volume": [1000000] * 20,
            },
            index=dates,
        )
        handler.get_latest_bars.return_value = data
        return handler

    @pytest.fixture
    def mock_data_handler_volatile(self):
        """Create mock data handler with high volatility (ATR=10.0)."""
        handler = MagicMock()
        dates = pd.date_range("2023-01-01", periods=20, freq="B")
        data = pd.DataFrame(
            {
                "open": [150.0] * 20,
                "high": [160.0] * 20,
                "low": [150.0] * 20,
                "close": [155.0] * 20,
                "volume": [1000000] * 20,
            },
            index=dates,
        )
        handler.get_latest_bars.return_value = data
        return handler

    def test_default_parameters(self):
        """Test default parameter values."""
        sizer = VolatilityBasedSizer()

        assert sizer.risk_fraction == 0.02
        assert sizer.atr_period == 14
        assert sizer.atr_multiplier == 2.0
        assert sizer.max_position_fraction == 0.20

    def test_invalid_risk_fraction(self):
        """Test ValueError for invalid risk_fraction."""
        with pytest.raises(ValueError, match="risk_fraction"):
            VolatilityBasedSizer(risk_fraction=0.0)

    def test_invalid_atr_period(self):
        """Test ValueError for invalid atr_period."""
        with pytest.raises(ValueError, match="atr_period"):
            VolatilityBasedSizer(atr_period=0)

    def test_invalid_atr_multiplier(self):
        """Test ValueError for invalid atr_multiplier."""
        with pytest.raises(ValueError, match="atr_multiplier"):
            VolatilityBasedSizer(atr_multiplier=0.0)

    def test_basic_atr_sizing(self, mock_data_handler_stable):
        """Test sizing with known ATR value.

        With stable data: high=152, low=149, TR=3.0, ATR=3.0
        risk_per_share = 3.0 * 2.0 = 6.0
        max_risk = 100000 * 0.02 = 2000
        raw_quantity = int(2000 / 6.0) = 333
        max_position cap = int(100000 * 0.50 / 151) = 331
        quantity = min(333, 331) = 331
        """
        sizer = VolatilityBasedSizer(
            risk_fraction=0.02,
            atr_period=14,
            atr_multiplier=2.0,
            max_position_fraction=0.50,  # Generous cap so risk is the binding constraint
        )
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=151.0,
            equity=100000.0,
            cash=100000.0,
            data_handler=mock_data_handler_stable,
        )

        assert result.quantity == 331
        assert result.method == "volatility_atr"

    def test_metadata_contains_atr(self, mock_data_handler_stable):
        """Test that metadata includes ATR information."""
        sizer = VolatilityBasedSizer()
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=151.0,
            equity=100000.0,
            cash=100000.0,
            data_handler=mock_data_handler_stable,
        )

        assert "atr" in result.metadata
        assert abs(result.metadata["atr"] - 3.0) < 0.01
        assert "risk_per_share" in result.metadata
        assert "max_risk" in result.metadata

    def test_high_volatility_reduces_size(
        self, mock_data_handler_stable, mock_data_handler_volatile
    ):
        """Test that higher volatility produces fewer shares."""
        sizer = VolatilityBasedSizer(risk_fraction=0.02, atr_multiplier=2.0)

        result_stable = sizer.calculate_size(
            symbol="AAPL",
            current_price=151.0,
            equity=100000.0,
            cash=100000.0,
            data_handler=mock_data_handler_stable,
        )
        result_volatile = sizer.calculate_size(
            symbol="AAPL",
            current_price=155.0,
            equity=100000.0,
            cash=100000.0,
            data_handler=mock_data_handler_volatile,
        )

        assert result_volatile.quantity < result_stable.quantity

    def test_fallback_without_data_handler(self):
        """Test fallback sizing when no data_handler provided."""
        sizer = VolatilityBasedSizer(risk_fraction=0.02)
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            data_handler=None,
        )

        assert result.method == "volatility_fallback"
        # int(100000 * 0.02 / 150) = 13
        assert result.quantity == 13

    def test_fallback_insufficient_bars(self):
        """Test fallback when data handler has too few bars."""
        handler = MagicMock()
        handler.get_latest_bars.return_value = pd.DataFrame(
            {"open": [150.0], "high": [152.0], "low": [149.0], "close": [151.0]},
            index=[pd.Timestamp("2023-01-01")],
        )

        sizer = VolatilityBasedSizer(atr_period=14)
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=151.0,
            equity=100000.0,
            cash=100000.0,
            data_handler=handler,
        )

        assert result.method == "volatility_fallback"

    def test_max_position_cap(self, mock_data_handler_stable):
        """Test that max_position_fraction caps the position size."""
        sizer = VolatilityBasedSizer(
            risk_fraction=0.10,  # Very aggressive
            atr_multiplier=0.1,  # Tiny multiplier = huge position
            max_position_fraction=0.05,  # But capped at 5%
        )
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=151.0,
            equity=100000.0,
            cash=100000.0,
            data_handler=mock_data_handler_stable,
        )

        # Max: int(100000 * 0.05 / 151) = 33
        max_quantity = int(100000 * 0.05 / 151.0)
        assert result.quantity <= max_quantity

    def test_zero_price_returns_zero(self):
        """Test that zero price returns quantity 0."""
        sizer = VolatilityBasedSizer()
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=0.0,
            equity=100000.0,
            cash=100000.0,
        )

        assert result.quantity == 0


class TestKellyCriterionSizer:
    """Tests for the KellyCriterionSizer class."""

    @staticmethod
    def _make_trades(
        wins: int, win_amount: float, losses: int, loss_amount: float
    ) -> list[Trade]:
        """Helper to create a list of Trade objects with known P&L."""
        trades: list[Trade] = []
        now = datetime.now()

        for _i in range(wins):
            # Buy trade (no P&L)
            trades.append(
                Trade(
                    timestamp=now,
                    symbol="AAPL",
                    side="BUY",
                    quantity=100,
                    price=150.0,
                    commission=1.0,
                )
            )
            # Winning sell trade
            trades.append(
                Trade(
                    timestamp=now,
                    symbol="AAPL",
                    side="SELL",
                    quantity=100,
                    price=150.0 + win_amount / 100,
                    commission=1.0,
                    pnl=win_amount,
                )
            )

        for _i in range(losses):
            trades.append(
                Trade(
                    timestamp=now,
                    symbol="AAPL",
                    side="BUY",
                    quantity=100,
                    price=150.0,
                    commission=1.0,
                )
            )
            trades.append(
                Trade(
                    timestamp=now,
                    symbol="AAPL",
                    side="SELL",
                    quantity=100,
                    price=150.0 - abs(loss_amount) / 100,
                    commission=1.0,
                    pnl=-abs(loss_amount),
                )
            )

        return trades

    def test_default_parameters(self):
        """Test default parameter values."""
        sizer = KellyCriterionSizer()

        assert sizer.kelly_fraction == 0.5
        assert sizer.min_trades == 20
        assert sizer.max_position_fraction == 0.25
        assert sizer.default_fraction == 0.10

    def test_invalid_kelly_fraction(self):
        """Test ValueError for invalid kelly_fraction."""
        with pytest.raises(ValueError, match="kelly_fraction"):
            KellyCriterionSizer(kelly_fraction=0.0)

    def test_invalid_min_trades(self):
        """Test ValueError for invalid min_trades."""
        with pytest.raises(ValueError, match="min_trades"):
            KellyCriterionSizer(min_trades=0)

    def test_no_trade_history_uses_default(self):
        """Test that no trade history falls back to default fraction."""
        sizer = KellyCriterionSizer(default_fraction=0.10)
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=None,
        )

        # Should use default_fraction=0.10: int(100000 * 0.10 * 0.5 / 150) = 33
        assert result.quantity > 0
        assert result.metadata["raw_kelly_pct"] == 0.10

    def test_empty_trades_uses_default(self):
        """Test that empty trade list falls back to default fraction."""
        sizer = KellyCriterionSizer(default_fraction=0.10)
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=[],
        )

        assert result.metadata["raw_kelly_pct"] == 0.10

    def test_insufficient_trades_uses_default(self):
        """Test fallback when fewer trades than min_trades."""
        trades = self._make_trades(wins=5, win_amount=500, losses=3, loss_amount=300)

        sizer = KellyCriterionSizer(min_trades=20, default_fraction=0.10)
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=trades,
        )

        # Only 8 sell trades, min_trades=20 -> uses default
        assert result.metadata["raw_kelly_pct"] == 0.10

    def test_all_winners_uses_max_fraction(self):
        """Test that all winning trades uses max_position_fraction."""
        trades = self._make_trades(wins=25, win_amount=500, losses=0, loss_amount=0)

        sizer = KellyCriterionSizer(
            min_trades=20, max_position_fraction=0.25, kelly_fraction=1.0
        )
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=trades,
        )

        # All winners -> raw kelly = max_position_fraction = 0.25
        assert result.metadata["raw_kelly_pct"] == 0.25

    def test_all_losers_returns_zero(self):
        """Test that all losing trades returns quantity 0."""
        trades = self._make_trades(wins=0, win_amount=0, losses=25, loss_amount=300)

        sizer = KellyCriterionSizer(min_trades=20)
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=trades,
        )

        assert result.quantity == 0
        assert result.metadata["raw_kelly_pct"] == 0.0

    def test_known_kelly_calculation(self):
        """Test Kelly formula with known win rate and win/loss ratio.

        60% win rate, avg_win=600, avg_loss=400 -> R=1.5
        Kelly = 0.60 - (1-0.60)/1.5 = 0.60 - 0.2667 = 0.3333
        """
        trades = self._make_trades(
            wins=15, win_amount=600.0, losses=10, loss_amount=400.0
        )

        sizer = KellyCriterionSizer(
            kelly_fraction=1.0,  # Full Kelly for this test
            min_trades=20,
        )
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=trades,
        )

        raw_kelly = result.metadata["raw_kelly_pct"]
        # Kelly = 0.60 - 0.40/1.5 = 0.3333
        assert abs(raw_kelly - 0.3333) < 0.01

    def test_half_kelly_halves_size(self):
        """Test that kelly_fraction=0.5 halves the raw Kelly fraction.

        Raw Kelly = 0.3333, so:
        - Full Kelly (uncapped): adjusted = 0.3333 * 1.0 = 0.3333
        - Half Kelly: adjusted = 0.3333 * 0.5 = 0.1667
        """
        trades = self._make_trades(
            wins=15, win_amount=600.0, losses=10, loss_amount=400.0
        )

        # Use high max_position_fraction so capping doesn't interfere
        sizer_full = KellyCriterionSizer(
            kelly_fraction=1.0, min_trades=20, max_position_fraction=0.50
        )
        sizer_half = KellyCriterionSizer(
            kelly_fraction=0.5, min_trades=20, max_position_fraction=0.50
        )

        result_full = sizer_full.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=trades,
        )
        result_half = sizer_half.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=trades,
        )

        # Half Kelly adjusted fraction should be half of full Kelly
        assert (
            abs(
                result_half.metadata["adjusted_fraction"]
                - result_full.metadata["adjusted_fraction"] / 2
            )
            < 0.01
        )

    def test_signal_strength_scaling(self):
        """Test that signal_strength scales the position size."""
        trades = self._make_trades(
            wins=15, win_amount=600.0, losses=10, loss_amount=400.0
        )

        sizer = KellyCriterionSizer(kelly_fraction=1.0, min_trades=20)

        result_full = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=trades,
            signal_strength=1.0,
        )
        result_half = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=trades,
            signal_strength=0.5,
        )

        assert result_half.metadata["signal_strength"] == 0.5
        assert result_half.quantity < result_full.quantity

    def test_max_position_cap(self):
        """Test that max_position_fraction caps the output."""
        trades = self._make_trades(
            wins=25, win_amount=1000.0, losses=1, loss_amount=100.0
        )

        sizer = KellyCriterionSizer(
            kelly_fraction=1.0,
            min_trades=20,
            max_position_fraction=0.05,
        )
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
            trades=trades,
        )

        # Adjusted fraction should not exceed max_position_fraction
        assert result.metadata["adjusted_fraction"] <= 0.05

    def test_method_name(self):
        """Test that method is 'kelly_criterion'."""
        sizer = KellyCriterionSizer()
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            equity=100000.0,
            cash=100000.0,
        )

        assert result.method == "kelly_criterion"

    def test_zero_price_returns_zero(self):
        """Test that zero price returns quantity 0."""
        sizer = KellyCriterionSizer()
        result = sizer.calculate_size(
            symbol="AAPL",
            current_price=0.0,
            equity=100000.0,
            cash=100000.0,
        )

        assert result.quantity == 0


class TestPositionSizerProtocol:
    """Tests for PositionSizer abstract base class."""

    def test_fixed_fraction_is_position_sizer(self):
        """Test that FixedFractionSizer is a PositionSizer."""
        sizer = FixedFractionSizer()
        assert isinstance(sizer, PositionSizer)

    def test_volatility_is_position_sizer(self):
        """Test that VolatilityBasedSizer is a PositionSizer."""
        sizer = VolatilityBasedSizer()
        assert isinstance(sizer, PositionSizer)

    def test_kelly_is_position_sizer(self):
        """Test that KellyCriterionSizer is a PositionSizer."""
        sizer = KellyCriterionSizer()
        assert isinstance(sizer, PositionSizer)

    def test_cannot_instantiate_abstract(self):
        """Test that PositionSizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PositionSizer()  # type: ignore[abstract]

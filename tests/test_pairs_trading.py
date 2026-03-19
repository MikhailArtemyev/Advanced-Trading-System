"""Tests for PairsTradingStrategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.events.event import SignalType
from src.events.queue import EventQueue
from src.strategy.pairs_trading import PairsTradingStrategy
from tests.helpers import FakeDataHandler, make_ohlcv


def _make_prices(values: list[float]) -> pd.DataFrame:
    return make_ohlcv(values)


def _cointegrated_pair(
    n: int,
    base: float = 100.0,
    spread_shock: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a cointegrated pair A ≈ B + noise, with optional spread shock at end."""
    np.random.seed(42)
    b_vals = base + np.cumsum(np.random.randn(n) * 0.5)
    noise = np.random.randn(n) * 0.3
    a_vals = b_vals + noise
    if spread_shock != 0:
        a_vals[-1] += spread_shock
    return _make_prices(a_vals.tolist()), _make_prices(b_vals.tolist())


# ===========================================================================
# Init
# ===========================================================================


class TestInit:
    def test_requires_two_symbols(self):
        with pytest.raises(ValueError, match="exactly 2"):
            PairsTradingStrategy(symbols=["A"])

    def test_rejects_three_symbols(self):
        with pytest.raises(ValueError, match="exactly 2"):
            PairsTradingStrategy(symbols=["A", "B", "C"])

    def test_defaults(self):
        s = PairsTradingStrategy(symbols=["A", "B"])
        assert s.lookback_period == 60
        assert s.entry_threshold == 2.0
        assert s.exit_threshold == 0.5
        assert s.min_bars == 60
        assert s.symbol_a == "A"
        assert s.symbol_b == "B"

    def test_custom_params(self):
        s = PairsTradingStrategy(
            symbols=["X", "Y"],
            parameters={"lookback_period": 30, "entry_threshold": 1.5, "min_bars": 10},
        )
        assert s.lookback_period == 30
        assert s.entry_threshold == 1.5
        assert s.min_bars == 10

    def test_lookback_too_small(self):
        with pytest.raises(ValueError, match="lookback_period"):
            PairsTradingStrategy(symbols=["A", "B"], parameters={"lookback_period": 1})

    def test_entry_threshold_nonpositive(self):
        with pytest.raises(ValueError, match="entry_threshold"):
            PairsTradingStrategy(symbols=["A", "B"], parameters={"entry_threshold": 0})


# ===========================================================================
# Hedge ratio
# ===========================================================================


class TestHedgeRatio:
    def test_perfect_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        hr = PairsTradingStrategy._calculate_hedge_ratio(a, b)
        assert abs(hr - 0.5) < 1e-10  # A = 0.5 * B

    def test_no_variance_returns_one(self):
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([3.0, 3.0, 3.0])
        hr = PairsTradingStrategy._calculate_hedge_ratio(a, b)
        assert hr == 1.0

    def test_negative_relationship(self):
        a = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        hr = PairsTradingStrategy._calculate_hedge_ratio(a, b)
        assert hr < 0


# ===========================================================================
# Signal generation
# ===========================================================================


class TestSignals:
    def _make_strategy(self, **kwargs):
        params = {"lookback_period": 20, "min_bars": 1, "entry_threshold": 2.0}
        params.update(kwargs)
        s = PairsTradingStrategy(symbols=["A", "B"], parameters=params)
        eq = EventQueue()
        s.set_event_queue(eq)
        return s

    def test_no_signals_before_min_bars(self):
        s = PairsTradingStrategy(
            symbols=["A", "B"],
            parameters={"min_bars": 100},
        )
        a, b = _cointegrated_pair(50)
        dh = FakeDataHandler({"A": a, "B": b})
        signals = s.calculate_signals(datetime.now(), dh)
        assert signals == []

    def test_no_signal_when_spread_normal(self):
        """Cointegrated pair with no shock → spread near zero → no entry."""
        s = self._make_strategy(lookback_period=30, entry_threshold=2.0)
        a, b = _cointegrated_pair(30, spread_shock=0.0)
        dh = FakeDataHandler({"A": a, "B": b})
        signals = s.calculate_signals(datetime.now(), dh)
        assert signals == []

    def test_entry_on_spread_high(self):
        """Large positive spread shock → short A, long B."""
        s = self._make_strategy(lookback_period=20, entry_threshold=1.0)
        a, b = _cointegrated_pair(20, spread_shock=20.0)
        dh = FakeDataHandler({"A": a, "B": b})
        signals = s.calculate_signals(datetime.now(), dh)
        assert len(signals) == 2
        sig_a = next(sig for sig in signals if sig.symbol == "A")
        sig_b = next(sig for sig in signals if sig.symbol == "B")
        assert sig_a.signal_type == SignalType.SHORT
        assert sig_b.signal_type == SignalType.LONG

    def test_entry_on_spread_low(self):
        """Large negative spread shock → long A, short B."""
        s = self._make_strategy(lookback_period=20, entry_threshold=1.0)
        a, b = _cointegrated_pair(20, spread_shock=-20.0)
        dh = FakeDataHandler({"A": a, "B": b})
        signals = s.calculate_signals(datetime.now(), dh)
        assert len(signals) == 2
        sig_a = next(sig for sig in signals if sig.symbol == "A")
        sig_b = next(sig for sig in signals if sig.symbol == "B")
        assert sig_a.signal_type == SignalType.LONG
        assert sig_b.signal_type == SignalType.SHORT

    def test_exit_on_reversion(self):
        """After entry, spread reverts → EXIT both legs."""
        s = self._make_strategy(
            lookback_period=20, entry_threshold=1.0, exit_threshold=2.0
        )
        # First: enter on spread shock
        a, b = _cointegrated_pair(20, spread_shock=20.0)
        dh = FakeDataHandler({"A": a, "B": b})
        entry_signals = s.calculate_signals(datetime.now(), dh)
        assert len(entry_signals) == 2

        # Simulate fill
        s.current_positions["A"] = -100
        s.current_positions["B"] = 100

        # Deterministic data: A = 2*B + small noise, last spread at mean
        # This ensures hedge_ratio ≈ 2, spread has std > 0, and z ≈ 0
        vals_b = [50.0 + i * 0.1 for i in range(20)]
        vals_a = [2 * v + (0.3 if i % 3 == 0 else -0.3) for i, v in enumerate(vals_b)]
        # Set last A value so spread[-1] = mean(spread)
        hr_approx = 2.0
        spread_prev = [vals_a[i] - hr_approx * vals_b[i] for i in range(19)]
        mean_spread = sum(spread_prev) / len(spread_prev)
        vals_a[-1] = mean_spread + hr_approx * vals_b[-1]
        a2 = _make_prices(vals_a)
        b2 = _make_prices(vals_b)
        dh2 = FakeDataHandler({"A": a2, "B": b2})
        exit_signals = s.calculate_signals(datetime.now(), dh2)
        assert len(exit_signals) == 2
        assert all(sig.signal_type == SignalType.EXIT for sig in exit_signals)

    def test_no_entry_when_already_in_position(self):
        s = self._make_strategy(lookback_period=20, entry_threshold=1.0)
        s.current_positions["A"] = 100
        s._in_position = True
        a, b = _cointegrated_pair(20, spread_shock=-20.0)
        dh = FakeDataHandler({"A": a, "B": b})
        signals = s.calculate_signals(datetime.now(), dh)
        # Should not get entry signals — already in position
        # (might get exit if spread reverted, but with this shock it won't)
        entry_sigs = [
            sig
            for sig in signals
            if sig.signal_type in (SignalType.LONG, SignalType.SHORT)
        ]
        assert entry_sigs == []

    def test_insufficient_data(self):
        s = self._make_strategy(lookback_period=50)
        a = _make_prices([100.0] * 10)
        b = _make_prices([100.0] * 10)
        dh = FakeDataHandler({"A": a, "B": b})
        signals = s.calculate_signals(datetime.now(), dh)
        assert signals == []

    def test_hedge_ratio_property(self):
        s = self._make_strategy(lookback_period=20)
        a, b = _cointegrated_pair(20, spread_shock=20.0)
        dh = FakeDataHandler({"A": a, "B": b})
        s.calculate_signals(datetime.now(), dh)
        # Hedge ratio should be computed
        assert isinstance(s.hedge_ratio, float)


# ===========================================================================
# Module exports
# ===========================================================================


class TestExports:
    def test_importable(self):
        from src.strategy import PairsTradingStrategy

        assert PairsTradingStrategy is not None

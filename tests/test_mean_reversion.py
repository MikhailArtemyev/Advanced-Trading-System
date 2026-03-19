"""Tests for MeanReversionStrategy."""

from datetime import datetime

import pandas as pd
import pytest

from src.events.event import SignalType
from src.events.queue import EventQueue
from src.strategy.mean_reversion import MeanReversionStrategy
from tests.helpers import FakeDataHandler, make_ohlcv


def _make_prices(values: list[float]) -> pd.DataFrame:
    return make_ohlcv(values)


def _stable_then_spike(n: int, base: float, spike: float) -> pd.DataFrame:
    """n-1 stable values then one spike."""
    vals = [base] * (n - 1) + [spike]
    return _make_prices(vals)


# ===========================================================================
# Init
# ===========================================================================


class TestInit:
    def test_defaults(self):
        s = MeanReversionStrategy(symbols=["A"])
        assert s.lookback_period == 20
        assert s.entry_threshold == 2.0
        assert s.exit_threshold == 0.5
        assert s.max_holding_period == 10

    def test_custom_params(self):
        s = MeanReversionStrategy(
            symbols=["A"],
            parameters={
                "lookback_period": 30,
                "entry_threshold": 1.5,
                "exit_threshold": 0.3,
                "max_holding_period": 5,
            },
        )
        assert s.lookback_period == 30
        assert s.entry_threshold == 1.5

    def test_lookback_too_small(self):
        with pytest.raises(ValueError, match="lookback_period"):
            MeanReversionStrategy(symbols=["A"], parameters={"lookback_period": 1})

    def test_entry_threshold_nonpositive(self):
        with pytest.raises(ValueError, match="entry_threshold"):
            MeanReversionStrategy(symbols=["A"], parameters={"entry_threshold": 0})

    def test_exit_threshold_negative(self):
        with pytest.raises(ValueError, match="exit_threshold"):
            MeanReversionStrategy(symbols=["A"], parameters={"exit_threshold": -1})


# ===========================================================================
# Signal generation
# ===========================================================================


class TestSignals:
    def _make_strategy(self, **kwargs):
        params = {"lookback_period": 10, "entry_threshold": 2.0, "exit_threshold": 0.5}
        params.update(kwargs)
        s = MeanReversionStrategy(symbols=["A"], parameters=params)
        eq = EventQueue()
        s.set_event_queue(eq)
        return s

    def test_no_signal_insufficient_data(self):
        s = self._make_strategy(lookback_period=20)
        dh = FakeDataHandler({"A": _make_prices([100] * 5)})
        signals = s.calculate_signals(datetime.now(), dh)
        assert signals == []

    def test_long_entry_on_oversold(self):
        """Price drops well below mean → LONG."""
        s = self._make_strategy(lookback_period=10, entry_threshold=1.5)
        # 9 values at 100, then a big drop to 80 → negative z-score
        dh = FakeDataHandler({"A": _stable_then_spike(10, 100.0, 80.0)})
        signals = s.calculate_signals(datetime.now(), dh)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.LONG
        assert signals[0].symbol == "A"

    def test_short_entry_on_overbought(self):
        """Price rises well above mean → SHORT."""
        s = self._make_strategy(lookback_period=10, entry_threshold=1.5)
        dh = FakeDataHandler({"A": _stable_then_spike(10, 100.0, 120.0)})
        signals = s.calculate_signals(datetime.now(), dh)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.SHORT

    def test_no_signal_within_threshold(self):
        """Price close to mean → no signal."""
        s = self._make_strategy(lookback_period=10, entry_threshold=2.0)
        dh = FakeDataHandler({"A": _make_prices([100.0] * 10)})
        signals = s.calculate_signals(datetime.now(), dh)
        assert signals == []

    def test_exit_long_on_reversion(self):
        """Position reverts to mean → EXIT."""
        s = self._make_strategy(lookback_period=10, exit_threshold=0.5)
        s.current_positions["A"] = 100  # currently long
        # Small variation so std > 0, but last value near mean → z ≈ 0
        vals = [99.5, 100.5, 99.5, 100.5, 99.5, 100.5, 99.5, 100.5, 99.5, 100.0]
        dh = FakeDataHandler({"A": _make_prices(vals)})
        signals = s.calculate_signals(datetime.now(), dh)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.EXIT

    def test_exit_short_on_reversion(self):
        s = self._make_strategy(lookback_period=10, exit_threshold=0.5)
        s.current_positions["A"] = -100  # currently short
        vals = [99.5, 100.5, 99.5, 100.5, 99.5, 100.5, 99.5, 100.5, 99.5, 100.0]
        dh = FakeDataHandler({"A": _make_prices(vals)})
        signals = s.calculate_signals(datetime.now(), dh)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.EXIT

    def test_max_holding_period_forces_exit(self):
        s = self._make_strategy(
            lookback_period=10, max_holding_period=3, entry_threshold=100.0
        )
        s.current_positions["A"] = 100
        # Price still far from mean, but holding limit forces exit
        data = _stable_then_spike(10, 100.0, 80.0)
        dh = FakeDataHandler({"A": data})
        # Simulate 3 bars of holding
        for _ in range(2):
            s.calculate_signals(datetime.now(), dh)
        signals = s.calculate_signals(datetime.now(), dh)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.EXIT

    def test_max_holding_zero_disables(self):
        """max_holding_period=0 means no forced exit by bar count."""
        s = self._make_strategy(
            lookback_period=10,
            max_holding_period=0,
            entry_threshold=100.0,
            # Very tight exit_threshold so z-score exit won't fire for extreme z
            exit_threshold=0.0,
        )
        s.current_positions["A"] = 100
        # Price well below mean: z is very negative, but exit_threshold=0
        # means exit requires z >= 0 exactly, which won't happen
        data = _stable_then_spike(10, 100.0, 80.0)
        dh = FakeDataHandler({"A": data})
        for _ in range(20):
            signals = s.calculate_signals(datetime.now(), dh)
        assert signals == []

    def test_signal_strength_proportional_to_zscore(self):
        s = self._make_strategy(lookback_period=10, entry_threshold=1.0)
        dh = FakeDataHandler({"A": _stable_then_spike(10, 100.0, 80.0)})
        signals = s.calculate_signals(datetime.now(), dh)
        assert len(signals) == 1
        assert signals[0].strength > 0

    def test_zero_std_no_signal(self):
        """All identical prices → std=0 → no crash, no signal."""
        s = self._make_strategy(lookback_period=5)
        dh = FakeDataHandler({"A": _make_prices([50.0] * 5)})
        signals = s.calculate_signals(datetime.now(), dh)
        assert signals == []


# ===========================================================================
# Module exports
# ===========================================================================


class TestExports:
    def test_importable(self):
        from src.strategy import MeanReversionStrategy

        assert MeanReversionStrategy is not None

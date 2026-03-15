"""Tests for MomentumStrategy."""

from datetime import datetime

import pandas as pd
import pytest

from src.events.event import SignalType
from src.events.queue import EventQueue
from src.strategy.momentum import MomentumStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeDataHandler:
    """Minimal DataHandler stub for strategy tests."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_latest_bars(self, symbol: str, n: int) -> pd.DataFrame:
        df = self._data.get(symbol, pd.DataFrame())
        return df.tail(n)


def _make_prices(start: float, n: int, trend: float = 0.0) -> pd.DataFrame:
    """Generate a simple price DataFrame with linear trend."""
    closes = [start + i * trend for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000] * n,
        }
    )


# ===========================================================================
# Init
# ===========================================================================


class TestInit:
    def test_defaults(self):
        s = MomentumStrategy(symbols=["A", "B", "C", "D"])
        assert s.lookback_period == 20
        assert s.top_n == 3
        assert s.bottom_n == 0
        assert s.rebalance_frequency == 5
        assert s.min_bars == 30

    def test_custom_params(self):
        s = MomentumStrategy(
            symbols=["A", "B", "C"],
            parameters={"lookback_period": 10, "top_n": 1, "bottom_n": 1},
        )
        assert s.lookback_period == 10
        assert s.top_n == 1
        assert s.bottom_n == 1

    def test_lookback_too_small(self):
        with pytest.raises(ValueError, match="lookback_period"):
            MomentumStrategy(symbols=["A"], parameters={"lookback_period": 1})

    def test_top_bottom_exceeds_symbols(self):
        with pytest.raises(ValueError, match="exceeds"):
            MomentumStrategy(
                symbols=["A", "B"],
                parameters={"top_n": 2, "bottom_n": 1},
            )


# ===========================================================================
# Signal generation
# ===========================================================================


class TestSignals:
    def _make_strategy(self, symbols, **kwargs):
        params = {"lookback_period": 5, "min_bars": 1, "rebalance_frequency": 1}
        params.update(kwargs)
        s = MomentumStrategy(symbols=symbols, parameters=params)
        eq = EventQueue()
        s.set_event_queue(eq)
        return s

    def test_no_signals_before_min_bars(self):
        s = MomentumStrategy(
            symbols=["A", "B", "C", "D"],
            parameters={"min_bars": 10, "rebalance_frequency": 1},
        )
        dh = FakeDataHandler({"A": _make_prices(100, 50)})
        signals = s.calculate_signals(datetime.now(), dh)
        assert signals == []

    def test_no_signals_between_rebalances(self):
        s = MomentumStrategy(
            symbols=["A", "B", "C", "D"],
            parameters={"min_bars": 1, "rebalance_frequency": 5},
        )
        dh = FakeDataHandler({sym: _make_prices(100, 50) for sym in "ABCD"})
        # bar 1 — not a rebalance bar (1 % 5 != 0)
        signals = s.calculate_signals(datetime.now(), dh)
        assert signals == []

    def test_long_signal_for_top_performers(self):
        s = self._make_strategy(["A", "B", "C", "D"], top_n=2, bottom_n=0)
        dh = FakeDataHandler(
            {
                "A": _make_prices(100, 10, trend=2.0),  # best
                "B": _make_prices(100, 10, trend=1.0),
                "C": _make_prices(100, 10, trend=-1.0),
                "D": _make_prices(100, 10, trend=-2.0),  # worst
            }
        )
        signals = s.calculate_signals(datetime.now(), dh)
        long_syms = {
            sig.symbol for sig in signals if sig.signal_type == SignalType.LONG
        }
        assert "A" in long_syms
        assert "B" in long_syms

    def test_short_signal_for_bottom_performers(self):
        s = self._make_strategy(["A", "B", "C", "D"], top_n=1, bottom_n=1)
        dh = FakeDataHandler(
            {
                "A": _make_prices(100, 10, trend=2.0),
                "B": _make_prices(100, 10, trend=1.0),
                "C": _make_prices(100, 10, trend=-1.0),
                "D": _make_prices(100, 10, trend=-2.0),
            }
        )
        signals = s.calculate_signals(datetime.now(), dh)
        short_syms = {
            sig.symbol for sig in signals if sig.signal_type == SignalType.SHORT
        }
        assert "D" in short_syms

    def test_exit_signal_for_middle_positions(self):
        s = self._make_strategy(["A", "B", "C", "D"], top_n=1, bottom_n=0)
        # Simulate having a position in B
        s.current_positions["B"] = 100
        dh = FakeDataHandler(
            {
                "A": _make_prices(100, 10, trend=2.0),
                "B": _make_prices(100, 10, trend=1.0),
                "C": _make_prices(100, 10, trend=-1.0),
                "D": _make_prices(100, 10, trend=-2.0),
            }
        )
        signals = s.calculate_signals(datetime.now(), dh)
        exit_syms = {
            sig.symbol for sig in signals if sig.signal_type == SignalType.EXIT
        }
        assert "B" in exit_syms

    def test_no_duplicate_long_when_already_long(self):
        s = self._make_strategy(["A", "B", "C", "D"], top_n=2, bottom_n=0)
        s.current_positions["A"] = 100
        dh = FakeDataHandler(
            {
                "A": _make_prices(100, 10, trend=2.0),
                "B": _make_prices(100, 10, trend=1.0),
                "C": _make_prices(100, 10, trend=-1.0),
                "D": _make_prices(100, 10, trend=-2.0),
            }
        )
        signals = s.calculate_signals(datetime.now(), dh)
        long_a = [
            s for s in signals if s.symbol == "A" and s.signal_type == SignalType.LONG
        ]
        assert long_a == []

    def test_insufficient_data_skips_symbol(self):
        s = self._make_strategy(["A", "B", "C", "D"], top_n=1, lookback_period=50)
        dh = FakeDataHandler(
            {
                "A": _make_prices(100, 10),  # only 10 bars, need 50
                "B": _make_prices(100, 10),
                "C": _make_prices(100, 10),
                "D": _make_prices(100, 10),
            }
        )
        signals = s.calculate_signals(datetime.now(), dh)
        assert signals == []

    def test_long_only_mode(self):
        """bottom_n=0 means no SHORT signals."""
        s = self._make_strategy(["A", "B", "C", "D"], top_n=2, bottom_n=0)
        dh = FakeDataHandler(
            {
                "A": _make_prices(100, 10, trend=2.0),
                "B": _make_prices(100, 10, trend=1.0),
                "C": _make_prices(100, 10, trend=-1.0),
                "D": _make_prices(100, 10, trend=-2.0),
            }
        )
        signals = s.calculate_signals(datetime.now(), dh)
        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        assert short_signals == []


# ===========================================================================
# Module exports
# ===========================================================================


class TestExports:
    def test_importable(self):
        from src.strategy import MomentumStrategy

        assert MomentumStrategy is not None

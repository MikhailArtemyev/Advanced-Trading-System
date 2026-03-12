"""Tests for LiveDataHandler — DataHandler interface over live data."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.live.bar_aggregator import BarAggregator
from src.live.data_feed import Tick
from src.live.data_handler import LiveDataHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tick(symbol, ts, price, volume=100.0):
    return Tick(symbol=symbol, timestamp=ts, price=price, volume=volume)


def _make_handler(symbols=None, interval_seconds=60):
    symbols = symbols or ["AAPL"]
    agg = BarAggregator(interval=timedelta(seconds=interval_seconds))
    handler = LiveDataHandler(symbols=symbols, bar_aggregator=agg)
    return handler, agg


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestLiveDataHandlerInit:
    def test_initial_state(self):
        handler, _ = _make_handler()
        assert handler.bar_count == 0
        assert handler.continue_backtest is True
        assert handler.get_all_symbols() == ["AAPL"]

    def test_multiple_symbols(self):
        handler, _ = _make_handler(symbols=["AAPL", "MSFT", "GOOGL"])
        assert handler.get_all_symbols() == ["AAPL", "MSFT", "GOOGL"]

    def test_get_all_symbols_is_copy(self):
        handler, _ = _make_handler()
        syms = handler.get_all_symbols()
        syms.append("MSFT")
        assert handler.get_all_symbols() == ["AAPL"]

    def test_registers_on_bar_callback(self):
        handler, agg = _make_handler()
        assert agg.on_bar is not None

    def test_custom_max_history(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        handler = LiveDataHandler(symbols=["AAPL"], bar_aggregator=agg, max_history=100)
        assert handler._max_history == 100


# ---------------------------------------------------------------------------
# get_latest_bars
# ---------------------------------------------------------------------------


class TestGetLatestBars:
    def test_empty_when_no_bars(self):
        handler, _ = _make_handler()
        df = handler.get_latest_bars("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_returns_completed_bars(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        # Feed ticks to fill one bar, then cross into next bar to complete it
        agg.on_tick(_make_tick("AAPL", base + timedelta(seconds=10), 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(seconds=30), 151.0))
        # Cross into next minute to emit bar
        agg.on_tick(_make_tick("AAPL", base + timedelta(seconds=70), 152.0))

        df = handler.get_latest_bars("AAPL", n=1)
        assert len(df) == 1
        assert df["open"].iloc[0] == 150.0
        assert df["close"].iloc[0] == 151.0
        assert df["high"].iloc[0] == 151.0
        assert df["low"].iloc[0] == 150.0

    def test_returns_n_bars(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        # Create 3 completed bars
        for minute in range(4):
            ts = base + timedelta(minutes=minute, seconds=10)
            agg.on_tick(_make_tick("AAPL", ts, 150.0 + minute))

        # 3 bar crossings = 3 completed bars
        df = handler.get_latest_bars("AAPL", n=2)
        assert len(df) == 2

    def test_returns_all_bars_when_n_exceeds_available(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        df = handler.get_latest_bars("AAPL", n=100)
        assert len(df) == 1

    def test_correct_columns(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        df = handler.get_latest_bars("AAPL")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "datetime"

    def test_unknown_symbol_returns_empty(self):
        handler, _ = _make_handler()
        df = handler.get_latest_bars("UNKNOWN")
        assert df.empty


# ---------------------------------------------------------------------------
# update_bars
# ---------------------------------------------------------------------------


class TestUpdateBars:
    def test_false_initially(self):
        handler, _ = _make_handler()
        assert handler.update_bars() is False

    def test_true_after_new_bar(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        assert handler.update_bars() is True

    def test_false_on_second_call(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        handler.update_bars()
        assert handler.update_bars() is False

    def test_true_again_after_another_bar(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))
        handler.update_bars()

        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=2), 152.0))
        assert handler.update_bars() is True

    def test_false_on_partial_bar_only(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        # Only one tick — no completed bar yet
        agg.on_tick(_make_tick("AAPL", base, 150.0))
        assert handler.update_bars() is False


# ---------------------------------------------------------------------------
# get_current_timestamp
# ---------------------------------------------------------------------------


class TestGetCurrentTimestamp:
    def test_returns_now_initially(self):
        handler, _ = _make_handler()
        ts = handler.get_current_timestamp()
        assert isinstance(ts, datetime)
        # Should be approximately now
        diff = abs((datetime.now() - ts).total_seconds())
        assert diff < 2.0

    def test_returns_bar_timestamp_after_bar(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        ts = handler.get_current_timestamp()
        assert ts == base  # First bar's timestamp

    def test_updates_with_each_bar(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=2), 152.0))

        ts = handler.get_current_timestamp()
        assert ts == base + timedelta(minutes=1)


# ---------------------------------------------------------------------------
# continue_backtest
# ---------------------------------------------------------------------------


class TestContinueBacktest:
    def test_always_true(self):
        handler, _ = _make_handler()
        assert handler.continue_backtest is True

    def test_true_after_bars(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        assert handler.continue_backtest is True

    def test_true_after_reset(self):
        handler, _ = _make_handler()
        handler.reset()
        assert handler.continue_backtest is True


# ---------------------------------------------------------------------------
# get_latest_bar_value
# ---------------------------------------------------------------------------


class TestGetLatestBarValue:
    def test_returns_close(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(seconds=30), 155.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 160.0))

        assert handler.get_latest_bar_value("AAPL", "close") == 155.0

    def test_returns_volume(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0, volume=200.0))
        agg.on_tick(
            _make_tick("AAPL", base + timedelta(seconds=30), 155.0, volume=300.0)
        )
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 160.0))

        assert handler.get_latest_bar_value("AAPL", "volume") == 500.0

    def test_raises_on_no_bars(self):
        handler, _ = _make_handler()
        with pytest.raises(ValueError, match="No bars available"):
            handler.get_latest_bar_value("AAPL", "close")

    def test_returns_float(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        result = handler.get_latest_bar_value("AAPL", "close")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_state(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        handler.reset()

        assert handler.bar_count == 0
        assert handler.update_bars() is False
        assert handler.get_latest_bars("AAPL").empty

    def test_resets_timestamp(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        handler.reset()

        ts = handler.get_current_timestamp()
        diff = abs((datetime.now() - ts).total_seconds())
        assert diff < 2.0

    def test_can_receive_bars_after_reset(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        handler.reset()

        base2 = datetime(2025, 1, 15, 11, 0, 0)
        agg.on_tick(_make_tick("AAPL", base2, 160.0))
        agg.on_tick(_make_tick("AAPL", base2 + timedelta(minutes=1), 161.0))

        assert handler.bar_count == 1
        assert handler.update_bars() is True


# ---------------------------------------------------------------------------
# bar_count
# ---------------------------------------------------------------------------


class TestBarCount:
    def test_zero_initially(self):
        handler, _ = _make_handler()
        assert handler.bar_count == 0

    def test_increments_on_each_bar(self):
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        for minute in range(5):
            agg.on_tick(
                _make_tick("AAPL", base + timedelta(minutes=minute), 150.0 + minute)
            )

        # 4 bar crossings = 4 completed bars
        assert handler.bar_count == 4

    def test_counts_across_symbols(self):
        handler, agg = _make_handler(symbols=["AAPL", "MSFT"], interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("MSFT", base, 200.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))
        agg.on_tick(_make_tick("MSFT", base + timedelta(minutes=1), 201.0))

        assert handler.bar_count == 2


# ---------------------------------------------------------------------------
# Multi-symbol
# ---------------------------------------------------------------------------


class TestMultiSymbol:
    def test_independent_bars(self):
        handler, agg = _make_handler(symbols=["AAPL", "MSFT"], interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("MSFT", base, 200.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))
        agg.on_tick(_make_tick("MSFT", base + timedelta(minutes=1), 201.0))

        aapl_df = handler.get_latest_bars("AAPL")
        msft_df = handler.get_latest_bars("MSFT")

        assert len(aapl_df) == 1
        assert len(msft_df) == 1
        assert aapl_df["close"].iloc[0] == 150.0
        assert msft_df["close"].iloc[0] == 200.0

    def test_one_symbol_has_bars_other_empty(self):
        handler, agg = _make_handler(symbols=["AAPL", "MSFT"], interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        agg.on_tick(_make_tick("AAPL", base + timedelta(minutes=1), 151.0))

        assert len(handler.get_latest_bars("AAPL")) == 1
        assert handler.get_latest_bars("MSFT").empty


# ---------------------------------------------------------------------------
# Integration: full tick → handler pipeline
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_tick_to_handler_pipeline(self):
        """Simulate a stream of ticks and verify the handler provides
        correct bars through the DataHandler interface."""
        handler, agg = _make_handler(symbols=["AAPL"], interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        # Simulate 5 minutes of ticks (3 ticks per minute)
        prices = [
            # Minute 0
            (0, 10, 150.0),
            (0, 20, 151.0),
            (0, 45, 149.5),
            # Minute 1
            (1, 5, 152.0),
            (1, 30, 153.0),
            (1, 55, 151.5),
            # Minute 2
            (2, 10, 154.0),
            (2, 25, 155.0),
            (2, 50, 154.5),
            # Minute 3
            (3, 15, 156.0),
            (3, 35, 157.0),
            (3, 50, 156.5),
            # Minute 4 (partial — won't complete until minute 5)
            (4, 10, 158.0),
        ]

        for minute, second, price in prices:
            ts = base + timedelta(minutes=minute, seconds=second)
            agg.on_tick(_make_tick("AAPL", ts, price, volume=100.0))

        # 4 completed bars (minutes 0-3), minute 4 is partial
        df = handler.get_latest_bars("AAPL", n=10)
        assert len(df) == 4
        assert handler.bar_count == 4

        # Verify first bar
        assert df["open"].iloc[0] == 150.0
        assert df["high"].iloc[0] == 151.0
        assert df["low"].iloc[0] == 149.5
        assert df["close"].iloc[0] == 149.5
        assert df["volume"].iloc[0] == 300.0

        # Verify last completed bar
        assert df["open"].iloc[3] == 156.0
        assert df["close"].iloc[3] == 156.5

        # update_bars should be True (bars arrived)
        assert handler.update_bars() is True
        assert handler.update_bars() is False

        # continue_backtest always True
        assert handler.continue_backtest is True

        # Timestamp is from the last completed bar
        assert handler.get_current_timestamp() == base + timedelta(minutes=3)

    def test_handler_is_datahandler_subclass(self):
        from src.data.data_handler import DataHandler

        handler, _ = _make_handler()
        assert isinstance(handler, DataHandler)

    def test_flush_triggers_update_bars(self):
        """Flushing the aggregator should emit a bar the handler sees."""
        handler, agg = _make_handler(interval_seconds=60)
        base = datetime(2025, 1, 15, 10, 0, 0)

        agg.on_tick(_make_tick("AAPL", base, 150.0))
        assert handler.update_bars() is False  # No completed bar yet

        agg.flush("AAPL")
        assert handler.update_bars() is True
        assert handler.bar_count == 1

        df = handler.get_latest_bars("AAPL")
        assert len(df) == 1
        assert df["close"].iloc[0] == 150.0

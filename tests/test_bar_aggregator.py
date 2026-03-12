"""Tests for tick-to-bar aggregation."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.live.bar_aggregator import Bar, BarAggregator
from src.live.data_feed import Tick


def _tick(
    symbol: str = "AAPL",
    ts: datetime | None = None,
    price: float = 150.0,
    volume: float = 100.0,
) -> Tick:
    """Helper to create ticks with defaults."""
    return Tick(
        symbol=symbol,
        timestamp=ts or datetime(2025, 1, 15, 10, 0, 0),
        price=price,
        volume=volume,
    )


# ---------------------------------------------------------------------------
# Bar dataclass tests
# ---------------------------------------------------------------------------


class TestBar:
    def test_bar_fields(self):
        bar = Bar(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 15, 10, 0),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000.0,
            tick_count=10,
        )
        assert bar.symbol == "AAPL"
        assert bar.open == 150.0
        assert bar.high == 152.0
        assert bar.low == 149.0
        assert bar.close == 151.0
        assert bar.volume == 1000.0
        assert bar.tick_count == 10

    def test_bar_is_mutable(self):
        bar = Bar(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 15),
            open=150.0,
            high=150.0,
            low=150.0,
            close=150.0,
            volume=0.0,
            tick_count=0,
        )
        bar.high = 155.0
        assert bar.high == 155.0


# ---------------------------------------------------------------------------
# BarAggregator construction tests
# ---------------------------------------------------------------------------


class TestBarAggregatorInit:
    def test_interval_stored(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        assert agg.interval == timedelta(minutes=1)

    def test_zero_interval_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BarAggregator(interval=timedelta(0))

    def test_negative_interval_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BarAggregator(interval=timedelta(seconds=-1))

    def test_callback_stored(self):
        cb = lambda bar: None  # noqa: E731
        agg = BarAggregator(interval=timedelta(minutes=1), on_bar=cb)
        assert agg.on_bar is cb

    def test_no_callback(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        assert agg.on_bar is None

    def test_set_callback(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        cb = lambda bar: None  # noqa: E731
        agg.on_bar = cb
        assert agg.on_bar is cb


# ---------------------------------------------------------------------------
# Tick processing tests
# ---------------------------------------------------------------------------


class TestBarAggregatorOnTick:
    def test_first_tick_creates_bar(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        tick = _tick(ts=datetime(2025, 1, 15, 10, 0, 15), price=150.0)
        agg.on_tick(tick)

        bar = agg.get_current_bar("AAPL")
        assert bar is not None
        assert bar.open == 150.0
        assert bar.high == 150.0
        assert bar.low == 150.0
        assert bar.close == 150.0
        assert bar.volume == 100.0
        assert bar.tick_count == 1
        # Floored to 10:00:00
        assert bar.timestamp == datetime(2025, 1, 15, 10, 0, 0)

    def test_second_tick_same_bar_updates_high(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 20), price=152.0))

        bar = agg.get_current_bar("AAPL")
        assert bar is not None
        assert bar.high == 152.0
        assert bar.close == 152.0
        assert bar.tick_count == 2

    def test_second_tick_same_bar_updates_low(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 20), price=148.0))

        bar = agg.get_current_bar("AAPL")
        assert bar is not None
        assert bar.low == 148.0

    def test_volume_accumulates(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        agg.on_tick(
            _tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0, volume=50.0)
        )
        agg.on_tick(
            _tick(ts=datetime(2025, 1, 15, 10, 0, 20), price=151.0, volume=75.0)
        )

        bar = agg.get_current_bar("AAPL")
        assert bar is not None
        assert bar.volume == 125.0

    def test_open_stays_at_first_tick(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 20), price=155.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 30), price=148.0))

        bar = agg.get_current_bar("AAPL")
        assert bar is not None
        assert bar.open == 150.0

    def test_close_is_last_tick(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 20), price=155.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 30), price=152.0))

        bar = agg.get_current_bar("AAPL")
        assert bar is not None
        assert bar.close == 152.0


# ---------------------------------------------------------------------------
# Bar completion tests
# ---------------------------------------------------------------------------


class TestBarCompletion:
    def test_new_interval_emits_bar(self):
        completed: list[Bar] = []
        agg = BarAggregator(
            interval=timedelta(minutes=1),
            on_bar=completed.append,
        )

        # Bar 1: 10:00
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 30), price=151.0))

        # Bar 2: 10:01 — triggers completion of bar 1
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 1, 5), price=152.0))

        assert len(completed) == 1
        assert completed[0].open == 150.0
        assert completed[0].close == 151.0
        assert completed[0].tick_count == 2

    def test_completed_bar_in_history(self):
        agg = BarAggregator(interval=timedelta(minutes=1))

        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 1, 5), price=152.0))

        bars = agg.get_completed_bars("AAPL")
        assert len(bars) == 1
        assert bars[0].open == 150.0

    def test_multiple_bars_completed(self):
        agg = BarAggregator(interval=timedelta(minutes=1))

        # 3 bars: 10:00, 10:01, 10:02 (current)
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 1, 10), price=151.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 2, 10), price=152.0))

        bars = agg.get_completed_bars("AAPL")
        assert len(bars) == 2
        assert bars[0].close == 150.0
        assert bars[1].close == 151.0

    def test_get_completed_bars_with_limit(self):
        agg = BarAggregator(interval=timedelta(minutes=1))

        for minute in range(5):
            agg.on_tick(
                _tick(
                    ts=datetime(2025, 1, 15, 10, minute, 10),
                    price=150.0 + minute,
                )
            )

        bars = agg.get_completed_bars("AAPL", n=2)
        assert len(bars) == 2
        # Last two completed bars (minutes 2 and 3, since minute 4 is current)
        assert bars[0].close == 152.0
        assert bars[1].close == 153.0

    def test_get_completed_bars_unknown_symbol(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        bars = agg.get_completed_bars("UNKNOWN")
        assert bars == []

    def test_get_current_bar_unknown_symbol(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        assert agg.get_current_bar("UNKNOWN") is None

    def test_callback_not_called_on_first_tick(self):
        completed: list[Bar] = []
        agg = BarAggregator(
            interval=timedelta(minutes=1),
            on_bar=completed.append,
        )
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        assert len(completed) == 0


# ---------------------------------------------------------------------------
# Multi-symbol tests
# ---------------------------------------------------------------------------


class TestMultiSymbol:
    def test_independent_tracking(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        ts = datetime(2025, 1, 15, 10, 0, 10)

        agg.on_tick(_tick(symbol="AAPL", ts=ts, price=150.0))
        agg.on_tick(_tick(symbol="MSFT", ts=ts, price=300.0))

        aapl = agg.get_current_bar("AAPL")
        msft = agg.get_current_bar("MSFT")
        assert aapl is not None
        assert msft is not None
        assert aapl.price_open if hasattr(aapl, "price_open") else aapl.open == 150.0
        assert msft.open == 300.0

    def test_one_symbol_completes_other_stays(self):
        completed: list[Bar] = []
        agg = BarAggregator(
            interval=timedelta(minutes=1),
            on_bar=completed.append,
        )

        # Both at 10:00
        agg.on_tick(
            _tick(symbol="AAPL", ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0)
        )
        agg.on_tick(
            _tick(symbol="MSFT", ts=datetime(2025, 1, 15, 10, 0, 10), price=300.0)
        )

        # Only AAPL advances to 10:01
        agg.on_tick(
            _tick(symbol="AAPL", ts=datetime(2025, 1, 15, 10, 1, 10), price=151.0)
        )

        assert len(completed) == 1
        assert completed[0].symbol == "AAPL"

        # MSFT still in current bar
        msft_current = agg.get_current_bar("MSFT")
        assert msft_current is not None
        assert msft_current.timestamp == datetime(2025, 1, 15, 10, 0, 0)


# ---------------------------------------------------------------------------
# Interval tests (different durations)
# ---------------------------------------------------------------------------


class TestIntervals:
    def test_5_minute_bars(self):
        agg = BarAggregator(interval=timedelta(minutes=5))

        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 2, 30), price=150.0))
        bar = agg.get_current_bar("AAPL")
        assert bar is not None
        assert bar.timestamp == datetime(2025, 1, 15, 10, 0, 0)

    def test_1_hour_bars(self):
        agg = BarAggregator(interval=timedelta(hours=1))

        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 45, 0), price=150.0))
        bar = agg.get_current_bar("AAPL")
        assert bar is not None
        assert bar.timestamp == datetime(2025, 1, 15, 10, 0, 0)

    def test_1_second_bars(self):
        completed: list[Bar] = []
        agg = BarAggregator(
            interval=timedelta(seconds=1),
            on_bar=completed.append,
        )

        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 0, 100000), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 0, 500000), price=151.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 1, 100000), price=152.0))

        assert len(completed) == 1
        assert completed[0].open == 150.0
        assert completed[0].close == 151.0

    def test_floor_timestamp_1m(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        ts = datetime(2025, 1, 15, 10, 3, 45)
        floored = agg._floor_timestamp(ts)
        assert floored == datetime(2025, 1, 15, 10, 3, 0)

    def test_floor_timestamp_5m(self):
        agg = BarAggregator(interval=timedelta(minutes=5))
        ts = datetime(2025, 1, 15, 10, 7, 30)
        floored = agg._floor_timestamp(ts)
        assert floored == datetime(2025, 1, 15, 10, 5, 0)

    def test_floor_timestamp_1h(self):
        agg = BarAggregator(interval=timedelta(hours=1))
        ts = datetime(2025, 1, 15, 10, 45, 0)
        floored = agg._floor_timestamp(ts)
        assert floored == datetime(2025, 1, 15, 10, 0, 0)


# ---------------------------------------------------------------------------
# DataFrame output tests
# ---------------------------------------------------------------------------


class TestDataFrame:
    def test_empty_dataframe(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        df = agg.get_bars_as_dataframe("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_dataframe_columns(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 1, 10), price=151.0))

        df = agg.get_bars_as_dataframe("AAPL")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "datetime"

    def test_dataframe_values(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        agg.on_tick(
            _tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0, volume=50.0)
        )
        agg.on_tick(
            _tick(ts=datetime(2025, 1, 15, 10, 0, 30), price=152.0, volume=75.0)
        )
        agg.on_tick(
            _tick(ts=datetime(2025, 1, 15, 10, 1, 10), price=151.0, volume=60.0)
        )

        df = agg.get_bars_as_dataframe("AAPL")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["open"] == 150.0
        assert row["high"] == 152.0
        assert row["low"] == 150.0
        assert row["close"] == 152.0
        assert row["volume"] == 125.0

    def test_dataframe_with_n_limit(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        for minute in range(5):
            agg.on_tick(
                _tick(
                    ts=datetime(2025, 1, 15, 10, minute, 10),
                    price=150.0 + minute,
                )
            )

        df = agg.get_bars_as_dataframe("AAPL", n=2)
        assert len(df) == 2

    def test_dataframe_unknown_symbol(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        df = agg.get_bars_as_dataframe("UNKNOWN")
        assert df.empty


# ---------------------------------------------------------------------------
# Flush tests
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_all(self):
        completed: list[Bar] = []
        agg = BarAggregator(
            interval=timedelta(minutes=1),
            on_bar=completed.append,
        )
        agg.on_tick(
            _tick(symbol="AAPL", ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0)
        )
        agg.on_tick(
            _tick(symbol="MSFT", ts=datetime(2025, 1, 15, 10, 0, 10), price=300.0)
        )

        agg.flush()

        assert len(completed) == 2
        assert agg.get_current_bar("AAPL") is None
        assert agg.get_current_bar("MSFT") is None

    def test_flush_single_symbol(self):
        completed: list[Bar] = []
        agg = BarAggregator(
            interval=timedelta(minutes=1),
            on_bar=completed.append,
        )
        agg.on_tick(
            _tick(symbol="AAPL", ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0)
        )
        agg.on_tick(
            _tick(symbol="MSFT", ts=datetime(2025, 1, 15, 10, 0, 10), price=300.0)
        )

        agg.flush("AAPL")

        assert len(completed) == 1
        assert completed[0].symbol == "AAPL"
        assert agg.get_current_bar("AAPL") is None
        assert agg.get_current_bar("MSFT") is not None

    def test_flush_moves_to_completed(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))

        agg.flush()

        bars = agg.get_completed_bars("AAPL")
        assert len(bars) == 1
        assert bars[0].open == 150.0

    def test_flush_nonexistent_symbol(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        # Should not raise
        agg.flush("UNKNOWN")


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_everything(self):
        agg = BarAggregator(interval=timedelta(minutes=1))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 1, 10), price=151.0))

        agg.reset()

        assert agg.get_current_bar("AAPL") is None
        assert agg.get_completed_bars("AAPL") == []
        assert agg.get_gaps() == []


# ---------------------------------------------------------------------------
# Gap detection tests
# ---------------------------------------------------------------------------


class TestGapDetection:
    def test_no_gap_on_consecutive_bars(self):
        agg = BarAggregator(interval=timedelta(minutes=1))

        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 1, 10), price=151.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 2, 10), price=152.0))

        assert len(agg.get_gaps()) == 0

    def test_gap_detected_when_bars_skip(self):
        agg = BarAggregator(interval=timedelta(minutes=1))

        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        # Jump from 10:00 to 10:05 — 4 missing bars
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 5, 10), price=151.0))

        gaps = agg.get_gaps()
        assert len(gaps) == 1
        assert gaps[0]["symbol"] == "AAPL"
        assert gaps[0]["duration_seconds"] == 240.0  # 4 minutes

    def test_no_gap_when_one_bar_skipped(self):
        """Skipping exactly one bar (2x interval) is the threshold boundary."""
        agg = BarAggregator(interval=timedelta(minutes=1))

        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        # Jump from 10:00 to 10:02 — only 1 missing bar, exactly at threshold
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 2, 10), price=151.0))

        # 10:02 bar_start, expected 10:01. 10:02 > 10:01 + 1min → no gap
        assert len(agg.get_gaps()) == 0

    def test_gap_records_correct_fields(self):
        agg = BarAggregator(interval=timedelta(minutes=5))

        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        # Jump from 10:00 to 10:20 — missing 10:05, 10:10, 10:15
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 20, 10), price=151.0))

        gaps = agg.get_gaps()
        assert len(gaps) == 1
        gap = gaps[0]
        assert "symbol" in gap
        assert "expected" in gap
        assert "actual" in gap
        assert "duration_seconds" in gap

    def test_gaps_per_symbol_independent(self):
        agg = BarAggregator(interval=timedelta(minutes=1))

        # AAPL has a gap
        agg.on_tick(
            _tick(symbol="AAPL", ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0)
        )
        agg.on_tick(
            _tick(symbol="AAPL", ts=datetime(2025, 1, 15, 10, 5, 10), price=151.0)
        )

        # MSFT has no gap
        agg.on_tick(
            _tick(symbol="MSFT", ts=datetime(2025, 1, 15, 10, 0, 10), price=300.0)
        )
        agg.on_tick(
            _tick(symbol="MSFT", ts=datetime(2025, 1, 15, 10, 1, 10), price=301.0)
        )

        gaps = agg.get_gaps()
        assert len(gaps) == 1
        assert gaps[0]["symbol"] == "AAPL"

    def test_reset_clears_gaps(self):
        agg = BarAggregator(interval=timedelta(minutes=1))

        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 0, 10), price=150.0))
        agg.on_tick(_tick(ts=datetime(2025, 1, 15, 10, 5, 10), price=151.0))
        assert len(agg.get_gaps()) == 1

        agg.reset()
        assert len(agg.get_gaps()) == 0


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_session_simulation(self):
        """Simulate 5 minutes of 1-second ticks."""
        completed: list[Bar] = []
        agg = BarAggregator(
            interval=timedelta(minutes=1),
            on_bar=completed.append,
        )

        base = datetime(2025, 1, 15, 10, 0, 0)
        price = 150.0

        for sec in range(300):  # 5 minutes = 300 seconds
            ts = base + timedelta(seconds=sec)
            price += (sec % 3 - 1) * 0.01  # oscillate price
            agg.on_tick(_tick(ts=ts, price=price, volume=10.0))

        # Should have 4 completed bars (0, 1, 2, 3) + 1 current (4)
        assert len(completed) == 4
        assert agg.get_current_bar("AAPL") is not None

        # Each bar should have 60 ticks
        for bar in completed:
            assert bar.tick_count == 60
            assert bar.volume == 600.0

    def test_tick_to_dataframe_pipeline(self):
        """End-to-end: ticks → aggregator → DataFrame."""
        agg = BarAggregator(interval=timedelta(minutes=1))

        prices = [150.0, 152.0, 149.0, 151.0, 153.0, 148.0, 150.0]
        base = datetime(2025, 1, 15, 10, 0, 0)

        for i, price in enumerate(prices):
            # Spread across 3+ minutes
            ts = base + timedelta(seconds=i * 30)
            agg.on_tick(_tick(ts=ts, price=price, volume=100.0))

        df = agg.get_bars_as_dataframe("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert all(
            col in df.columns for col in ["open", "high", "low", "close", "volume"]
        )

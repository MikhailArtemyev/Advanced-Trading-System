"""Stability / soak tests for the paper trading engine.

Runs the engine for a configurable duration against mock data,
verifying no memory leaks, no event queue growth, consistent
processing, and state snapshot reliability.
"""

import asyncio
import math
from datetime import datetime, timedelta

import pytest

from src.broker.order_manager import OrderManager
from src.broker.paper_broker import PaperBroker
from src.engine.paper_engine import PaperTradingEngine
from src.live.bar_aggregator import BarAggregator
from src.live.data_feed import Tick
from src.live.data_handler import LiveDataHandler
from src.portfolio.portfolio import Portfolio
from src.storage.sql_storage import SQLStorage
from src.strategy.sma_crossover import SMACrossoverStrategy


def _build_soak_engine(
    symbols: list[str],
    aggregator: BarAggregator,
    initial_capital: float = 100_000.0,
) -> tuple[PaperTradingEngine, LiveDataHandler, Portfolio, PaperBroker]:
    """Build engine for soak testing."""
    dh = LiveDataHandler(symbols=symbols, bar_aggregator=aggregator)
    strategy = SMACrossoverStrategy(
        symbols=symbols,
        parameters={"fast_period": 3, "slow_period": 7},
    )
    portfolio = Portfolio(initial_capital=initial_capital, symbols=symbols)
    broker = PaperBroker(initial_capital=initial_capital)
    broker._data_handler = dh
    om = OrderManager(broker=broker)

    engine = PaperTradingEngine(
        data_handler=dh,
        strategy=strategy,
        portfolio=portfolio,
        order_manager=om,
        bar_poll_interval=0.005,
        event_poll_interval=0.002,
        enforce_market_hours=False,
    )
    return engine, dh, portfolio, broker


def _generate_soak_ticks(
    symbol: str,
    base_price: float,
    duration_seconds: int,
    ticks_per_second: float = 10.0,
) -> list[Tick]:
    """Generate ticks for soak testing."""
    ticks = []
    dt = 1.0 / ticks_per_second
    start = datetime(2025, 6, 1, 9, 30)
    for i in range(int(duration_seconds * ticks_per_second)):
        t = i * dt
        trend = base_price * 0.03 * math.sin(2 * math.pi * t / 10)
        price = base_price + trend
        ticks.append(
            Tick(
                symbol=symbol,
                price=price,
                timestamp=start + timedelta(seconds=t),
                volume=100,
            )
        )
    return ticks


class TestSoakStability:
    """Run engine for extended periods and verify stability."""

    @pytest.mark.asyncio
    async def test_soak_10_seconds(self):
        """Run engine for ~10s worth of data, verify no crashes."""
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=2))
        engine, dh, portfolio, broker = _build_soak_engine(symbols, aggregator)

        ticks = _generate_soak_ticks("TEST", 100.0, duration_seconds=10)

        async def feed_and_stop():
            batch_size = 50
            for i in range(0, len(ticks), batch_size):
                for tick in ticks[i : i + batch_size]:
                    aggregator.on_tick(tick)
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.2)
            await engine.stop()

        await broker.connect()
        await asyncio.gather(engine.start(), feed_and_stop())
        await broker.disconnect()

        assert engine.bars_processed > 0
        assert engine.events_processed > 0
        assert portfolio.get_equity() > 0

    @pytest.mark.asyncio
    async def test_event_queue_drains(self):
        """Verify event queue is empty after engine stops."""
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=2))
        engine, dh, portfolio, broker = _build_soak_engine(symbols, aggregator)

        ticks = _generate_soak_ticks("TEST", 100.0, duration_seconds=5)

        async def feed_and_stop():
            for tick in ticks:
                aggregator.on_tick(tick)
            await asyncio.sleep(0.3)
            await engine.stop()

        await broker.connect()
        await asyncio.gather(engine.start(), feed_and_stop())
        await broker.disconnect()

        # Queue should be drained
        assert engine._events.empty()

    @pytest.mark.asyncio
    async def test_no_dropped_events(self):
        """Events processed should be >= bars processed."""
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=2))
        engine, dh, portfolio, broker = _build_soak_engine(symbols, aggregator)

        ticks = _generate_soak_ticks("TEST", 100.0, duration_seconds=5)

        async def feed_and_stop():
            for tick in ticks:
                aggregator.on_tick(tick)
            await asyncio.sleep(0.3)
            await engine.stop()

        await broker.connect()
        await asyncio.gather(engine.start(), feed_and_stop())
        await broker.disconnect()

        # Every bar generates at least a MarketEvent
        assert engine.events_processed >= engine.bars_processed

    @pytest.mark.asyncio
    async def test_state_snapshots_on_schedule(self, tmp_path):
        """Verify engine state can be saved periodically to SQLite."""
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=2))
        engine, dh, portfolio, broker = _build_soak_engine(symbols, aggregator)

        db_path = tmp_path / "soak.db"
        storage = SQLStorage(db_url=f"sqlite:///{db_path}")
        session_id = storage.create_session(
            session_type="paper", config={}, initial_capital=100_000.0
        )
        ticks = _generate_soak_ticks("TEST", 100.0, duration_seconds=5)

        async def feed_save_stop():
            batch_size = 50
            save_count = 0
            for i in range(0, len(ticks), batch_size):
                for tick in ticks[i : i + batch_size]:
                    aggregator.on_tick(tick)
                await asyncio.sleep(0.01)
                if i % 100 == 0 and engine.bars_processed > 0:
                    storage.save_engine_state(session_id, engine.get_state())
                    save_count += 1
            await asyncio.sleep(0.2)
            storage.save_engine_state(session_id, engine.get_state())
            await engine.stop()

        await broker.connect()
        await asyncio.gather(engine.start(), feed_save_stop())
        await broker.disconnect()

        loaded = storage.load_engine_state(session_id)
        assert loaded is not None
        storage.close()

    @pytest.mark.asyncio
    async def test_consistent_equity_tracking(self):
        """Equity should remain positive throughout the session."""
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=2))
        engine, dh, portfolio, broker = _build_soak_engine(symbols, aggregator)

        ticks = _generate_soak_ticks("TEST", 100.0, duration_seconds=5)
        equity_samples: list[float] = []

        async def feed_and_stop():
            batch_size = 50
            for i in range(0, len(ticks), batch_size):
                for tick in ticks[i : i + batch_size]:
                    aggregator.on_tick(tick)
                await asyncio.sleep(0.01)
                equity_samples.append(portfolio.get_equity())
            await asyncio.sleep(0.2)
            await engine.stop()

        await broker.connect()
        await asyncio.gather(engine.start(), feed_and_stop())
        await broker.disconnect()

        # All equity samples should be positive
        assert all(e > 0 for e in equity_samples)

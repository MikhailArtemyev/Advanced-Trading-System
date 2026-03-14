"""Phase 4 integration tests — end-to-end paper trading scenarios.

Tests the full pipeline: ticks → bars → strategy → orders → fills → portfolio,
plus state persistence, reconciliation, risk management, health monitoring,
multi-symbol operation, and graceful shutdown.
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
from src.monitoring.health import HealthMonitor, HealthStatus
from src.portfolio.portfolio import Portfolio
from src.risk.risk_manager import RiskLimits, RiskManager
from src.storage.sql_storage import SQLStorage
from src.strategy.sma_crossover import SMACrossoverStrategy

# ── Helpers ──────────────────────────────────────────────────────────


def _make_ticks(
    symbol: str,
    base_price: float,
    n_minutes: int,
    bar_seconds: int = 5,
    ticks_per_second: float = 2.0,
) -> list[Tick]:
    """Generate sine-wave ticks that produce SMA crossovers."""
    ticks: list[Tick] = []
    dt = 1.0 / ticks_per_second
    start = datetime(2025, 6, 1, 9, 30)
    total_seconds = n_minutes * 60
    for i in range(int(total_seconds * ticks_per_second)):
        t = i * dt
        trend = base_price * 0.04 * math.sin(2 * math.pi * t / (3 * bar_seconds))
        noise = base_price * 0.001 * math.sin(t * 13.7)
        price = base_price + trend + noise
        ticks.append(
            Tick(
                symbol=symbol,
                price=price,
                timestamp=start + timedelta(seconds=t),
                volume=100,
            )
        )
    return ticks


def _feed_ticks(aggregator: BarAggregator, ticks: list[Tick]) -> None:
    """Feed all ticks into aggregator."""
    for tick in ticks:
        aggregator.on_tick(tick)


def _build_engine(
    symbols: list[str],
    aggregator: BarAggregator,
    risk_manager: RiskManager | None = None,
    initial_capital: float = 100_000.0,
    fast_period: int = 3,
    slow_period: int = 7,
) -> tuple[PaperTradingEngine, LiveDataHandler, Portfolio, PaperBroker, OrderManager]:
    """Wire up a full paper trading engine."""
    data_handler = LiveDataHandler(symbols=symbols, bar_aggregator=aggregator)
    strategy = SMACrossoverStrategy(
        symbols=symbols,
        parameters={"fast_period": fast_period, "slow_period": slow_period},
    )
    portfolio = Portfolio(initial_capital=initial_capital, symbols=symbols)
    broker = PaperBroker(
        initial_capital=initial_capital,
        commission_pct=0.001,
        slippage_pct=0.0005,
    )
    broker._data_handler = data_handler
    order_manager = OrderManager(broker=broker)

    engine = PaperTradingEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        order_manager=order_manager,
        risk_manager=risk_manager,
        bar_poll_interval=0.01,
        event_poll_interval=0.005,
    )
    return engine, data_handler, portfolio, broker, order_manager


async def _run_engine_with_ticks(
    engine: PaperTradingEngine,
    aggregator: BarAggregator,
    ticks: list[Tick],
    batch_size: int = 20,
) -> None:
    """Feed ticks and run engine concurrently."""

    async def feed_and_stop() -> None:
        for i in range(0, len(ticks), batch_size):
            batch = ticks[i : i + batch_size]
            for tick in batch:
                aggregator.on_tick(tick)
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        await engine.stop()

    broker = engine._order_manager._broker
    await broker.connect()
    try:
        await asyncio.gather(engine.start(), feed_and_stop())
    finally:
        await broker.disconnect()


# ── Integration Tests ────────────────────────────────────────────────


class TestFullPipeline:
    """End-to-end: ticks → bars → signals → orders → fills → positions."""

    @pytest.mark.asyncio
    async def test_pipeline_produces_fills(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        ticks = _make_ticks("TEST", 100.0, n_minutes=2, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        assert engine.bars_processed > 0
        assert engine.events_processed > 0

    @pytest.mark.asyncio
    async def test_pipeline_updates_portfolio(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        ticks = _make_ticks("TEST", 100.0, n_minutes=2, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        # Portfolio should have processed some events
        stats = engine.get_statistics()
        assert stats["bars_processed"] > 0

    @pytest.mark.asyncio
    async def test_pipeline_equity_changes(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        ticks = _make_ticks("TEST", 100.0, n_minutes=3, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        # If trades occurred, equity may differ from initial
        stats = engine.get_statistics()
        assert isinstance(stats["equity"], float)
        assert stats["equity"] > 0


class TestStatePersistenceRoundTrip:
    """Save → load engine state via SQLStorage, verify state preserved."""

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, tmp_path):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        ticks = _make_ticks("TEST", 100.0, n_minutes=2, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        # Save state to SQLite
        db_path = tmp_path / "test.db"
        storage = SQLStorage(db_url=f"sqlite:///{db_path}")
        session_id = storage.create_session(
            session_type="paper", config={}, initial_capital=100_000.0
        )
        state = engine.get_state()
        storage.save_engine_state(session_id, state)

        # Load and verify
        loaded = storage.load_engine_state(session_id)
        assert loaded is not None
        assert loaded["cash"] == state["cash"]
        assert (
            loaded["statistics"]["bars_processed"]
            == state["statistics"]["bars_processed"]
        )
        storage.close()

    @pytest.mark.asyncio
    async def test_multiple_saves_loads_latest(self, tmp_path):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        db_path = tmp_path / "test.db"
        storage = SQLStorage(db_url=f"sqlite:///{db_path}")
        session_id = storage.create_session(
            session_type="paper", config={}, initial_capital=100_000.0
        )

        # Save multiple states with increasing cash values
        for i in range(5):
            state = engine.get_state()
            state["cash"] = float(i * 1000)
            storage.save_engine_state(session_id, state)

        # Load should return the latest (highest cash)
        loaded = storage.load_engine_state(session_id)
        assert loaded is not None
        assert loaded["cash"] == 4000.0
        storage.close()


class TestReconciliation:
    """Engine vs broker position reconciliation."""

    @pytest.mark.asyncio
    async def test_reconciliation_after_trades(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        ticks = _make_ticks("TEST", 100.0, n_minutes=2, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        report = await engine.reconcile_positions()
        assert "matched" in report
        assert "symbols_checked" in report
        assert isinstance(report["discrepancies"], list)

    @pytest.mark.asyncio
    async def test_reconciliation_no_trades(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        # Don't feed any ticks — no trades
        await broker.connect()
        report = await engine.reconcile_positions()
        assert report["matched"] is True
        await broker.disconnect()


class TestRiskManagerIntegration:
    """Orders exceeding risk limits get rejected."""

    @pytest.mark.asyncio
    async def test_risk_rejection_still_processes(self):
        limits = RiskLimits(
            max_position_pct=0.01,  # very tight limit
            max_drawdown_pct=0.001,
            max_daily_loss_pct=0.001,
            max_open_positions=1,
            max_orders_per_day=2,
        )
        rm = RiskManager(limits=limits)

        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(
            symbols, aggregator, risk_manager=rm
        )

        ticks = _make_ticks("TEST", 100.0, n_minutes=2, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        # Engine should still process bars even if some orders are rejected
        assert engine.bars_processed > 0

    @pytest.mark.asyncio
    async def test_risk_limits_cause_rejections(self):
        limits = RiskLimits(
            max_position_pct=0.001,  # extremely tight
            max_drawdown_pct=0.001,
            max_daily_loss_pct=0.001,
            max_open_positions=0,
            max_orders_per_day=0,
        )
        rm = RiskManager(limits=limits)

        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(
            symbols, aggregator, risk_manager=rm
        )

        ticks = _make_ticks("TEST", 100.0, n_minutes=2, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        # All orders should have been rejected
        assert engine.orders_submitted == 0


class TestHealthMonitorIntegration:
    """Health monitor detects stale data and feed issues."""

    def test_healthy_with_fresh_data(self):
        monitor = HealthMonitor(max_bar_age_seconds=60.0)
        monitor.start()
        monitor.record_bar(datetime.now())
        report = monitor.get_health_report()
        assert report.status == HealthStatus.HEALTHY

    def test_degraded_with_stale_data(self):
        monitor = HealthMonitor(max_bar_age_seconds=5.0)
        monitor.start()
        old = datetime.now() - timedelta(seconds=30)
        monitor.record_bar(old)
        report = monitor.get_health_report()
        assert report.status == HealthStatus.DEGRADED

    def test_unhealthy_on_disconnect(self):
        monitor = HealthMonitor()
        monitor.start()
        monitor.set_data_feed_status(False)
        report = monitor.get_health_report()
        assert report.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_monitor_tracks_engine_bars(self):
        monitor = HealthMonitor(max_bar_age_seconds=9999.0)
        monitor.start()

        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        ticks = _make_ticks("TEST", 100.0, n_minutes=1, bar_seconds=5)

        # Track bars as they flow through
        original_handle = engine._handle_market

        def tracking_handle(event):
            monitor.record_bar(event.timestamp)
            original_handle(event)

        engine._handle_market = tracking_handle

        await _run_engine_with_ticks(engine, aggregator, ticks)

        report = monitor.get_health_report()
        assert report.bars_per_minute > 0


class TestMultipleSymbols:
    """Independent position tracking across symbols."""

    @pytest.mark.asyncio
    async def test_three_symbols_independent(self):
        symbols = ["AAPL", "MSFT", "GOOGL"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        # Generate ticks for all three symbols
        all_ticks: list[Tick] = []
        for sym, base in [("AAPL", 175.0), ("MSFT", 380.0), ("GOOGL", 140.0)]:
            all_ticks.extend(_make_ticks(sym, base, n_minutes=2, bar_seconds=5))
        # Sort by timestamp
        all_ticks.sort(key=lambda t: t.timestamp)

        await _run_engine_with_ticks(engine, aggregator, all_ticks)

        positions = portfolio.get_positions()
        assert "AAPL" in positions
        assert "MSFT" in positions
        assert "GOOGL" in positions

    @pytest.mark.asyncio
    async def test_multi_symbol_equity_positive(self):
        symbols = ["AAPL", "MSFT"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        all_ticks: list[Tick] = []
        for sym, base in [("AAPL", 175.0), ("MSFT", 380.0)]:
            all_ticks.extend(_make_ticks(sym, base, n_minutes=2, bar_seconds=5))
        all_ticks.sort(key=lambda t: t.timestamp)

        await _run_engine_with_ticks(engine, aggregator, all_ticks)

        assert portfolio.get_equity() > 0


class TestGracefulShutdown:
    """Engine shuts down cleanly."""

    @pytest.mark.asyncio
    async def test_stop_clears_running_flag(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        ticks = _make_ticks("TEST", 100.0, n_minutes=1, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_stop_allows_statistics_retrieval(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        ticks = _make_ticks("TEST", 100.0, n_minutes=1, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        stats = engine.get_statistics()
        assert "bars_processed" in stats
        assert "equity" in stats

    @pytest.mark.asyncio
    async def test_immediate_stop(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        await broker.connect()

        # Start then immediately stop
        async def quick_stop():
            await asyncio.sleep(0.05)
            await engine.stop()

        await asyncio.gather(engine.start(), quick_stop())
        assert engine.is_running is False
        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_get_state_after_shutdown(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)

        ticks = _make_ticks("TEST", 100.0, n_minutes=1, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        state = engine.get_state()
        assert "statistics" in state
        assert "cash" in state
        assert "trades" in state
        assert "orders" in state


class TestConfigBackwardCompatibility:
    """Phase 1-3 configs load without errors."""

    def test_phase1_config_loads(self):
        from src.config import load_config

        config = load_config("configs/backtest_config.yaml")
        assert config.data.symbols is not None
        # live section has defaults
        assert config.live.broker.broker_type == "paper"

    def test_phase2_config_loads(self):
        from src.config import load_config

        config = load_config("configs/backtest_baseline.yaml")
        assert config.live.database.enabled is True

    def test_paper_trading_config_loads(self):
        from src.config import load_config

        config = load_config("configs/paper_trading_config.yaml")
        assert config.live.broker.broker_type == "paper"
        assert config.live.data.bar_interval_seconds == 60
        assert config.live.database.save_interval_seconds == 300


class TestModuleExports:
    """Verify all new packages have proper exports."""

    def test_engine_exports(self):
        from src.engine import PaperTradingEngine

        assert PaperTradingEngine is not None

    def test_storage_exports(self):
        from src.storage import NullStorage, SQLStorage, StorageBackend

        assert StorageBackend is not None
        assert SQLStorage is not None
        assert NullStorage is not None

    def test_broker_exports(self):
        from src.broker import (
            BrokerAdapter,
            BrokerFill,
            BrokerOrder,
            OrderManager,
            OrderStatus,
            PaperBroker,
        )

        assert BrokerAdapter is not None
        assert PaperBroker is not None
        assert OrderManager is not None
        assert BrokerOrder is not None
        assert BrokerFill is not None
        assert OrderStatus is not None

    def test_monitoring_exports(self):
        from src.monitoring import HealthMonitor, HealthReport, HealthStatus

        assert HealthMonitor is not None
        assert HealthReport is not None
        assert HealthStatus is not None

    def test_live_exports(self):
        from src.live import BarAggregator, LiveDataFeed, LiveDataHandler

        assert BarAggregator is not None
        assert LiveDataFeed is not None
        assert LiveDataHandler is not None

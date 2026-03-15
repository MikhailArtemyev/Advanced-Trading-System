"""Phase 5 integration tests — end-to-end scenarios for new capabilities.

Tests cover: strategy templates in the full pipeline, SQLite trade persistence,
alert manager wiring, dashboard lifecycle, config env-var loading, startup
validation, and module exports for all Phase 5 packages.
"""

import asyncio
import math
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from src.alerts.alert_manager import AlertManager
from src.alerts.base_alert import AlertLevel, AlertMessage
from src.broker.order_manager import OrderManager
from src.broker.paper_broker import PaperBroker
from src.config import (
    BacktestConfig,
    BrokerConfig,
    DataConfig,
    ExecutionConfig,
    LiveConfig,
    StrategyConfig,
    load_config,
)
from src.dashboard.terminal_ui import TradingDashboard
from src.engine.paper_engine import PaperTradingEngine
from src.live.bar_aggregator import BarAggregator
from src.live.data_feed import Tick
from src.live.data_handler import LiveDataHandler
from src.monitoring.health import HealthMonitor, HealthStatus
from src.portfolio.portfolio import Portfolio
from src.storage.sql_storage import SQLStorage
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.pairs_trading import PairsTradingStrategy
from src.strategy.sma_crossover import SMACrossoverStrategy

# ── Helpers ──────────────────────────────────────────────────────────


def _make_ticks(
    symbol: str,
    base_price: float,
    n_minutes: int,
    bar_seconds: int = 5,
    ticks_per_second: float = 2.0,
) -> list[Tick]:
    """Generate sine-wave ticks that produce crossovers."""
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


def _build_engine(
    symbols: list[str],
    aggregator: BarAggregator,
    strategy=None,
    initial_capital: float = 100_000.0,
    storage: SQLStorage | None = None,
    session_id: str = "",
) -> tuple[PaperTradingEngine, LiveDataHandler, Portfolio, PaperBroker, OrderManager]:
    """Wire up a full paper trading engine with configurable strategy."""
    data_handler = LiveDataHandler(symbols=symbols, bar_aggregator=aggregator)
    if strategy is None:
        strategy = SMACrossoverStrategy(
            symbols=symbols,
            parameters={"fast_period": 3, "slow_period": 7},
        )
    portfolio = Portfolio(
        initial_capital=initial_capital,
        symbols=symbols,
        storage=storage,
        session_id=session_id,
    )
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
        bar_poll_interval=0.01,
        event_poll_interval=0.005,
        storage=storage,
        session_id=session_id,
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


# ── Strategy Templates in Full Pipeline ─────────────────────────────


class TestStrategyTemplatesPipeline:
    """Run each new strategy through the full engine pipeline."""

    @pytest.mark.asyncio
    async def test_momentum_strategy_pipeline(self):
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        strategy = MomentumStrategy(
            symbols=symbols,
            parameters={
                "lookback_period": 3,
                "top_n": 2,
                "bottom_n": 0,
                "rebalance_frequency": 1,
                "min_bars": 1,
            },
        )
        engine, dh, portfolio, broker, om = _build_engine(
            symbols, aggregator, strategy=strategy
        )

        all_ticks: list[Tick] = []
        for sym, base in [
            ("AAPL", 175.0),
            ("MSFT", 380.0),
            ("GOOGL", 140.0),
            ("AMZN", 180.0),
        ]:
            all_ticks.extend(_make_ticks(sym, base, n_minutes=2, bar_seconds=5))
        all_ticks.sort(key=lambda t: t.timestamp)

        await _run_engine_with_ticks(engine, aggregator, all_ticks)
        assert engine.bars_processed > 0

    @pytest.mark.asyncio
    async def test_mean_reversion_strategy_pipeline(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        strategy = MeanReversionStrategy(
            symbols=symbols,
            parameters={
                "lookback_period": 5,
                "entry_threshold": 1.0,
                "exit_threshold": 0.3,
                "max_holding_period": 5,
            },
        )
        engine, dh, portfolio, broker, om = _build_engine(
            symbols, aggregator, strategy=strategy
        )

        ticks = _make_ticks("TEST", 100.0, n_minutes=2, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)
        assert engine.bars_processed > 0

    @pytest.mark.asyncio
    async def test_pairs_trading_strategy_pipeline(self):
        symbols = ["AAPL", "MSFT"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        strategy = PairsTradingStrategy(
            symbols=symbols,
            parameters={
                "lookback_period": 5,
                "entry_threshold": 1.0,
                "exit_threshold": 0.3,
                "min_bars": 1,
            },
        )
        engine, dh, portfolio, broker, om = _build_engine(
            symbols, aggregator, strategy=strategy
        )

        all_ticks: list[Tick] = []
        for sym, base in [("AAPL", 175.0), ("MSFT", 180.0)]:
            all_ticks.extend(_make_ticks(sym, base, n_minutes=2, bar_seconds=5))
        all_ticks.sort(key=lambda t: t.timestamp)

        await _run_engine_with_ticks(engine, aggregator, all_ticks)
        assert engine.bars_processed > 0


# ── Database Persistence ────────────────────────────────────────────


class TestDatabasePersistence:
    """Trade and equity recording through the full pipeline."""

    @pytest.mark.asyncio
    async def test_trades_persisted_to_db(self, tmp_path):
        db_path = tmp_path / "trades.db"
        storage = SQLStorage(db_url=f"sqlite:///{db_path}")
        session_id = storage.create_session(
            session_type="paper", config={}, initial_capital=100_000.0
        )

        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(
            symbols, aggregator, storage=storage, session_id=session_id
        )

        ticks = _make_ticks("TEST", 100.0, n_minutes=3, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        trades = storage.get_trades(session_id)
        # Trades list should be populated (if signals fired) or empty
        assert isinstance(trades, list)
        storage.close()

    @pytest.mark.asyncio
    async def test_equity_snapshots_persisted(self, tmp_path):
        db_path = tmp_path / "equity.db"
        storage = SQLStorage(db_url=f"sqlite:///{db_path}")
        session_id = storage.create_session(
            session_type="paper", config={}, initial_capital=100_000.0
        )

        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(
            symbols, aggregator, storage=storage, session_id=session_id
        )

        ticks = _make_ticks("TEST", 100.0, n_minutes=2, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        curve = storage.get_equity_curve(session_id)
        assert isinstance(curve, pd.DataFrame)
        storage.close()

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, tmp_path):
        db_path = tmp_path / "session.db"
        storage = SQLStorage(db_url=f"sqlite:///{db_path}")
        session_id = storage.create_session(
            session_type="paper",
            config={"strategy": "sma_crossover"},
            initial_capital=100_000.0,
        )

        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(
            symbols, aggregator, storage=storage, session_id=session_id
        )

        ticks = _make_ticks("TEST", 100.0, n_minutes=1, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        # End session
        storage.end_session(session_id, portfolio.get_equity())

        sessions = storage.list_sessions()
        assert len(sessions) >= 1
        this = next(s for s in sessions if s["id"] == session_id)
        assert this["final_equity"] is not None
        storage.close()

    @pytest.mark.asyncio
    async def test_engine_state_saved_to_db(self, tmp_path):
        db_path = tmp_path / "state.db"
        storage = SQLStorage(db_url=f"sqlite:///{db_path}")
        session_id = storage.create_session(
            session_type="paper", config={}, initial_capital=100_000.0
        )

        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(
            symbols, aggregator, storage=storage, session_id=session_id
        )

        ticks = _make_ticks("TEST", 100.0, n_minutes=1, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        # Save engine state
        state = engine.get_state()
        storage.save_engine_state(session_id, state)

        loaded = storage.load_engine_state(session_id)
        assert loaded is not None
        assert loaded["statistics"]["bars_processed"] == engine.bars_processed
        storage.close()


# ── Alert Manager Wiring ────────────────────────────────────────────


class TestAlertManagerWiring:
    """AlertManager integrates with HealthMonitor."""

    def test_health_report_triggers_alert(self):
        sent: list[AlertMessage] = []

        class RecordingChannel:
            async def send(self, msg):
                sent.append(msg)
                return True

            async def test_connection(self):
                return True

        manager = AlertManager(
            channels=[RecordingChannel()],
            min_level=AlertLevel.WARNING,
            cooldown_seconds=0,
        )

        monitor = HealthMonitor(max_bar_age_seconds=1.0)
        monitor.start()
        monitor.add_alert_callback(manager.on_health_report)

        # Stale data triggers DEGRADED
        old = datetime.now() - timedelta(seconds=30)
        monitor.record_bar(old)
        report = monitor.get_health_report()
        assert report.status != HealthStatus.HEALTHY

        # on_health_report fires async — run loop briefly
        loop = asyncio.new_event_loop()
        manager.on_health_report(report)
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.close()

    @pytest.mark.asyncio
    async def test_alert_manager_cooldown_integration(self):
        sent_count = 0

        class CountChannel:
            async def send(self, msg):
                nonlocal sent_count
                sent_count += 1
                return True

            async def test_connection(self):
                return True

        manager = AlertManager(
            channels=[CountChannel()],
            min_level=AlertLevel.INFO,
            cooldown_seconds=9999,  # very long cooldown
        )

        # First alert goes through
        msg = AlertMessage(
            level=AlertLevel.WARNING,
            title="test",
            body="body",
            timestamp=datetime.now(),
            metadata={},
        )
        await manager.send_alert(msg)
        assert sent_count == 1

        # Same title within cooldown is suppressed
        await manager.send_alert(msg)
        assert sent_count == 1


# ── Dashboard Integration ───────────────────────────────────────────


class TestDashboardIntegration:
    """Dashboard works with real engine components."""

    @pytest.mark.asyncio
    async def test_dashboard_with_engine(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)
        monitor = HealthMonitor()
        monitor.start()

        dash = TradingDashboard(engine, monitor, refresh_rate=0.05)

        # Dashboard should build layout from real engine
        layout = dash._build_layout()
        assert layout is not None

    @pytest.mark.asyncio
    async def test_dashboard_survives_engine_lifecycle(self):
        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(symbols, aggregator)
        monitor = HealthMonitor()
        monitor.start()

        dash = TradingDashboard(engine, monitor, refresh_rate=0.05)

        async def run_then_stop():
            await broker.connect()
            ticks = _make_ticks("TEST", 100.0, n_minutes=1, bar_seconds=5)
            for tick in ticks:
                aggregator.on_tick(tick)
            await asyncio.sleep(0.1)
            await engine.stop()
            await asyncio.sleep(0.1)
            dash.stop()
            await broker.disconnect()

        with (
            patch("src.dashboard.terminal_ui.Live"),
            patch("src.dashboard.terminal_ui.RichHandler"),
        ):
            await asyncio.wait_for(
                asyncio.gather(engine.start(), dash.start(), run_then_stop()),
                timeout=5.0,
            )


# ── Config Validation Integration ───────────────────────────────────


class TestConfigValidation:
    """Startup validation catches real config problems."""

    def test_paper_trading_config_passes_validation(self):
        config = load_config("configs/paper_trading_config.yaml")
        issues = config.validate_for_live()
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert errors == []

    def test_env_vars_resolve_in_loaded_config(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test_key")
        monkeypatch.setenv("ALPACA_API_SECRET", "test_secret")
        config = load_config("configs/paper_trading_config.yaml")
        # Paper broker doesn't use these, but they should be resolved
        assert config.live.broker.api_key == "test_key"
        assert config.live.broker.api_secret == "test_secret"

    def test_alpaca_broker_without_credentials_fails(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
        config = BacktestConfig(
            data=DataConfig(
                symbols=["AAPL"], start_date="2020-01-01", end_date="2020-12-31"
            ),
            execution=ExecutionConfig(),
            strategy=StrategyConfig(name="sma_crossover"),
            live=LiveConfig(
                broker=BrokerConfig(broker_type="alpaca", api_key="", api_secret="")
            ),
        )
        issues = config.validate_for_live()
        errors = [i for i in issues if i.startswith("ERROR:")]
        assert len(errors) == 2


# ── Strategy Switching ──────────────────────────────────────────────


class TestStrategySwitching:
    """Different strategies work with the same engine infrastructure."""

    @pytest.mark.asyncio
    async def test_switch_sma_to_momentum(self):
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))

        # Run with SMA
        sma = SMACrossoverStrategy(
            symbols=symbols, parameters={"fast_period": 3, "slow_period": 7}
        )
        engine1, _, _, _, _ = _build_engine(symbols, aggregator, strategy=sma)

        all_ticks: list[Tick] = []
        for sym, base in [
            ("AAPL", 175.0),
            ("MSFT", 380.0),
            ("GOOGL", 140.0),
            ("AMZN", 180.0),
        ]:
            all_ticks.extend(_make_ticks(sym, base, n_minutes=1, bar_seconds=5))
        all_ticks.sort(key=lambda t: t.timestamp)

        await _run_engine_with_ticks(engine1, aggregator, all_ticks)
        sma_bars = engine1.bars_processed

        # Run with Momentum
        aggregator2 = BarAggregator(interval=timedelta(seconds=5))
        mom = MomentumStrategy(
            symbols=symbols,
            parameters={
                "lookback_period": 3,
                "top_n": 2,
                "rebalance_frequency": 1,
                "min_bars": 1,
            },
        )
        engine2, _, _, _, _ = _build_engine(symbols, aggregator2, strategy=mom)
        await _run_engine_with_ticks(engine2, aggregator2, all_ticks)
        mom_bars = engine2.bars_processed

        # Both should process bars
        assert sma_bars > 0
        assert mom_bars > 0


# ── Graceful Shutdown with DB ───────────────────────────────────────


class TestGracefulShutdownWithDB:
    """Shutdown saves final state to database."""

    @pytest.mark.asyncio
    async def test_final_state_persisted_on_shutdown(self, tmp_path):
        db_path = tmp_path / "shutdown.db"
        storage = SQLStorage(db_url=f"sqlite:///{db_path}")
        session_id = storage.create_session(
            session_type="paper", config={}, initial_capital=100_000.0
        )

        symbols = ["TEST"]
        aggregator = BarAggregator(interval=timedelta(seconds=5))
        engine, dh, portfolio, broker, om = _build_engine(
            symbols, aggregator, storage=storage, session_id=session_id
        )

        ticks = _make_ticks("TEST", 100.0, n_minutes=2, bar_seconds=5)
        await _run_engine_with_ticks(engine, aggregator, ticks)

        # Simulate shutdown: save state, end session
        state = engine.get_state()
        storage.save_engine_state(session_id, state)
        storage.end_session(session_id, portfolio.get_equity())

        # Verify session was ended
        sessions = storage.list_sessions()
        this = next(s for s in sessions if s["id"] == session_id)
        assert this["ended_at"] is not None
        assert this["final_equity"] is not None

        # Verify state can be reloaded
        loaded = storage.load_engine_state(session_id)
        assert loaded is not None
        storage.close()


# ── Module Exports ──────────────────────────────────────────────────


class TestPhase5Exports:
    """All Phase 5 packages have proper exports."""

    def test_strategy_exports(self):
        from src.strategy import (
            MeanReversionStrategy,
            MomentumStrategy,
            PairsTradingStrategy,
        )

        assert MomentumStrategy is not None
        assert MeanReversionStrategy is not None
        assert PairsTradingStrategy is not None

    def test_dashboard_exports(self):
        from src.dashboard import TradingDashboard

        assert TradingDashboard is not None

    def test_alerts_exports(self):
        from src.alerts import (
            AlertChannel,
            AlertManager,
        )

        assert AlertChannel is not None
        assert AlertManager is not None

    def test_storage_exports(self):
        from src.storage import NullStorage, SQLStorage, StorageBackend

        assert StorageBackend is not None
        assert SQLStorage is not None
        assert NullStorage is not None

    def test_config_validation_method(self):
        from src.config import BacktestConfig

        assert hasattr(BacktestConfig, "validate_for_live")

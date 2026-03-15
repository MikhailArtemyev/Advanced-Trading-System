"""Tests for TradingDashboard."""

import asyncio
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.dashboard.terminal_ui import TradingDashboard
from src.monitoring.health import HealthReport, HealthStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_health_report(**kwargs):
    defaults = {
        "status": HealthStatus.HEALTHY,
        "timestamp": datetime.now(),
        "uptime_seconds": 60.0,
        "data_feed_connected": True,
        "last_bar_age_seconds": 5.0,
        "event_latency_ms": 10.0,
        "bars_per_minute": 12.0,
        "fill_rate": 1.0,
        "warnings": [],
        "errors": [],
    }
    defaults.update(kwargs)
    return HealthReport(**defaults)


def _make_engine_mock(**stat_overrides):
    engine = MagicMock()
    engine.is_running = True
    engine._portfolio.initial_capital = 100_000.0
    engine._portfolio.cash = 95_000.0
    engine._portfolio.get_trade_history.return_value = []

    strategy = MagicMock()
    strategy.__class__.__name__ = "SMACrossover"
    strategy.symbols = ["AAPL", "MSFT"]
    engine._strategy = strategy

    stats = {
        "is_running": True,
        "uptime_seconds": 120.0,
        "bars_processed": 50,
        "events_processed": 100,
        "orders_submitted": 10,
        "orders_rejected": 1,
        "fills_processed": 9,
        "equity": 101_000.0,
        "positions": {},
    }
    stats.update(stat_overrides)
    engine.get_statistics.return_value = stats
    return engine


def _make_health_monitor_mock(**kwargs):
    monitor = MagicMock()
    monitor.get_health_report.return_value = _make_health_report(**kwargs)
    return monitor


# ===========================================================================
# Init
# ===========================================================================


class TestInit:
    def test_create_dashboard(self):
        engine = _make_engine_mock()
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        assert dash._refresh_rate == 1.0
        assert dash._running is False

    def test_custom_refresh_rate(self):
        engine = _make_engine_mock()
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor, refresh_rate=0.5)
        assert dash._refresh_rate == 0.5


# ===========================================================================
# Layout building
# ===========================================================================


class TestLayout:
    def test_build_layout_returns_layout(self):
        from rich.layout import Layout

        engine = _make_engine_mock()
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        layout = dash._build_layout()
        assert isinstance(layout, Layout)

    def test_build_layout_calls_statistics(self):
        engine = _make_engine_mock()
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        dash._build_layout()
        engine.get_statistics.assert_called_once()
        monitor.get_health_report.assert_called_once()

    def test_layout_with_positions(self):
        engine = _make_engine_mock(
            positions={
                "AAPL": {
                    "quantity": 100,
                    "avg_cost": 175.0,
                    "market_value": 17_600.0,
                    "unrealized_pnl": 100.0,
                },
            }
        )
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        layout = dash._build_layout()
        assert layout is not None

    def test_layout_with_no_positions(self):
        engine = _make_engine_mock(positions={})
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        layout = dash._build_layout()
        assert layout is not None


# ===========================================================================
# Panels
# ===========================================================================


class TestPanels:
    def test_header_panel(self):
        engine = _make_engine_mock(uptime_seconds=3661.0)
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        stats = engine.get_statistics()
        panel = dash._build_header(stats)
        assert panel is not None

    def test_equity_panel_positive_pnl(self):
        engine = _make_engine_mock(equity=105_000.0)
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        stats = engine.get_statistics()
        panel = dash._build_equity_panel(stats)
        assert panel is not None

    def test_equity_panel_negative_pnl(self):
        engine = _make_engine_mock(equity=95_000.0)
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        stats = engine.get_statistics()
        panel = dash._build_equity_panel(stats)
        assert panel is not None

    def test_trades_panel_with_trades(self):
        engine = _make_engine_mock()
        trade = MagicMock()
        trade.timestamp = datetime.now()
        trade.symbol = "AAPL"
        trade.side = "BUY"
        trade.quantity = 100
        trade.price = 175.50
        trade.pnl = 0.0
        engine._portfolio.get_trade_history.return_value = [trade]
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        panel = dash._build_trades_panel()
        assert panel is not None

    def test_health_panel_degraded(self):
        engine = _make_engine_mock()
        monitor = _make_health_monitor_mock(
            status=HealthStatus.DEGRADED,
            warnings=["High latency"],
        )
        dash = TradingDashboard(engine, monitor)
        report = monitor.get_health_report()
        panel = dash._build_health_panel(report)
        assert panel is not None

    def test_health_panel_unhealthy_with_errors(self):
        engine = _make_engine_mock()
        monitor = _make_health_monitor_mock(
            status=HealthStatus.UNHEALTHY,
            errors=["Feed disconnected"],
        )
        dash = TradingDashboard(engine, monitor)
        report = monitor.get_health_report()
        panel = dash._build_health_panel(report)
        assert panel is not None

    def test_stats_panel(self):
        engine = _make_engine_mock()
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        stats = engine.get_statistics()
        panel = dash._build_stats_panel(stats)
        assert panel is not None

    def test_positions_panel_skips_zero_qty(self):
        engine = _make_engine_mock(
            positions={
                "AAPL": {
                    "quantity": 0,
                    "avg_cost": 0,
                    "market_value": 0,
                    "unrealized_pnl": 0,
                },
            }
        )
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        stats = engine.get_statistics()
        panel = dash._build_positions_panel(stats)
        assert panel is not None


# ===========================================================================
# Lifecycle
# ===========================================================================


class TestLifecycle:
    def test_stop_sets_running_false(self):
        engine = _make_engine_mock()
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor)
        dash._running = True
        dash.stop()
        assert dash._running is False

    @pytest.mark.asyncio
    async def test_keeps_running_after_engine_stops(self):
        """Dashboard stays up after engine stops, showing final state."""
        engine = _make_engine_mock()
        engine.is_running = True
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor, refresh_rate=0.05)

        async def stop_after_engine():
            await asyncio.sleep(0.05)
            engine.is_running = False  # engine stops
            await asyncio.sleep(0.1)  # dashboard should still be running
            assert dash._running is True
            dash.stop()  # explicit stop needed

        with (
            patch("src.dashboard.terminal_ui.Live"),
            patch("src.dashboard.terminal_ui.RichHandler"),
        ):
            await asyncio.wait_for(
                asyncio.gather(dash.start(), stop_after_engine()),
                timeout=2.0,
            )
        assert dash._running is False

    @pytest.mark.asyncio
    async def test_start_stops_on_stop_call(self):
        engine = _make_engine_mock()
        engine.is_running = True
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor, refresh_rate=0.05)

        async def stop_soon():
            await asyncio.sleep(0.1)
            dash.stop()

        with (
            patch("src.dashboard.terminal_ui.Live"),
            patch("src.dashboard.terminal_ui.RichHandler"),
        ):
            await asyncio.wait_for(
                asyncio.gather(dash.start(), stop_soon()),
                timeout=2.0,
            )
        assert dash._running is False

    @pytest.mark.asyncio
    async def test_start_restores_log_handlers(self):
        """Logging handlers are restored after dashboard stops."""
        engine = _make_engine_mock()
        engine.is_running = True
        monitor = _make_health_monitor_mock()
        dash = TradingDashboard(engine, monitor, refresh_rate=0.05)

        root = logging.getLogger()
        original_handlers = root.handlers[:]

        async def stop_soon():
            await asyncio.sleep(0.1)
            dash.stop()

        with (
            patch("src.dashboard.terminal_ui.Live"),
            patch("src.dashboard.terminal_ui.RichHandler"),
        ):
            await asyncio.wait_for(
                asyncio.gather(dash.start(), stop_soon()),
                timeout=2.0,
            )

        assert root.handlers == original_handlers


# ===========================================================================
# Module exports
# ===========================================================================


class TestExports:
    def test_importable(self):
        from src.dashboard import TradingDashboard

        assert TradingDashboard is not None

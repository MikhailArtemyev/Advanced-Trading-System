"""Tests for the pub/sub alerting system."""

import asyncio
from datetime import datetime

import pytest

from src.alerts.alert_manager import AlertPublisher
from src.alerts.base_alert import AlertLevel, AlertMessage, AlertSubscriber
from src.monitoring.health import HealthReport, HealthStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSubscriber(AlertSubscriber):
    """In-memory subscriber for testing."""

    def __init__(self, succeed: bool = True) -> None:
        self.received: list[AlertMessage] = []
        self._succeed = succeed

    async def on_alert(self, message: AlertMessage) -> bool:
        self.received.append(message)
        return self._succeed

    async def test_connection(self) -> bool:
        return self._succeed


def _make_health_report(
    status: HealthStatus = HealthStatus.DEGRADED,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> HealthReport:
    return HealthReport(
        status=status,
        timestamp=datetime.now(),
        uptime_seconds=60.0,
        data_feed_connected=True,
        last_bar_age_seconds=5.0,
        event_latency_ms=10.0,
        bars_per_minute=1.0,
        fill_rate=1.0,
        warnings=warnings or [],
        errors=errors or [],
    )


# ===========================================================================
# AlertLevel / AlertMessage
# ===========================================================================


class TestAlertLevel:
    def test_values(self):
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"

    def test_all_members(self):
        assert len(AlertLevel) == 3


class TestAlertMessage:
    def test_defaults(self):
        msg = AlertMessage(level=AlertLevel.INFO, title="t", body="b")
        assert msg.level == AlertLevel.INFO
        assert msg.title == "t"
        assert msg.body == "b"
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}

    def test_with_metadata(self):
        msg = AlertMessage(
            level=AlertLevel.CRITICAL,
            title="x",
            body="y",
            metadata={"key": "val"},
        )
        assert msg.metadata == {"key": "val"}


# ===========================================================================
# AlertPublisher — subscribe / publish
# ===========================================================================


class TestAlertPublisher:
    @pytest.mark.asyncio
    async def test_publish_to_all_subscribers(self):
        sub1 = FakeSubscriber()
        sub2 = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub1, min_level=AlertLevel.INFO)
        pub.subscribe(sub2, min_level=AlertLevel.INFO)

        msg = AlertMessage(level=AlertLevel.INFO, title="T", body="B")
        results = await pub.publish(msg)

        assert results == [True, True]
        assert len(sub1.received) == 1
        assert len(sub2.received) == 1

    @pytest.mark.asyncio
    async def test_level_filtering(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.WARNING)

        info_msg = AlertMessage(level=AlertLevel.INFO, title="I", body="B")
        results = await pub.publish(info_msg)
        assert results == []
        assert len(sub.received) == 0

        warn_msg = AlertMessage(level=AlertLevel.WARNING, title="W", body="B")
        results = await pub.publish(warn_msg)
        assert results == [True]
        assert len(sub.received) == 1

    @pytest.mark.asyncio
    async def test_critical_passes_warning_filter(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.WARNING)

        msg = AlertMessage(level=AlertLevel.CRITICAL, title="C", body="B")
        results = await pub.publish(msg)
        assert results == [True]

    @pytest.mark.asyncio
    async def test_per_subscriber_level_filtering(self):
        info_sub = FakeSubscriber()
        warn_sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(info_sub, min_level=AlertLevel.INFO)
        pub.subscribe(warn_sub, min_level=AlertLevel.WARNING)

        msg = AlertMessage(level=AlertLevel.INFO, title="T", body="B")
        results = await pub.publish(msg)

        assert results == [True]  # only info_sub receives it
        assert len(info_sub.received) == 1
        assert len(warn_sub.received) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.INFO)
        pub.unsubscribe(sub)

        msg = AlertMessage(level=AlertLevel.INFO, title="T", body="B")
        results = await pub.publish(msg)

        assert results == []
        assert len(sub.received) == 0

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_repeat(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=60.0)
        pub.subscribe(sub, min_level=AlertLevel.INFO)

        msg = AlertMessage(level=AlertLevel.WARNING, title="Dup", body="B")
        r1 = await pub.publish(msg)
        r2 = await pub.publish(msg)

        assert r1 == [True]
        assert r2 == []  # suppressed
        assert len(sub.received) == 1

    @pytest.mark.asyncio
    async def test_cooldown_zero_disables(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.INFO)

        msg = AlertMessage(level=AlertLevel.WARNING, title="Dup", body="B")
        await pub.publish(msg)
        await pub.publish(msg)
        assert len(sub.received) == 2

    @pytest.mark.asyncio
    async def test_cooldown_expires(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0.01)
        pub.subscribe(sub, min_level=AlertLevel.INFO)

        msg = AlertMessage(level=AlertLevel.WARNING, title="Exp", body="B")
        await pub.publish(msg)
        await asyncio.sleep(0.02)
        await pub.publish(msg)
        assert len(sub.received) == 2

    @pytest.mark.asyncio
    async def test_subscriber_failure_reported(self):
        sub = FakeSubscriber(succeed=False)
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.INFO)

        msg = AlertMessage(level=AlertLevel.WARNING, title="T", body="B")
        results = await pub.publish(msg)
        assert results == [False]

    @pytest.mark.asyncio
    async def test_subscriber_exception_handled(self):
        sub = FakeSubscriber()

        async def boom(message: AlertMessage) -> bool:
            raise RuntimeError("kaboom")

        sub.on_alert = boom  # type: ignore[assignment]
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.INFO)

        msg = AlertMessage(level=AlertLevel.WARNING, title="T", body="B")
        results = await pub.publish(msg)
        assert results == [False]

    @pytest.mark.asyncio
    async def test_publish_trade(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.INFO)

        trade = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "price": 150.0,
            "pnl": 0.0,
        }
        results = await pub.publish_trade(trade)
        assert results == [True]
        assert "AAPL" in sub.received[0].title

    @pytest.mark.asyncio
    async def test_publish_health(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.WARNING)

        report = _make_health_report(HealthStatus.DEGRADED, warnings=["stale bar"])
        results = await pub.publish_health(report)
        assert results == [True]
        assert "degraded" in sub.received[0].title.lower()

    @pytest.mark.asyncio
    async def test_publish_health_critical(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.WARNING)

        report = _make_health_report(HealthStatus.UNHEALTHY, errors=["feed down"])
        results = await pub.publish_health(report)
        assert results == [True]
        assert sub.received[0].level == AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_publish_drawdown_warning(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.WARNING)

        results = await pub.publish_drawdown(5.0)
        assert results == [True]
        assert sub.received[0].level == AlertLevel.WARNING

    @pytest.mark.asyncio
    async def test_publish_drawdown_critical(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.WARNING)

        results = await pub.publish_drawdown(12.0)
        assert results == [True]
        assert sub.received[0].level == AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_on_health_report_fires_task(self):
        sub = FakeSubscriber()
        pub = AlertPublisher(cooldown_seconds=0)
        pub.subscribe(sub, min_level=AlertLevel.WARNING)

        report = _make_health_report(HealthStatus.DEGRADED, warnings=["w"])
        pub.on_health_report(report)
        # Give the fire-and-forget task a moment
        await asyncio.sleep(0.05)
        assert len(sub.received) == 1


# ===========================================================================
# Config
# ===========================================================================


class TestAlertConfig:
    def test_defaults(self):
        from src.config import AlertConfig

        cfg = AlertConfig()
        assert cfg.enabled is False
        assert cfg.channels == []
        assert cfg.min_level == "warning"
        assert cfg.cooldown_seconds == 300.0

    def test_custom_values(self):
        from src.config import AlertConfig

        cfg = AlertConfig(
            enabled=True,
            channels=[{"type": "telegram"}],
            min_level="critical",
            cooldown_seconds=60.0,
        )
        assert cfg.enabled is True
        assert len(cfg.channels) == 1
        assert cfg.min_level == "critical"

    def test_invalid_min_level(self):
        from pydantic import ValidationError

        from src.config import AlertConfig

        with pytest.raises(ValidationError):
            AlertConfig(min_level="invalid")

    def test_live_config_has_alerts(self):
        from src.config import LiveConfig

        live = LiveConfig()
        assert live.alerts.enabled is False

    def test_backtest_config_roundtrip(self, tmp_path):
        import yaml

        from src.config import load_config

        cfg = {
            "data": {
                "symbols": ["AAPL"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
            "execution": {"initial_capital": 100000},
            "strategy": {"name": "sma_crossover"},
            "live": {
                "alerts": {
                    "enabled": True,
                    "min_level": "info",
                    "cooldown_seconds": 60,
                    "channels": [{"type": "telegram"}],
                }
            },
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg))
        loaded = load_config(str(path))
        assert loaded.live.alerts.enabled is True
        assert loaded.live.alerts.min_level == "info"
        assert len(loaded.live.alerts.channels) == 1


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    def test_alerts_package_exports(self):
        from src.alerts import (
            AlertLevel,
            AlertMessage,
            AlertPublisher,
            AlertSubscriber,
        )

        assert AlertLevel is not None
        assert AlertMessage is not None
        assert AlertPublisher is not None
        assert AlertSubscriber is not None

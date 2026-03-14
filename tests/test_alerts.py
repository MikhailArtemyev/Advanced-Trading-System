"""Tests for the alerting and notification system."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.alerts.alert_manager import AlertManager
from src.alerts.base_alert import AlertChannel, AlertLevel, AlertMessage
from src.alerts.email_alert import EmailAlert
from src.alerts.slack_alert import SlackAlert
from src.alerts.webhook_alert import WebhookAlert
from src.monitoring.health import HealthReport, HealthStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeChannel(AlertChannel):
    """In-memory channel for testing."""

    def __init__(self, succeed: bool = True) -> None:
        self.sent: list[AlertMessage] = []
        self._succeed = succeed

    async def send(self, message: AlertMessage) -> bool:
        self.sent.append(message)
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
# SlackAlert
# ===========================================================================


def _mock_aiohttp_session(response_status: int, response_text: str = "ok"):
    """Create a properly mocked aiohttp.ClientSession context manager."""
    mock_resp = MagicMock()
    mock_resp.status = response_status
    mock_resp.text = AsyncMock(return_value=response_text)

    mock_resp_ctx = AsyncMock()
    mock_resp_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp_ctx)
    mock_session.put = MagicMock(return_value=mock_resp_ctx)

    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    return mock_session_ctx


class TestSlackAlert:
    @pytest.mark.asyncio
    async def test_send_success(self):
        slack = SlackAlert(webhook_url="https://hooks.slack.com/test")
        msg = AlertMessage(level=AlertLevel.WARNING, title="t", body="b")

        with patch(
            "src.alerts.slack_alert.aiohttp.ClientSession",
            return_value=_mock_aiohttp_session(200),
        ):
            result = await slack.send(msg)

        assert result is True

    @pytest.mark.asyncio
    async def test_send_failure_status(self):
        slack = SlackAlert(webhook_url="https://hooks.slack.com/test")
        msg = AlertMessage(level=AlertLevel.WARNING, title="t", body="b")

        with patch(
            "src.alerts.slack_alert.aiohttp.ClientSession",
            return_value=_mock_aiohttp_session(500, "error"),
        ):
            result = await slack.send(msg)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_exception(self):
        slack = SlackAlert(webhook_url="https://hooks.slack.com/test")
        msg = AlertMessage(level=AlertLevel.INFO, title="t", body="b")

        with patch(
            "src.alerts.slack_alert.aiohttp.ClientSession",
            side_effect=Exception("network"),
        ):
            result = await slack.send(msg)

        assert result is False

    def test_build_payload_basic(self):
        slack = SlackAlert(webhook_url="https://hooks.slack.com/test")
        msg = AlertMessage(level=AlertLevel.WARNING, title="T", body="B")
        payload = slack._build_payload(msg)
        assert "text" in payload
        assert "*T*" in payload["text"]
        assert "B" in payload["text"]

    def test_build_payload_with_channel(self):
        slack = SlackAlert(
            webhook_url="https://hooks.slack.com/test", channel="#alerts"
        )
        msg = AlertMessage(level=AlertLevel.INFO, title="T", body="B")
        payload = slack._build_payload(msg)
        assert payload["channel"] == "#alerts"

    def test_build_payload_with_metadata(self):
        slack = SlackAlert(webhook_url="https://hooks.slack.com/test")
        msg = AlertMessage(
            level=AlertLevel.INFO,
            title="T",
            body="B",
            metadata={"foo": "bar"},
        )
        payload = slack._build_payload(msg)
        assert "blocks" in payload

    @pytest.mark.asyncio
    async def test_test_connection(self):
        slack = SlackAlert(webhook_url="https://hooks.slack.com/test")
        with patch.object(slack, "send", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True
            result = await slack.test_connection()
        assert result is True
        assert mock_send.call_count == 1


# ===========================================================================
# EmailAlert
# ===========================================================================


class TestEmailAlert:
    def _make_email(self) -> EmailAlert:
        return EmailAlert(
            smtp_host="smtp.test.com",
            smtp_port=587,
            username="user",
            password="pass",
            from_address="from@test.com",
            to_addresses=["to@test.com"],
        )

    @pytest.mark.asyncio
    async def test_send_success(self):
        email = self._make_email()
        msg = AlertMessage(level=AlertLevel.WARNING, title="T", body="B")

        with patch.object(email, "_send_sync") as mock_sync:
            result = await email.send(msg)

        assert result is True
        mock_sync.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_send_failure(self):
        email = self._make_email()
        msg = AlertMessage(level=AlertLevel.WARNING, title="T", body="B")

        with patch.object(email, "_send_sync", side_effect=Exception("smtp error")):
            result = await email.send(msg)

        assert result is False

    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        email = self._make_email()

        with patch.object(email, "_connect_and_quit"):
            result = await email.test_connection()

        assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        email = self._make_email()

        with patch.object(email, "_connect_and_quit", side_effect=Exception("refused")):
            result = await email.test_connection()

        assert result is False

    def test_send_sync_builds_message(self):
        email = self._make_email()
        msg = AlertMessage(
            level=AlertLevel.CRITICAL,
            title="Alert",
            body="Body text",
            metadata={"key": "val"},
        )

        with patch("src.alerts.email_alert.smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            email._send_sync(msg)

        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.send_message.assert_called_once()


# ===========================================================================
# WebhookAlert
# ===========================================================================


class TestWebhookAlert:
    @pytest.mark.asyncio
    async def test_send_success(self):
        wh = WebhookAlert(url="https://example.com/hook")
        msg = AlertMessage(level=AlertLevel.WARNING, title="T", body="B")

        with patch(
            "src.alerts.webhook_alert.aiohttp.ClientSession",
            return_value=_mock_aiohttp_session(200),
        ):
            result = await wh.send(msg)

        assert result is True

    @pytest.mark.asyncio
    async def test_send_non_2xx(self):
        wh = WebhookAlert(url="https://example.com/hook")
        msg = AlertMessage(level=AlertLevel.WARNING, title="T", body="B")

        with patch(
            "src.alerts.webhook_alert.aiohttp.ClientSession",
            return_value=_mock_aiohttp_session(403, "forbidden"),
        ):
            result = await wh.send(msg)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_exception(self):
        wh = WebhookAlert(url="https://example.com/hook")
        msg = AlertMessage(level=AlertLevel.INFO, title="T", body="B")

        with patch(
            "src.alerts.webhook_alert.aiohttp.ClientSession",
            side_effect=Exception("net"),
        ):
            result = await wh.send(msg)

        assert result is False

    def test_build_payload(self):
        payload = WebhookAlert._build_payload(
            AlertMessage(
                level=AlertLevel.CRITICAL,
                title="T",
                body="B",
                metadata={"k": "v"},
            )
        )
        assert payload["level"] == "critical"
        assert payload["title"] == "T"
        assert payload["body"] == "B"
        assert payload["metadata"] == {"k": "v"}
        assert "timestamp" in payload

    @pytest.mark.asyncio
    async def test_custom_method_put(self):
        wh = WebhookAlert(url="https://example.com/hook", method="PUT")
        msg = AlertMessage(level=AlertLevel.INFO, title="T", body="B")

        with patch(
            "src.alerts.webhook_alert.aiohttp.ClientSession",
            return_value=_mock_aiohttp_session(200),
        ):
            result = await wh.send(msg)

        assert result is True

    @pytest.mark.asyncio
    async def test_test_connection(self):
        wh = WebhookAlert(url="https://example.com/hook")
        with patch.object(wh, "send", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True
            result = await wh.test_connection()
        assert result is True


# ===========================================================================
# AlertManager
# ===========================================================================


class TestAlertManager:
    @pytest.mark.asyncio
    async def test_send_to_all_channels(self):
        ch1 = FakeChannel()
        ch2 = FakeChannel()
        mgr = AlertManager(channels=[ch1, ch2], min_level=AlertLevel.INFO)

        msg = AlertMessage(level=AlertLevel.INFO, title="T", body="B")
        results = await mgr.send_alert(msg)

        assert results == [True, True]
        assert len(ch1.sent) == 1
        assert len(ch2.sent) == 1

    @pytest.mark.asyncio
    async def test_level_filtering(self):
        ch = FakeChannel()
        mgr = AlertManager(channels=[ch], min_level=AlertLevel.WARNING)

        info_msg = AlertMessage(level=AlertLevel.INFO, title="I", body="B")
        results = await mgr.send_alert(info_msg)
        assert results == []
        assert len(ch.sent) == 0

        warn_msg = AlertMessage(level=AlertLevel.WARNING, title="W", body="B")
        results = await mgr.send_alert(warn_msg)
        assert results == [True]
        assert len(ch.sent) == 1

    @pytest.mark.asyncio
    async def test_critical_passes_warning_filter(self):
        ch = FakeChannel()
        mgr = AlertManager(channels=[ch], min_level=AlertLevel.WARNING)

        msg = AlertMessage(level=AlertLevel.CRITICAL, title="C", body="B")
        results = await mgr.send_alert(msg)
        assert results == [True]

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_repeat(self):
        ch = FakeChannel()
        mgr = AlertManager(
            channels=[ch],
            min_level=AlertLevel.INFO,
            cooldown_seconds=60.0,
        )

        msg = AlertMessage(level=AlertLevel.WARNING, title="Dup", body="B")
        r1 = await mgr.send_alert(msg)
        r2 = await mgr.send_alert(msg)

        assert r1 == [True]
        assert r2 == []  # suppressed
        assert len(ch.sent) == 1

    @pytest.mark.asyncio
    async def test_cooldown_zero_disables(self):
        ch = FakeChannel()
        mgr = AlertManager(
            channels=[ch],
            min_level=AlertLevel.INFO,
            cooldown_seconds=0,
        )

        msg = AlertMessage(level=AlertLevel.WARNING, title="Dup", body="B")
        await mgr.send_alert(msg)
        await mgr.send_alert(msg)
        assert len(ch.sent) == 2

    @pytest.mark.asyncio
    async def test_cooldown_expires(self):
        ch = FakeChannel()
        mgr = AlertManager(
            channels=[ch],
            min_level=AlertLevel.INFO,
            cooldown_seconds=0.01,
        )

        msg = AlertMessage(level=AlertLevel.WARNING, title="Exp", body="B")
        await mgr.send_alert(msg)
        await asyncio.sleep(0.02)
        await mgr.send_alert(msg)
        assert len(ch.sent) == 2

    @pytest.mark.asyncio
    async def test_channel_failure_reported(self):
        ch = FakeChannel(succeed=False)
        mgr = AlertManager(channels=[ch], min_level=AlertLevel.INFO)

        msg = AlertMessage(level=AlertLevel.WARNING, title="T", body="B")
        results = await mgr.send_alert(msg)
        assert results == [False]

    @pytest.mark.asyncio
    async def test_channel_exception_handled(self):
        ch = FakeChannel()

        async def boom(message: AlertMessage) -> bool:
            raise RuntimeError("kaboom")

        ch.send = boom  # type: ignore[assignment]
        mgr = AlertManager(channels=[ch], min_level=AlertLevel.INFO)

        msg = AlertMessage(level=AlertLevel.WARNING, title="T", body="B")
        results = await mgr.send_alert(msg)
        assert results == [False]

    @pytest.mark.asyncio
    async def test_send_trade_alert(self):
        ch = FakeChannel()
        mgr = AlertManager(channels=[ch], min_level=AlertLevel.INFO)

        trade = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "price": 150.0,
            "pnl": 0.0,
        }
        results = await mgr.send_trade_alert(trade)
        assert results == [True]
        assert "AAPL" in ch.sent[0].title

    @pytest.mark.asyncio
    async def test_send_health_alert(self):
        ch = FakeChannel()
        mgr = AlertManager(channels=[ch], min_level=AlertLevel.WARNING)

        report = _make_health_report(HealthStatus.DEGRADED, warnings=["stale bar"])
        results = await mgr.send_health_alert(report)
        assert results == [True]
        assert "degraded" in ch.sent[0].title.lower()

    @pytest.mark.asyncio
    async def test_send_health_alert_critical(self):
        ch = FakeChannel()
        mgr = AlertManager(channels=[ch], min_level=AlertLevel.WARNING)

        report = _make_health_report(HealthStatus.UNHEALTHY, errors=["feed down"])
        results = await mgr.send_health_alert(report)
        assert results == [True]
        assert ch.sent[0].level == AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_send_drawdown_alert_warning(self):
        ch = FakeChannel()
        mgr = AlertManager(channels=[ch], min_level=AlertLevel.WARNING)

        results = await mgr.send_drawdown_alert(5.0)
        assert results == [True]
        assert ch.sent[0].level == AlertLevel.WARNING

    @pytest.mark.asyncio
    async def test_send_drawdown_alert_critical(self):
        ch = FakeChannel()
        mgr = AlertManager(channels=[ch], min_level=AlertLevel.WARNING)

        results = await mgr.send_drawdown_alert(12.0)
        assert results == [True]
        assert ch.sent[0].level == AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_on_health_report_fires_task(self):
        ch = FakeChannel()
        mgr = AlertManager(
            channels=[ch],
            min_level=AlertLevel.WARNING,
            cooldown_seconds=0,
        )

        report = _make_health_report(HealthStatus.DEGRADED, warnings=["w"])
        mgr.on_health_report(report)
        # Give the fire-and-forget task a moment
        await asyncio.sleep(0.05)
        assert len(ch.sent) == 1


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
            channels=[{"type": "slack", "webhook_url": "https://x"}],
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
                    "channels": [{"type": "webhook", "url": "https://example.com"}],
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
            AlertChannel,
            AlertLevel,
            AlertManager,
            AlertMessage,
            EmailAlert,
            SlackAlert,
            WebhookAlert,
        )

        assert AlertChannel is not None
        assert AlertLevel is not None
        assert AlertManager is not None
        assert AlertMessage is not None
        assert EmailAlert is not None
        assert SlackAlert is not None
        assert WebhookAlert is not None

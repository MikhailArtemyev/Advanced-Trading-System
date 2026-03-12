"""Tests for HealthMonitor — engine health tracking."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.monitoring.health import HealthMonitor, HealthReport, HealthStatus


@pytest.fixture
def monitor():
    """HealthMonitor with tight thresholds for testing."""
    m = HealthMonitor(
        max_bar_age_seconds=10.0,
        max_event_latency_ms=100.0,
        history_window=50,
    )
    m.start()
    return m


class TestHealthStatusEnum:
    def test_healthy_value(self):
        assert HealthStatus.HEALTHY.value == "healthy"

    def test_degraded_value(self):
        assert HealthStatus.DEGRADED.value == "degraded"

    def test_unhealthy_value(self):
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestHealthyState:
    def test_healthy_when_all_normal(self, monitor):
        now = datetime.now()
        monitor.record_bar(now)
        monitor.record_event_latency(5.0)
        monitor.record_order_result(filled=True)

        report = monitor.get_health_report()
        assert report.status == HealthStatus.HEALTHY
        assert report.warnings == []
        assert report.errors == []

    def test_healthy_with_no_bars_under_threshold(self):
        m = HealthMonitor(max_bar_age_seconds=9999.0)
        m.start()
        report = m.get_health_report()
        assert report.status == HealthStatus.HEALTHY


class TestDegradedState:
    def test_degraded_when_bar_age_exceeds_threshold(self, monitor):
        old_time = datetime.now() - timedelta(seconds=30)
        monitor.record_bar(old_time)

        report = monitor.get_health_report()
        assert report.status == HealthStatus.DEGRADED
        assert any("bar age" in w.lower() for w in report.warnings)

    def test_degraded_when_latency_exceeds_threshold(self, monitor):
        now = datetime.now()
        monitor.record_bar(now)
        for _ in range(10):
            monitor.record_event_latency(200.0)

        report = monitor.get_health_report()
        assert report.status == HealthStatus.DEGRADED
        assert any("latency" in w.lower() for w in report.warnings)

    def test_multiple_warnings_combine(self, monitor):
        old_time = datetime.now() - timedelta(seconds=30)
        monitor.record_bar(old_time)
        for _ in range(5):
            monitor.record_event_latency(200.0)

        report = monitor.get_health_report()
        assert report.status == HealthStatus.DEGRADED
        assert len(report.warnings) == 2


class TestUnhealthyState:
    def test_unhealthy_when_feed_disconnected(self, monitor):
        monitor.set_data_feed_status(False)
        report = monitor.get_health_report()
        assert report.status == HealthStatus.UNHEALTHY
        assert any("disconnected" in e.lower() for e in report.errors)

    def test_unhealthy_overrides_degraded(self, monitor):
        monitor.set_data_feed_status(False)
        old_time = datetime.now() - timedelta(seconds=30)
        monitor.record_bar(old_time)

        report = monitor.get_health_report()
        assert report.status == HealthStatus.UNHEALTHY


class TestRecordBar:
    def test_updates_last_bar_time(self, monitor):
        now = datetime.now()
        monitor.record_bar(now)
        report = monitor.get_health_report()
        assert report.last_bar_age_seconds < 1.0

    def test_increments_bar_count(self, monitor):
        now = datetime.now()
        monitor.record_bar(now)
        monitor.record_bar(now)
        assert monitor._bar_count == 2


class TestRecordEventLatency:
    def test_average_computed(self, monitor):
        monitor.record_bar(datetime.now())
        monitor.record_event_latency(10.0)
        monitor.record_event_latency(20.0)
        monitor.record_event_latency(30.0)

        report = monitor.get_health_report()
        assert abs(report.event_latency_ms - 20.0) < 0.01

    def test_respects_history_window(self):
        m = HealthMonitor(history_window=3)
        m.start()
        m.record_bar(datetime.now())
        m.record_event_latency(100.0)
        m.record_event_latency(100.0)
        m.record_event_latency(100.0)
        m.record_event_latency(10.0)
        m.record_event_latency(10.0)
        m.record_event_latency(10.0)

        report = m.get_health_report()
        assert abs(report.event_latency_ms - 10.0) < 0.01

    def test_zero_latency_when_no_measurements(self, monitor):
        monitor.record_bar(datetime.now())
        report = monitor.get_health_report()
        assert report.event_latency_ms == 0.0


class TestRecordOrderResult:
    def test_fill_rate_all_filled(self, monitor):
        monitor.record_bar(datetime.now())
        monitor.record_order_result(filled=True)
        monitor.record_order_result(filled=True)

        report = monitor.get_health_report()
        assert report.fill_rate == 1.0

    def test_fill_rate_half_filled(self, monitor):
        monitor.record_bar(datetime.now())
        monitor.record_order_result(filled=True)
        monitor.record_order_result(filled=False)

        report = monitor.get_health_report()
        assert report.fill_rate == 0.5

    def test_fill_rate_default_when_no_orders(self, monitor):
        monitor.record_bar(datetime.now())
        report = monitor.get_health_report()
        assert report.fill_rate == 1.0


class TestBarsPerMinute:
    def test_calculation(self):
        m = HealthMonitor(max_bar_age_seconds=9999.0)
        m._started_at = datetime.now() - timedelta(minutes=2)
        m.record_bar(datetime.now())
        m.record_bar(datetime.now())
        m.record_bar(datetime.now())
        m.record_bar(datetime.now())
        m.record_bar(datetime.now())
        m.record_bar(datetime.now())

        report = m.get_health_report()
        assert abs(report.bars_per_minute - 3.0) < 0.2

    def test_zero_when_no_uptime(self):
        m = HealthMonitor()
        report = m.get_health_report()
        assert report.bars_per_minute == 0.0


class TestAlertCallbacks:
    def test_callback_fired_on_degraded(self, monitor):
        cb = MagicMock()
        monitor.add_alert_callback(cb)

        old_time = datetime.now() - timedelta(seconds=30)
        monitor.record_bar(old_time)
        monitor.get_health_report()

        cb.assert_called_once()
        report = cb.call_args[0][0]
        assert isinstance(report, HealthReport)
        assert report.status == HealthStatus.DEGRADED

    def test_callback_fired_on_unhealthy(self, monitor):
        cb = MagicMock()
        monitor.add_alert_callback(cb)

        monitor.set_data_feed_status(False)
        monitor.get_health_report()

        cb.assert_called_once()

    def test_callback_not_fired_on_healthy(self, monitor):
        cb = MagicMock()
        monitor.add_alert_callback(cb)

        monitor.record_bar(datetime.now())
        monitor.get_health_report()

        cb.assert_not_called()

    def test_multiple_callbacks(self, monitor):
        cb1, cb2 = MagicMock(), MagicMock()
        monitor.add_alert_callback(cb1)
        monitor.add_alert_callback(cb2)

        monitor.set_data_feed_status(False)
        monitor.get_health_report()

        cb1.assert_called_once()
        cb2.assert_called_once()


class TestReset:
    def test_clears_bar_count(self, monitor):
        monitor.record_bar(datetime.now())
        monitor.reset()
        assert monitor._bar_count == 0

    def test_clears_latencies(self, monitor):
        monitor.record_event_latency(50.0)
        monitor.reset()
        assert len(monitor._latencies) == 0

    def test_clears_order_counts(self, monitor):
        monitor.record_order_result(filled=True)
        monitor.reset()
        assert monitor._orders_total == 0
        assert monitor._orders_filled == 0

    def test_resets_feed_status(self, monitor):
        monitor.set_data_feed_status(False)
        monitor.reset()
        assert monitor._data_feed_connected is True

    def test_clears_started_at(self, monitor):
        monitor.reset()
        assert monitor._started_at is None


class TestHealthReport:
    def test_has_timestamp(self, monitor):
        monitor.record_bar(datetime.now())
        report = monitor.get_health_report()
        assert isinstance(report.timestamp, datetime)

    def test_uptime_seconds(self):
        m = HealthMonitor(max_bar_age_seconds=9999.0)
        m._started_at = datetime.now() - timedelta(seconds=60)
        m.record_bar(datetime.now())
        report = m.get_health_report()
        assert 59.0 < report.uptime_seconds < 62.0

    def test_data_feed_connected_field(self, monitor):
        monitor.record_bar(datetime.now())
        report = monitor.get_health_report()
        assert report.data_feed_connected is True

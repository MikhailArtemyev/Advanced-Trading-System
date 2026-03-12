"""HealthMonitor — real-time health tracking for paper trading.

Monitors bar freshness, event latency, order fill rates, and
data feed status. Generates HealthReports with HEALTHY / DEGRADED /
UNHEALTHY status and fires alert callbacks on non-healthy state.
"""

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Engine health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthReport:
    """Snapshot of engine health metrics.

    Attributes:
        status: Overall health (HEALTHY / DEGRADED / UNHEALTHY).
        timestamp: When the report was generated.
        uptime_seconds: Seconds since monitoring started.
        data_feed_connected: Whether the live data feed is up.
        last_bar_age_seconds: Seconds since the most recent bar.
        event_latency_ms: Average recent event processing latency.
        bars_per_minute: Bar throughput rate.
        fill_rate: Fraction of orders filled (0.0–1.0).
        warnings: Warning messages for degraded conditions.
        errors: Error messages for unhealthy conditions.
    """

    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    data_feed_connected: bool
    last_bar_age_seconds: float
    event_latency_ms: float
    bars_per_minute: float
    fill_rate: float
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class HealthMonitor:
    """Tracks engine health metrics and generates reports.

    Args:
        max_bar_age_seconds: Max seconds without a bar before DEGRADED.
        max_event_latency_ms: Max event latency before DEGRADED.
        history_window: Number of recent latency measurements to keep.
    """

    def __init__(
        self,
        max_bar_age_seconds: float = 120.0,
        max_event_latency_ms: float = 500.0,
        history_window: int = 100,
    ) -> None:
        self._max_bar_age_seconds = max_bar_age_seconds
        self._max_event_latency_ms = max_event_latency_ms
        self._history_window = history_window

        self._started_at: datetime | None = None
        self._last_bar_time: datetime | None = None
        self._bar_count = 0
        self._data_feed_connected = True

        self._latencies: deque[float] = deque(maxlen=history_window)
        self._orders_filled = 0
        self._orders_total = 0

        self._alert_callbacks: list[Callable[[HealthReport], None]] = []

    def start(self) -> None:
        """Mark the start of monitoring."""
        self._started_at = datetime.now()
        logger.info("Health monitoring started")

    def record_bar(self, timestamp: datetime) -> None:
        """Record that a bar was processed.

        Args:
            timestamp: The bar's timestamp.
        """
        self._last_bar_time = timestamp
        self._bar_count += 1

    def record_event_latency(self, latency_ms: float) -> None:
        """Record an event processing latency measurement.

        Args:
            latency_ms: Processing time in milliseconds.
        """
        self._latencies.append(latency_ms)

    def record_order_result(self, filled: bool) -> None:
        """Record whether an order was filled or rejected.

        Args:
            filled: True if filled, False if rejected.
        """
        self._orders_total += 1
        if filled:
            self._orders_filled += 1

    def set_data_feed_status(self, connected: bool) -> None:
        """Update the data feed connection status.

        Args:
            connected: Whether the feed is connected.
        """
        self._data_feed_connected = connected

    def add_alert_callback(self, callback: Callable[[HealthReport], None]) -> None:
        """Register a callback fired when health is not HEALTHY.

        Args:
            callback: Function receiving the HealthReport.
        """
        self._alert_callbacks.append(callback)

    def get_health_report(self) -> HealthReport:
        """Generate a current health report.

        Status logic:
        - UNHEALTHY if data feed disconnected.
        - DEGRADED if bar age or latency exceeds thresholds.
        - HEALTHY otherwise.

        Returns:
            HealthReport with current metrics.
        """
        now = datetime.now()
        warnings: list[str] = []
        errors: list[str] = []

        # Uptime
        uptime = (now - self._started_at).total_seconds() if self._started_at else 0.0

        # Bar age
        if self._last_bar_time is not None:
            bar_age = (now - self._last_bar_time).total_seconds()
        else:
            bar_age = uptime  # no bars yet — age equals uptime

        # Average latency
        avg_latency = (
            sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
        )

        # Bars per minute
        if uptime > 0:
            bars_per_minute = self._bar_count / (uptime / 60.0)
        else:
            bars_per_minute = 0.0

        # Fill rate
        fill_rate = (
            self._orders_filled / self._orders_total if self._orders_total > 0 else 1.0
        )

        # Determine status
        status = HealthStatus.HEALTHY

        if not self._data_feed_connected:
            status = HealthStatus.UNHEALTHY
            errors.append("Data feed disconnected")

        if bar_age > self._max_bar_age_seconds:
            if status != HealthStatus.UNHEALTHY:
                status = HealthStatus.DEGRADED
            warnings.append(
                f"Last bar age {bar_age:.1f}s exceeds "
                f"threshold {self._max_bar_age_seconds:.1f}s"
            )

        if avg_latency > self._max_event_latency_ms:
            if status != HealthStatus.UNHEALTHY:
                status = HealthStatus.DEGRADED
            warnings.append(
                f"Event latency {avg_latency:.1f}ms exceeds "
                f"threshold {self._max_event_latency_ms:.1f}ms"
            )

        report = HealthReport(
            status=status,
            timestamp=now,
            uptime_seconds=uptime,
            data_feed_connected=self._data_feed_connected,
            last_bar_age_seconds=bar_age,
            event_latency_ms=avg_latency,
            bars_per_minute=bars_per_minute,
            fill_rate=fill_rate,
            warnings=warnings,
            errors=errors,
        )

        # Fire alert callbacks for non-healthy status
        if status != HealthStatus.HEALTHY:
            for cb in self._alert_callbacks:
                try:
                    cb(report)
                except Exception:
                    logger.exception("Alert callback failed")

        return report

    def reset(self) -> None:
        """Reset all monitoring state."""
        self._started_at = None
        self._last_bar_time = None
        self._bar_count = 0
        self._data_feed_connected = True
        self._latencies.clear()
        self._orders_filled = 0
        self._orders_total = 0
        logger.info("Health monitor reset")

"""AlertPublisher — pub/sub alert system.

Publishers emit AlertMessages. Subscribers receive them filtered by
minimum severity level and deduplicated by cooldown.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from src.monitoring.health import HealthReport

from .base_alert import AlertLevel, AlertMessage, AlertSubscriber

logger = logging.getLogger(__name__)

_LEVEL_ORDER: dict[AlertLevel, int] = {
    AlertLevel.INFO: 0,
    AlertLevel.WARNING: 1,
    AlertLevel.CRITICAL: 2,
}


class AlertPublisher:
    """Central alert bus — subscribers register, components publish.

    Args:
        cooldown_seconds: Minimum interval between repeated alerts with
            the same title. Set to 0 to disable cooldown.
    """

    def __init__(self, cooldown_seconds: float = 300.0) -> None:
        self._subscribers: list[tuple[AlertSubscriber, AlertLevel]] = []
        self._cooldown_seconds = cooldown_seconds
        self._last_sent: dict[str, datetime] = {}

    def subscribe(
        self,
        subscriber: AlertSubscriber,
        min_level: AlertLevel = AlertLevel.WARNING,
    ) -> None:
        """Register a subscriber to receive alerts at or above min_level."""
        self._subscribers.append((subscriber, min_level))

    def unsubscribe(self, subscriber: AlertSubscriber) -> None:
        """Remove a subscriber."""
        self._subscribers = [
            (s, lvl) for s, lvl in self._subscribers if s is not subscriber
        ]

    async def publish(self, message: AlertMessage) -> list[bool]:
        """Publish a message to all eligible subscribers.

        Respects per-subscriber level filtering and global cooldown.

        Returns:
            A list of booleans, one per notified subscriber.
        """
        if self._is_in_cooldown(message.title):
            logger.debug("Alert '%s' suppressed by cooldown", message.title)
            return []

        msg_order = _LEVEL_ORDER[message.level]
        results: list[bool] = []
        for subscriber, min_level in self._subscribers:
            if msg_order < _LEVEL_ORDER[min_level]:
                continue
            try:
                ok = await subscriber.on_alert(message)
                results.append(ok)
            except Exception:
                logger.exception("Subscriber %s failed", type(subscriber).__name__)
                results.append(False)

        if results:
            self._last_sent[message.title] = datetime.now()
        return results

    # ------------------------------------------------------------------
    # Convenience publishers
    # ------------------------------------------------------------------

    async def publish_trade(self, trade: dict[str, Any]) -> list[bool]:
        """Publish a trade notification."""
        symbol = trade.get("symbol", "?")
        side = trade.get("side", "?")
        qty = trade.get("quantity", 0)
        price = trade.get("price", 0.0)
        pnl = trade.get("pnl", 0.0)

        msg = AlertMessage(
            level=AlertLevel.INFO,
            title=f"Trade: {side} {qty} {symbol}",
            body=f"Filled at ${price:,.2f} | PnL: ${pnl:,.2f}",
            metadata={"symbol": symbol, "side": side, "quantity": qty},
        )
        return await self.publish(msg)

    async def publish_health(self, report: HealthReport) -> list[bool]:
        """Publish a health-status notification."""
        level = AlertLevel.CRITICAL if report.errors else AlertLevel.WARNING
        msg = AlertMessage(
            level=level,
            title=f"Health: {report.status.value}",
            body=self._format_health_body(report),
            metadata={
                "uptime_s": f"{report.uptime_seconds:.0f}",
                "bar_age_s": f"{report.last_bar_age_seconds:.1f}",
                "latency_ms": f"{report.event_latency_ms:.1f}",
            },
        )
        return await self.publish(msg)

    async def publish_drawdown(self, drawdown_pct: float) -> list[bool]:
        """Publish a drawdown alert."""
        level = AlertLevel.CRITICAL if drawdown_pct >= 10.0 else AlertLevel.WARNING
        msg = AlertMessage(
            level=level,
            title=f"Drawdown: {drawdown_pct:.1f}%",
            body=f"Portfolio drawdown has reached {drawdown_pct:.1f}%.",
            metadata={"drawdown_pct": f"{drawdown_pct:.2f}"},
        )
        return await self.publish(msg)

    # ------------------------------------------------------------------
    # HealthMonitor callback adapter
    # ------------------------------------------------------------------

    def on_health_report(self, report: HealthReport) -> None:
        """Synchronous callback for HealthMonitor.add_alert_callback.

        Fires-and-forgets the async publish via the running event loop.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish_health(report))
        except RuntimeError:
            logger.warning("No event loop for health alert: %s", report.status.value)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_in_cooldown(self, title: str) -> bool:
        if self._cooldown_seconds <= 0:
            return False
        last = self._last_sent.get(title)
        if last is None:
            return False
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed < self._cooldown_seconds

    @staticmethod
    def _format_health_body(report: HealthReport) -> str:
        lines = [f"Status: {report.status.value}"]
        if report.warnings:
            lines.append("Warnings:")
            for w in report.warnings:
                lines.append(f"  - {w}")
        if report.errors:
            lines.append("Errors:")
            for e in report.errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)

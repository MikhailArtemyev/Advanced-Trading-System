"""AlertManager — routes alerts to channels with filtering and cooldown.

Integrates with HealthMonitor callbacks and can be wired into the
PaperTradingEngine for trade and drawdown notifications.
"""

import logging
from datetime import datetime
from typing import Any

from src.monitoring.health import HealthReport

from .base_alert import AlertChannel, AlertLevel, AlertMessage

logger = logging.getLogger(__name__)

# Ordered severity so we can compare levels.
_LEVEL_ORDER: dict[AlertLevel, int] = {
    AlertLevel.INFO: 0,
    AlertLevel.WARNING: 1,
    AlertLevel.CRITICAL: 2,
}


class AlertManager:
    """Dispatch alerts to multiple channels with level filtering and cooldown.

    Args:
        channels: List of alert backends.
        min_level: Minimum severity to send. Alerts below this are dropped.
        cooldown_seconds: Minimum interval between repeated alerts with the
            same title.  Set to 0 to disable cooldown.
    """

    def __init__(
        self,
        channels: list[AlertChannel],
        min_level: AlertLevel = AlertLevel.WARNING,
        cooldown_seconds: float = 300.0,
    ) -> None:
        self._channels = list(channels)
        self._min_level = min_level
        self._cooldown_seconds = cooldown_seconds
        self._last_sent: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    async def send_alert(self, message: AlertMessage) -> list[bool]:
        """Send *message* to all channels, respecting level and cooldown.

        Returns:
            A list of booleans, one per channel, indicating success/failure.
        """
        if _LEVEL_ORDER[message.level] < _LEVEL_ORDER[self._min_level]:
            return []

        if self._is_in_cooldown(message.title):
            logger.debug("Alert '%s' suppressed by cooldown", message.title)
            return []

        results: list[bool] = []
        for channel in self._channels:
            try:
                ok = await channel.send(message)
                results.append(ok)
            except Exception:
                logger.exception("Channel %s failed", type(channel).__name__)
                results.append(False)

        self._last_sent[message.title] = datetime.now()
        return results

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    async def send_trade_alert(self, trade: dict[str, Any]) -> list[bool]:
        """Format and send a trade notification."""
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
        return await self.send_alert(msg)

    async def send_health_alert(self, report: HealthReport) -> list[bool]:
        """Format and send a health-status notification."""
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
        return await self.send_alert(msg)

    async def send_drawdown_alert(self, drawdown_pct: float) -> list[bool]:
        """Alert on a significant drawdown percentage."""
        level = AlertLevel.CRITICAL if drawdown_pct >= 10.0 else AlertLevel.WARNING
        msg = AlertMessage(
            level=level,
            title=f"Drawdown: {drawdown_pct:.1f}%",
            body=f"Portfolio drawdown has reached {drawdown_pct:.1f}%.",
            metadata={"drawdown_pct": f"{drawdown_pct:.2f}"},
        )
        return await self.send_alert(msg)

    # ------------------------------------------------------------------
    # HealthMonitor callback adapter
    # ------------------------------------------------------------------

    def on_health_report(self, report: HealthReport) -> None:
        """Synchronous callback suitable for HealthMonitor.add_alert_callback.

        Fires-and-forgets the async send via the running event loop.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.send_health_alert(report))
        except RuntimeError:
            # No running loop — just log
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

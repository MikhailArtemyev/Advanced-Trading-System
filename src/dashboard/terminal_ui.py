"""Rich-based terminal dashboard for paper trading sessions.

Displays live-updating panels for equity, positions, recent trades,
health status, and engine statistics.
"""

import asyncio
import logging
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.engine.paper_engine import PaperTradingEngine
from src.monitoring.health import HealthMonitor, HealthStatus

logger = logging.getLogger(__name__)

_STATUS_STYLE: dict[HealthStatus, str] = {
    HealthStatus.HEALTHY: "bold green",
    HealthStatus.DEGRADED: "bold yellow",
    HealthStatus.UNHEALTHY: "bold red",
}


class TradingDashboard:
    """Real-time terminal dashboard for a paper trading session.

    Args:
        engine: The running PaperTradingEngine.
        health_monitor: HealthMonitor for status metrics.
        refresh_rate: Seconds between screen refreshes.
    """

    def __init__(
        self,
        engine: PaperTradingEngine,
        health_monitor: HealthMonitor,
        refresh_rate: float = 1.0,
    ) -> None:
        self._engine = engine
        self._health_monitor = health_monitor
        self._refresh_rate = refresh_rate
        self._running = False
        self._console = Console()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the dashboard until the engine stops or ``stop()`` is called.

        Redirects root logger output through Rich so log lines don't
        corrupt the live display. Original handlers are restored on exit.
        """
        self._running = True

        # Redirect all logging through Rich's console so output doesn't
        # collide with the Live display.
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        rich_handler = RichHandler(
            console=self._console,
            show_path=False,
            show_time=True,
        )
        root.handlers = [rich_handler]

        logger.info("Dashboard started (refresh=%.1fs)", self._refresh_rate)

        try:
            with Live(
                self._build_layout(),
                console=self._console,
                refresh_per_second=1.0 / self._refresh_rate,
                screen=True,
            ) as live:
                # Wait up to 30s for the engine to actually start
                for _ in range(300):
                    if self._engine.is_running:
                        break
                    if not self._running:
                        break
                    await asyncio.sleep(0.1)

                # Keep refreshing while running. When the engine stops
                # the display freezes on the final state until stop()
                # is called (e.g. by Ctrl-C / shutdown handler).
                while self._running:
                    live.update(self._build_layout())
                    await asyncio.sleep(self._refresh_rate)
        except Exception:
            logger.exception("Dashboard error")
        finally:
            self._running = False
            root.handlers = original_handlers
            logger.info("Dashboard stopped")

    def stop(self) -> None:
        """Signal the dashboard to stop."""
        self._running = False

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> Layout:
        """Compose the full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        layout["left"].split_column(
            Layout(name="equity", ratio=1),
            Layout(name="positions", ratio=2),
        )

        layout["right"].split_column(
            Layout(name="trades", ratio=2),
            Layout(name="health", ratio=1),
        )

        stats = self._engine.get_statistics()
        health_report = self._health_monitor.get_health_report()

        layout["header"].update(self._build_header(stats))
        layout["equity"].update(self._build_equity_panel(stats))
        layout["positions"].update(self._build_positions_panel(stats))
        layout["trades"].update(self._build_trades_panel())
        layout["health"].update(self._build_health_panel(health_report))
        layout["footer"].update(self._build_stats_panel(stats))

        return layout

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------

    def _build_header(self, stats: dict[str, Any]) -> Panel:
        """Session header with strategy and runtime info."""
        strategy_name = type(self._engine._strategy).__name__
        symbols = ", ".join(self._engine._strategy.symbols)
        uptime = stats.get("uptime_seconds", 0.0)
        mins, secs = divmod(int(uptime), 60)
        hrs, mins = divmod(mins, 60)

        text = Text()
        text.append("PAPER TRADING", style="bold cyan")
        text.append(f"  |  {strategy_name}")
        text.append(f"  |  {symbols}")
        text.append(f"  |  Uptime: {hrs:02d}:{mins:02d}:{secs:02d}")

        return Panel(text, style="blue")

    def _build_equity_panel(self, stats: dict[str, Any]) -> Panel:
        """Equity, P&L, and return percentage."""
        equity = stats.get("equity", 0.0)
        initial = self._engine._portfolio.initial_capital
        pnl = equity - initial
        ret_pct = (pnl / initial * 100) if initial > 0 else 0.0
        cash = self._engine._portfolio.cash

        pnl_style = "green" if pnl >= 0 else "red"

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("label", style="dim")
        table.add_column("value", justify="right")
        table.add_row("Equity", f"${equity:,.2f}")
        table.add_row("Cash", f"${cash:,.2f}")
        table.add_row("P&L", Text(f"${pnl:+,.2f}", style=pnl_style))
        table.add_row("Return", Text(f"{ret_pct:+.2f}%", style=pnl_style))

        return Panel(table, title="Equity", border_style="green")

    def _build_positions_panel(self, stats: dict[str, Any]) -> Panel:
        """Current positions table."""
        positions = stats.get("positions", {})

        table = Table(box=None, padding=(0, 1))
        table.add_column("Symbol", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("Avg Cost", justify="right")
        table.add_column("Mkt Value", justify="right")
        table.add_column("Unreal P&L", justify="right")

        for symbol, pos in sorted(positions.items()):
            if isinstance(pos, dict):
                qty = pos.get("quantity", 0)
                if qty == 0:
                    continue
                avg = pos.get("avg_cost", 0.0)
                mv = pos.get("market_value", 0.0)
                upnl = pos.get("unrealized_pnl", 0.0)
                pnl_style = "green" if upnl >= 0 else "red"
                table.add_row(
                    symbol,
                    str(qty),
                    f"${avg:,.2f}",
                    f"${mv:,.2f}",
                    Text(f"${upnl:+,.2f}", style=pnl_style),
                )

        if table.row_count == 0:
            table.add_row("—", "No open positions", "", "", "")

        return Panel(table, title="Positions", border_style="cyan")

    def _build_trades_panel(self) -> Panel:
        """Last 10 trades."""
        trades = self._engine._portfolio.get_trade_history()
        recent = trades[-10:] if trades else []

        table = Table(box=None, padding=(0, 1))
        table.add_column("Time", style="dim")
        table.add_column("Symbol", style="cyan")
        table.add_column("Side")
        table.add_column("Qty", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("P&L", justify="right")

        for trade in reversed(recent):
            ts = str(getattr(trade, "timestamp", ""))[:19]
            sym = getattr(trade, "symbol", "?")
            side = getattr(trade, "side", "?")
            qty = getattr(trade, "quantity", 0)
            price = getattr(trade, "price", 0.0)
            pnl = getattr(trade, "pnl", 0.0)
            pnl_style = "green" if pnl >= 0 else "red"
            side_style = "green" if side == "BUY" else "red"
            table.add_row(
                ts,
                sym,
                Text(side, style=side_style),
                str(qty),
                f"${price:,.2f}",
                Text(f"${pnl:+,.2f}", style=pnl_style),
            )

        if table.row_count == 0:
            table.add_row("", "No trades yet", "", "", "", "")

        return Panel(table, title="Recent Trades", border_style="yellow")

    def _build_health_panel(self, report: Any) -> Panel:
        """Health status with key metrics."""
        status = report.status
        style = _STATUS_STYLE.get(status, "")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("label", style="dim")
        table.add_column("value", justify="right")
        table.add_row("Status", Text(status.value.upper(), style=style))
        table.add_row("Feed", "Connected" if report.data_feed_connected else "DOWN")
        table.add_row("Bar Age", f"{report.last_bar_age_seconds:.1f}s")
        table.add_row("Latency", f"{report.event_latency_ms:.1f}ms")
        table.add_row("Bars/min", f"{report.bars_per_minute:.1f}")
        table.add_row("Fill Rate", f"{report.fill_rate * 100:.0f}%")

        if report.warnings:
            for w in report.warnings:
                table.add_row("Warning", Text(w, style="yellow"))
        if report.errors:
            for e in report.errors:
                table.add_row("Error", Text(e, style="red"))

        return Panel(table, title="Health", border_style=style or "white")

    def _build_stats_panel(self, stats: dict[str, Any]) -> Panel:
        """Engine statistics footer."""
        parts = [
            f"Bars: {stats.get('bars_processed', 0)}",
            f"Events: {stats.get('events_processed', 0)}",
            f"Orders: {stats.get('orders_submitted', 0)}",
            f"Rejected: {stats.get('orders_rejected', 0)}",
            f"Fills: {stats.get('fills_processed', 0)}",
        ]
        text = Text(" | ".join(parts), style="dim")
        return Panel(text, style="blue")

#!/usr/bin/env python3
"""Paper Trading Script — run a live paper trading session.

Wires together LiveDataHandler, PaperBroker, OrderManager,
PaperTradingEngine, SQLStorage, and HealthMonitor.

Supports graceful shutdown (Ctrl+C), periodic state saves to SQLite,
and optional state recovery from the database.

Usage:
    python scripts/run_paper_trading.py --config configs/paper_trading_config.yaml
"""

import argparse
import asyncio
import logging
import math
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts.alert_manager import AlertManager
from src.alerts.base_alert import AlertChannel, AlertLevel
from src.alerts.email_alert import EmailAlert
from src.alerts.slack_alert import SlackAlert
from src.alerts.webhook_alert import WebhookAlert
from src.broker.alpaca_broker import AlpacaBroker
from src.broker.order_manager import OrderManager
from src.broker.paper_broker import PaperBroker
from src.config import BacktestConfig, load_config
from src.dashboard.terminal_ui import TradingDashboard
from src.engine.paper_engine import PaperTradingEngine
from src.live.bar_aggregator import BarAggregator, Tick
from src.live.data_handler import LiveDataHandler
from src.monitoring.health import HealthMonitor
from src.portfolio.portfolio import Portfolio
from src.risk.position_sizer import FixedFractionSizer
from src.risk.risk_manager import RiskLimits, RiskManager
from src.storage.sql_storage import SQLStorage
from src.strategy.sma_crossover import SMACrossoverStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paper_trading")


def build_alert_channels(config: BacktestConfig) -> list[AlertChannel]:
    """Build alert channels from config."""
    channels: list[AlertChannel] = []
    alert_cfg = config.live.alerts
    if not alert_cfg.enabled:
        return channels

    for ch_cfg in alert_cfg.channels:
        ch_type = ch_cfg.get("type", "")
        if ch_type == "slack":
            channels.append(
                SlackAlert(
                    webhook_url=ch_cfg.get("webhook_url", ""),
                    channel=ch_cfg.get("channel", ""),
                )
            )
        elif ch_type == "email":
            channels.append(
                EmailAlert(
                    smtp_host=ch_cfg.get("smtp_host", ""),
                    smtp_port=ch_cfg.get("smtp_port", 587),
                    username=ch_cfg.get("username", ""),
                    password=ch_cfg.get("password", ""),
                    from_address=ch_cfg.get("from_address", ""),
                    to_addresses=ch_cfg.get("to_addresses", []),
                )
            )
        elif ch_type == "webhook":
            channels.append(
                WebhookAlert(
                    url=ch_cfg.get("url", ""),
                    headers=ch_cfg.get("headers", {}),
                    method=ch_cfg.get("method", "POST"),
                )
            )
        else:
            logger.warning("Unknown alert channel type: %s", ch_type)

    return channels


def build_components(
    config: BacktestConfig,
) -> tuple[
    LiveDataHandler,
    SMACrossoverStrategy,
    Portfolio,
    OrderManager,
    RiskManager | None,
    SQLStorage,
    HealthMonitor,
    BarAggregator,
    AlertManager | None,
]:
    """Build all paper trading components from config."""
    symbols = config.data.symbols
    live_cfg = config.live

    # Bar aggregator
    interval = timedelta(seconds=live_cfg.data.bar_interval_seconds)
    aggregator = BarAggregator(interval=interval)

    # Data handler
    data_handler = LiveDataHandler(
        symbols=symbols,
        bar_aggregator=aggregator,
        max_history=live_cfg.data.max_history,
    )

    # Strategy
    params = dict(config.strategy.parameters)
    strategy = SMACrossoverStrategy(symbols=symbols, parameters=params)

    # Position sizer
    sizer_params = config.sizing.parameters
    sizer = FixedFractionSizer(fraction=sizer_params.get("fraction", 0.05))

    # Storage
    storage = SQLStorage(db_url=live_cfg.database.db_url)
    session_id = storage.create_session(
        session_type="paper",
        config=config.model_dump(),
        initial_capital=config.execution.initial_capital,
    )

    # Portfolio
    portfolio = Portfolio(
        initial_capital=config.execution.initial_capital,
        symbols=symbols,
        commission_pct=config.execution.commission_pct,
        position_sizer=sizer,
        storage=storage,
        session_id=session_id,
    )

    # Broker — select based on config
    broker_type = live_cfg.broker.broker_type
    if broker_type == "alpaca":
        api_key = live_cfg.broker.api_key
        api_secret = live_cfg.broker.api_secret
        if not api_key or not api_secret:
            logger.error(
                "Alpaca broker requires api_key and api_secret in config "
                "or ALPACA_API_KEY/ALPACA_API_SECRET env vars"
            )
            sys.exit(1)
        broker = AlpacaBroker(
            api_key=api_key,
            api_secret=api_secret,
            base_url=live_cfg.broker.base_url,
            paper_mode=live_cfg.broker.paper_mode,
        )
    else:
        broker = PaperBroker(
            initial_capital=config.execution.initial_capital,
            commission_pct=config.execution.commission_pct,
            slippage_pct=config.execution.slippage_pct,
            fill_delay_seconds=live_cfg.broker.fill_delay_ms / 1000.0,
        )
        broker.set_data_handler(data_handler)

    # Order manager
    order_manager = OrderManager(broker=broker)

    # Risk manager
    risk_manager: RiskManager | None = None
    if config.risk.enabled:
        limits = RiskLimits(
            max_position_pct=config.risk.max_position_pct,
            max_portfolio_exposure_pct=config.risk.max_portfolio_exposure_pct,
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            max_drawdown_pct=config.risk.max_drawdown_pct,
            max_open_positions=config.risk.max_open_positions,
            max_orders_per_day=config.risk.max_orders_per_day,
        )
        risk_manager = RiskManager(limits=limits)

    # Health monitor
    health_monitor = HealthMonitor(
        max_bar_age_seconds=live_cfg.data.bar_interval_seconds * 2.0,
        max_event_latency_ms=500.0,
    )

    # Alert manager
    alert_channels = build_alert_channels(config)
    alert_manager: AlertManager | None = None
    if alert_channels:
        level_map = {
            "info": AlertLevel.INFO,
            "warning": AlertLevel.WARNING,
            "critical": AlertLevel.CRITICAL,
        }
        alert_manager = AlertManager(
            channels=alert_channels,
            min_level=level_map.get(live_cfg.alerts.min_level, AlertLevel.WARNING),
            cooldown_seconds=live_cfg.alerts.cooldown_seconds,
        )

    return (
        data_handler,
        strategy,
        portfolio,
        order_manager,
        risk_manager,
        storage,
        health_monitor,
        aggregator,
        alert_manager,
    )


def generate_synthetic_ticks(
    symbols: list[str],
    duration_minutes: int = 30,
    bar_interval_seconds: int = 60,
    ticks_per_second: float = 2.0,
) -> list[Tick]:
    """Generate synthetic ticks for demo paper trading.

    Creates realistic-looking price movements with a sine wave trend
    so SMA crossover strategies can generate signals.
    """
    ticks: list[Tick] = []
    base_prices = {"AAPL": 175.0, "MSFT": 380.0, "GOOGL": 140.0}
    total_seconds = duration_minutes * 60
    dt = 1.0 / ticks_per_second
    start = datetime.now()

    for i in range(int(total_seconds * ticks_per_second)):
        t = i * dt
        timestamp = start + timedelta(seconds=t)
        for symbol in symbols:
            base = base_prices.get(symbol, 100.0)
            # Sine wave for crossover + small noise
            trend = base * 0.03 * math.sin(2 * math.pi * t / (5 * 60))
            noise = base * 0.001 * math.sin(t * 17.3 + hash(symbol) % 100)
            price = base + trend + noise
            ticks.append(
                Tick(symbol=symbol, price=price, timestamp=timestamp, volume=100)
            )

    return ticks


async def run_paper_trading(config: BacktestConfig, dashboard: bool = False) -> None:
    """Run a paper trading session."""
    (
        data_handler,
        strategy,
        portfolio,
        order_manager,
        risk_manager,
        storage,
        health_monitor,
        aggregator,
        alert_manager,
    ) = build_components(config)

    # Session ID is already set on storage — extract from portfolio
    session_id = portfolio.session_id

    # Build engine
    engine = PaperTradingEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        order_manager=order_manager,
        risk_manager=risk_manager,
        storage=storage,
        session_id=session_id,
    )

    # Try to restore from database
    if engine.restore_from_storage():
        logger.info("Restored state from database for session %s", session_id)
    else:
        logger.info("No previous state found — starting fresh")

    # Health monitor setup
    health_monitor.start()

    def on_health_alert(report: "HealthMonitor") -> None:
        logger.warning(
            "Health alert: %s | warnings=%s errors=%s",
            report.status.value,
            report.warnings,
            report.errors,
        )

    health_monitor.add_alert_callback(on_health_alert)

    # Wire alert manager into health monitor
    if alert_manager is not None:
        health_monitor.add_alert_callback(alert_manager.on_health_report)

    # Graceful shutdown
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    # Generate ticks for demo
    symbols = config.data.symbols
    live_cfg = config.live
    ticks = generate_synthetic_ticks(
        symbols=symbols,
        duration_minutes=10,
        bar_interval_seconds=live_cfg.data.bar_interval_seconds,
    )

    save_interval = live_cfg.database.save_interval_seconds

    async def feed_ticks() -> None:
        """Feed synthetic ticks to the aggregator."""
        bar_interval = live_cfg.data.bar_interval_seconds
        ticks_per_bar = int(bar_interval * 2)  # 2 ticks/sec
        batch_size = max(1, ticks_per_bar)

        # When dashboard is active, pace ticks so bars arrive every 1s
        # instead of every 50ms — otherwise the session ends too fast to see.
        tick_delay = 1.0 if dashboard else 0.05

        for i in range(0, len(ticks), batch_size):
            if stop_event.is_set():
                break
            batch = ticks[i : i + batch_size]
            for tick in batch:
                aggregator.on_tick(tick)

            # Record bar for health monitor
            health_monitor.record_bar(datetime.now())

            await asyncio.sleep(tick_delay)

        # Let engine process remaining events
        await asyncio.sleep(0.5)
        await engine.stop()
        stop_event.set()
        if dash is not None:
            dash.stop()

    async def periodic_save() -> None:
        """Periodically save engine state to database."""
        while not stop_event.is_set():
            await asyncio.sleep(min(save_interval, 5))  # faster for demo
            if engine.is_running:
                state = engine.get_state()
                storage.save_engine_state(session_id, state)
                logger.info("Engine state saved to database")

    async def monitor_health() -> None:
        """Periodically check health (skipped when dashboard is active)."""
        while not stop_event.is_set():
            await asyncio.sleep(5)
            report = health_monitor.get_health_report()
            logger.info(
                "Health: %s | bars/min=%.1f latency=%.1fms fill_rate=%.0f%%",
                report.status.value,
                report.bars_per_minute,
                report.event_latency_ms,
                report.fill_rate * 100,
            )

    print("\n" + "=" * 60)
    print("  PAPER TRADING SESSION")
    print("=" * 60)
    print(f"  Symbols:  {symbols}")
    print(f"  Strategy: {config.strategy.name} {dict(config.strategy.parameters)}")
    print(f"  Capital:  ${config.execution.initial_capital:,.2f}")
    print(f"  Broker:   {live_cfg.broker.broker_type}")
    print(f"  Storage:  {live_cfg.database.db_url}")
    print("=" * 60 + "\n")

    # Optional dashboard
    dash: TradingDashboard | None = None
    if dashboard:
        dash = TradingDashboard(engine, health_monitor, refresh_rate=1.0)

    # Graceful shutdown — registered after dashboard so we can stop it
    def handle_signal() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()
        if dash is not None:
            dash.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    # Connect broker
    await order_manager._broker.connect()

    tasks = [
        engine.start(),
        feed_ticks(),
        periodic_save(),
    ]
    # Dashboard replaces the health log task — it shows health in a panel
    if dash is not None:
        tasks.append(dash.start())
    else:
        tasks.append(monitor_health())

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        # Final state save
        state = engine.get_state()
        storage.save_engine_state(session_id, state)
        logger.info("Final state saved to database")

        # End session
        storage.end_session(session_id, portfolio.get_equity())

        # Reconciliation check
        report = await engine.reconcile_positions()
        if report["matched"]:
            logger.info("Reconciliation: all positions matched")
        else:
            logger.warning("Reconciliation discrepancies: %s", report["discrepancies"])

        # Final health report
        health_report = health_monitor.get_health_report()

        # Disconnect
        await order_manager._broker.disconnect()

        # Close storage
        storage.close()

        # Print summary
        stats = engine.get_statistics()
        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Session ID:        {session_id}")
        print(f"  Bars processed:    {stats['bars_processed']}")
        print(f"  Orders submitted:  {stats['orders_submitted']}")
        print(f"  Orders rejected:   {stats['orders_rejected']}")
        print(f"  Fills processed:   {stats['fills_processed']}")
        print(f"  Final equity:      ${stats['equity']:,.2f}")
        print(f"  Health status:     {health_report.status.value}")
        print("=" * 60 + "\n")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run paper trading session")
    parser.add_argument(
        "--config",
        default="configs/paper_trading_config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable Rich terminal dashboard",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Startup validation
    issues = config.validate_for_live()
    if issues:
        for issue in issues:
            if issue.startswith("ERROR:"):
                logger.error(issue)
            else:
                logger.warning(issue)
        errors = [i for i in issues if i.startswith("ERROR:")]
        if errors:
            logger.error(
                "Configuration validation failed with %d error(s)", len(errors)
            )
            sys.exit(1)

    asyncio.run(run_paper_trading(config, dashboard=args.dashboard))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper Trading Script — run a live paper trading session.

Wires together LiveDataHandler, PaperBroker, OrderManager,
PaperTradingEngine, StateManager, and HealthMonitor.

Supports graceful shutdown (Ctrl+C), periodic state snapshots,
and optional state recovery from the latest snapshot.

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

from src.broker.alpaca_broker import AlpacaBroker
from src.broker.order_manager import OrderManager
from src.broker.paper_broker import PaperBroker
from src.config import BacktestConfig, load_config
from src.engine.paper_engine import PaperTradingEngine
from src.engine.state_manager import StateManager
from src.live.bar_aggregator import BarAggregator, Tick
from src.live.data_handler import LiveDataHandler
from src.monitoring.health import HealthMonitor
from src.portfolio.portfolio import Portfolio
from src.risk.position_sizer import FixedFractionSizer
from src.risk.risk_manager import RiskLimits, RiskManager
from src.strategy.sma_crossover import SMACrossoverStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paper_trading")


def build_components(
    config: BacktestConfig,
) -> tuple[
    LiveDataHandler,
    SMACrossoverStrategy,
    Portfolio,
    OrderManager,
    RiskManager | None,
    StateManager,
    HealthMonitor,
    BarAggregator,
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

    # Portfolio
    portfolio = Portfolio(
        initial_capital=config.execution.initial_capital,
        symbols=symbols,
        commission_pct=config.execution.commission_pct,
        position_sizer=sizer,
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
        broker._data_handler = data_handler

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

    # State manager
    state_manager = StateManager(
        state_dir=live_cfg.persistence.state_dir,
        max_snapshots=live_cfg.persistence.max_snapshots,
    )

    # Health monitor
    health_monitor = HealthMonitor(
        max_bar_age_seconds=live_cfg.data.bar_interval_seconds * 2.0,
        max_event_latency_ms=500.0,
    )

    return (
        data_handler,
        strategy,
        portfolio,
        order_manager,
        risk_manager,
        state_manager,
        health_monitor,
        aggregator,
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


async def run_paper_trading(config: BacktestConfig) -> None:
    """Run a paper trading session."""
    (
        data_handler,
        strategy,
        portfolio,
        order_manager,
        risk_manager,
        state_manager,
        health_monitor,
        aggregator,
    ) = build_components(config)

    # Build engine
    engine = PaperTradingEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        order_manager=order_manager,
        risk_manager=risk_manager,
    )

    # Try to restore from last snapshot
    snapshot = state_manager.load_latest_snapshot()
    if snapshot is not None:
        engine.restore_state(snapshot)
        logger.info(
            "Restored state from snapshot: %s", snapshot.get("_snapshot_timestamp")
        )
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

    # Graceful shutdown
    stop_event = asyncio.Event()

    def handle_signal() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    # Generate ticks for demo
    symbols = config.data.symbols
    live_cfg = config.live
    ticks = generate_synthetic_ticks(
        symbols=symbols,
        duration_minutes=10,
        bar_interval_seconds=live_cfg.data.bar_interval_seconds,
    )

    save_interval = live_cfg.persistence.save_interval_seconds

    async def feed_ticks() -> None:
        """Feed synthetic ticks to the aggregator."""
        bar_interval = live_cfg.data.bar_interval_seconds
        ticks_per_bar = int(bar_interval * 2)  # 2 ticks/sec
        batch_size = max(1, ticks_per_bar)

        for i in range(0, len(ticks), batch_size):
            if stop_event.is_set():
                break
            batch = ticks[i : i + batch_size]
            for tick in batch:
                aggregator.on_tick(tick)

            # Record bar for health monitor
            health_monitor.record_bar(datetime.now())

            await asyncio.sleep(0.05)  # fast simulation

        # Let engine process remaining events
        await asyncio.sleep(0.5)
        await engine.stop()

    async def periodic_save() -> None:
        """Periodically save state snapshots."""
        while not stop_event.is_set():
            await asyncio.sleep(min(save_interval, 5))  # faster for demo
            if engine.is_running:
                state = engine.get_state()
                state_manager.save_snapshot(state)
                logger.info("State snapshot saved")

    async def monitor_health() -> None:
        """Periodically check health."""
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
    print("=" * 60 + "\n")

    # Connect broker
    await order_manager._broker.connect()

    try:
        await asyncio.gather(
            engine.start(),
            feed_ticks(),
            periodic_save(),
            monitor_health(),
        )
    except asyncio.CancelledError:
        pass
    finally:
        # Final state save
        state = engine.get_state()
        state_manager.save_snapshot(state)
        logger.info("Final state saved")

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

        # Print summary
        stats = engine.get_statistics()
        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Bars processed:    {stats['bars_processed']}")
        print(f"  Orders submitted:  {stats['orders_submitted']}")
        print(f"  Orders rejected:   {stats['orders_rejected']}")
        print(f"  Fills processed:   {stats['fills_processed']}")
        print(f"  Final equity:      ${stats['equity']:,.2f}")
        print(f"  Health status:     {health_report.status.value}")
        print(f"  Snapshots saved:   {len(state_manager.list_snapshots())}")
        print("=" * 60 + "\n")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run paper trading session")
    parser.add_argument(
        "--config",
        default="configs/paper_trading_config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    asyncio.run(run_paper_trading(config))


if __name__ == "__main__":
    main()

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
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import STRATEGY_MAP, generate_synthetic_ticks
from src.alerts.alert_manager import AlertPublisher
from src.broker.alpaca_broker import AlpacaBroker
from src.broker.base_broker import BrokerFill, OrderStatus
from src.broker.order_manager import OrderManager
from src.broker.paper_broker import PaperBroker
from src.config import BacktestConfig, load_config
from src.engine.paper_engine import PaperTradingEngine
from src.live.alpaca_feed import AlpacaDataFeed
from src.live.alpaca_historical import AlpacaHistoricalClient, backfill_aggregator
from src.live.bar_aggregator import Bar, BarAggregator, Tick
from src.live.data_handler import LiveDataHandler
from src.monitoring.health import HealthMonitor
from src.portfolio.portfolio import Portfolio
from src.risk.position_sizer import FixedFractionSizer
from src.risk.risk_manager import RiskLimits, RiskManager
from src.storage.sql_storage import SQLStorage
from src.strategy.base_strategy import Strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paper_trading")


# ---------------------------------------------------------------------------
# Helper functions (extracted from run_paper_trading for readability)
# ---------------------------------------------------------------------------


def on_health_alert(report: Any) -> None:
    """Log health monitor alerts."""
    logger.warning(
        "Health alert: %s | warnings=%s errors=%s",
        report.status.value,
        report.warnings,
        report.errors,
    )


async def feed_synthetic_ticks(
    ticks: list[Tick],
    aggregator: BarAggregator,
    health_monitor: HealthMonitor,
    bar_interval_seconds: int,
    stop_event: asyncio.Event,
    engine: PaperTradingEngine,
) -> None:
    """Feed synthetic ticks to the aggregator (demo mode only)."""
    ticks_per_bar = int(bar_interval_seconds * 2)  # 2 ticks/sec
    batch_size = max(1, ticks_per_bar)
    tick_delay = 0.05

    for i in range(0, len(ticks), batch_size):
        if stop_event.is_set():
            break
        batch = ticks[i : i + batch_size]
        for tick in batch:
            aggregator.on_tick(tick)

        health_monitor.record_bar(datetime.now())
        await asyncio.sleep(tick_delay)

    # Let engine process remaining events
    await asyncio.sleep(0.5)
    await engine.stop()
    stop_event.set()


async def run_alpaca_feed(
    alpaca_feed: AlpacaDataFeed,
    symbols: list[str],
    stop_event: asyncio.Event,
) -> None:
    """Connect to Alpaca WebSocket and stream until stopped."""
    await alpaca_feed.connect()
    await alpaca_feed.subscribe(symbols)
    logger.info("Alpaca data feed streaming for %s", symbols)

    await stop_event.wait()

    await alpaca_feed.disconnect()


async def poll_order_fills(
    stop_event: asyncio.Event,
    order_manager: OrderManager,
    portfolio: Portfolio,
    engine: PaperTradingEngine,
) -> None:
    """Poll Alpaca for fill status on active orders.

    Alpaca REST API returns orders as 'new'/'accepted', not 'filled'.
    We must poll to detect when they fill, then emit FillEvents so the
    portfolio and strategy stay in sync.
    """
    while not stop_event.is_set():
        await asyncio.sleep(1.0)
        active = order_manager.active_orders
        if not active:
            continue
        for order in list(active):
            try:
                updated = await order_manager._broker.get_order_status(order.order_id)
            except ValueError:
                # Order not tracked by broker (rejected before storage)
                order.status = OrderStatus.REJECTED
                continue
            except Exception:
                logger.exception(
                    "Failed to poll order status for %s", order.order_id[:8]
                )
                continue

            if updated.status == OrderStatus.FILLED:
                fill = BrokerFill(
                    order_id=updated.order_id,
                    symbol=updated.symbol,
                    side=updated.side,
                    quantity=updated.filled_quantity,
                    price=updated.filled_avg_price,
                    commission=abs(
                        updated.filled_avg_price
                        * updated.filled_quantity
                        * portfolio.commission_pct
                    ),
                    timestamp=updated.filled_at or datetime.now(),
                )
                fill_event = order_manager.on_fill(fill)
                engine._events.put(fill_event)
            elif updated.status in (
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
            ):
                logger.warning(
                    "Order %s %s: %s",
                    updated.symbol,
                    updated.side,
                    updated.status.name,
                )


async def periodic_save(
    stop_event: asyncio.Event,
    save_interval: float,
    engine: PaperTradingEngine,
    storage: SQLStorage,
    session_id: str,
) -> None:
    """Periodically save engine state to database."""
    while not stop_event.is_set():
        await asyncio.sleep(save_interval)
        if engine.is_running:
            state = engine.get_state()
            storage.save_engine_state(session_id, state)
            logger.info("Engine state saved to database")


async def monitor_health(
    stop_event: asyncio.Event,
    engine: PaperTradingEngine,
    interval: float = 60.0,
) -> None:
    """Periodically log engine status."""
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        stats = engine.get_statistics()
        logger.info(
            "bars=%d orders=%d fills=%d equity=$%.2f",
            stats["bars_processed"],
            stats["orders_submitted"],
            stats["fills_processed"],
            stats["equity"],
        )


# ---------------------------------------------------------------------------
# Component builder
# ---------------------------------------------------------------------------


def build_components(
    config: BacktestConfig,
) -> tuple[
    LiveDataHandler,
    Strategy,
    Portfolio,
    OrderManager,
    RiskManager | None,
    SQLStorage,
    HealthMonitor,
    BarAggregator,
    AlertPublisher,
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
    strategy_name = config.strategy.name
    strategy_cls = STRATEGY_MAP.get(strategy_name)
    if strategy_cls is None:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available: {list(STRATEGY_MAP.keys())}"
        )
    params = dict(config.strategy.parameters)
    strategy = strategy_cls(symbols=symbols, parameters=params)

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

    # Alert publisher (subscribers are added externally)
    alert_publisher = AlertPublisher(
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
        alert_publisher,
    )


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------


async def run_paper_trading(config: BacktestConfig) -> None:
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
        alert_publisher,
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
    health_monitor.add_alert_callback(on_health_alert)
    health_monitor.add_alert_callback(alert_publisher.on_health_report)

    # Graceful shutdown
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    symbols = config.data.symbols
    live_cfg = config.live
    use_alpaca_feed = live_cfg.broker.broker_type == "alpaca"

    # Set up Alpaca data feed or synthetic ticks
    alpaca_feed: AlpacaDataFeed | None = None
    ticks: list[Tick] = []
    if use_alpaca_feed:
        alpaca_feed = AlpacaDataFeed(
            api_key=live_cfg.broker.api_key,
            api_secret=live_cfg.broker.api_secret,
            feed="iex",
            max_reconnect_attempts=live_cfg.data.reconnect_attempts,
        )

        # Wire Alpaca ticks into the bar aggregator
        alpaca_feed.add_tick_callback(aggregator.on_tick)

        # Record bars for health monitor only when a bar completes
        original_bar_callback = aggregator.bar_callback

        def _on_bar_complete(bar: Bar) -> None:
            if original_bar_callback is not None:
                original_bar_callback(bar)
            health_monitor.record_bar(datetime.now())

        aggregator.bar_callback = _on_bar_complete
    else:
        ticks = generate_synthetic_ticks(
            symbols=symbols,
            duration_minutes=10,
            bar_interval_seconds=live_cfg.data.bar_interval_seconds,
        )

    save_interval = live_cfg.database.save_interval_seconds

    feed_mode = "Alpaca WebSocket (iex)" if use_alpaca_feed else "synthetic ticks"
    print("\n" + "=" * 60)
    print("  PAPER TRADING SESSION")
    print("=" * 60)
    print(f"  Symbols:  {symbols}")
    print(f"  Strategy: {config.strategy.name} {dict(config.strategy.parameters)}")
    print(f"  Capital:  ${config.execution.initial_capital:,.2f}")
    print(f"  Broker:   {live_cfg.broker.broker_type}")
    print(f"  Data:     {feed_mode}")
    print(f"  Storage:  {live_cfg.database.db_url}")
    print("=" * 60 + "\n")

    # Graceful shutdown
    gathered: asyncio.Future[None] | None = None

    def handle_signal() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()
        asyncio.ensure_future(engine.stop())
        if gathered is not None and not gathered.done():
            gathered.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    # Connect broker and sync real account equity
    await order_manager._broker.connect()

    if use_alpaca_feed:
        account = await order_manager._broker.get_account_info()
        real_equity = account["equity"]
        if real_equity > 0:
            portfolio.cash = real_equity
            portfolio.initial_capital = real_equity
            logger.info("Synced capital from Alpaca account: $%.2f", real_equity)

        # Backfill historical bars so strategy has enough history immediately.
        # Each backfilled bar = 1 row in get_latest_bars(), so we need
        # at least lookback_period bars. Pick the Alpaca timeframe that
        # best matches bar_interval_seconds.
        lookback = config.strategy.parameters.get("lookback_period", 60)
        bar_secs = live_cfg.data.bar_interval_seconds
        alpaca_timeframe_map = {
            60: "1Min",
            300: "5Min",
            900: "15Min",
            3600: "1Hour",
            86400: "1Day",
        }
        # Find the closest Alpaca timeframe <= bar_interval_seconds
        best_tf = "1Min"
        for secs, tf in sorted(alpaca_timeframe_map.items()):
            if secs <= bar_secs:
                best_tf = tf
        hist_client = AlpacaHistoricalClient(
            api_key=live_cfg.broker.api_key,
            api_secret=live_cfg.broker.api_secret,
            feed="iex",
        )
        try:
            counts = await backfill_aggregator(
                aggregator=aggregator,
                client=hist_client,
                symbols=symbols,
                num_bars=lookback,
                timeframe=best_tf,
            )
            total = sum(counts.values())
            logger.info(
                "Backfilled %d %s bars across %d symbols (lookback=%d)",
                total,
                best_tf,
                len(counts),
                lookback,
            )
        except Exception:
            logger.exception("Historical backfill failed — starting without history")
        finally:
            await hist_client.close()

    tasks: list[Any] = [
        engine.start(),
        periodic_save(stop_event, save_interval, engine, storage, session_id),
        monitor_health(stop_event, engine, interval=live_cfg.data.bar_interval_seconds),
    ]

    if use_alpaca_feed:
        assert alpaca_feed is not None
        tasks.append(run_alpaca_feed(alpaca_feed, symbols, stop_event))
        tasks.append(poll_order_fills(stop_event, order_manager, portfolio, engine))
    else:
        tasks.append(
            feed_synthetic_ticks(
                ticks,
                aggregator,
                health_monitor,
                live_cfg.data.bar_interval_seconds,
                stop_event,
                engine,
            )
        )

    try:
        gathered = asyncio.gather(*tasks)
        await gathered
    except asyncio.CancelledError:
        pass
    finally:
        # Stop engine if still running
        if engine.is_running:
            await engine.stop()

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

        # Disconnect Alpaca feed if active
        if alpaca_feed is not None:
            await alpaca_feed.disconnect()

        # Disconnect broker
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
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

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

    asyncio.run(run_paper_trading(config))


if __name__ == "__main__":
    main()

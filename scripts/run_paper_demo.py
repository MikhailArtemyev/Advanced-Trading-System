#!/usr/bin/env python3
"""Paper Trading Demo — simulates a live trading session.

Generates synthetic tick data that mimics a real market feed,
aggregates ticks into 1-minute bars, runs an SMA crossover strategy,
and executes trades through the PaperBroker. Prints real-time
updates as bars complete and trades execute.

Usage:
    python scripts/run_paper_demo.py
    make demo
"""

import asyncio
import math
import random
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, ".")

from src.broker.order_manager import OrderManager
from src.broker.paper_broker import PaperBroker
from src.engine.paper_engine import PaperTradingEngine
from src.live.bar_aggregator import BarAggregator
from src.live.data_feed import Tick
from src.live.data_handler import LiveDataHandler
from src.portfolio.portfolio import Portfolio
from src.strategy.sma_crossover import SMACrossoverStrategy


# ---------------------------------------------------------------------------
# Synthetic tick generator
# ---------------------------------------------------------------------------


def generate_ticks(
    symbol: str,
    start_price: float,
    start_time: datetime,
    duration_minutes: int = 120,
    ticks_per_second: int = 2,
) -> list[Tick]:
    """Generate realistic synthetic ticks with trends and noise.

    Creates a price series with:
    - A slow sine wave (trend component)
    - Random walk noise
    - Volume variation
    """
    ticks = []
    price = start_price
    total_ticks = duration_minutes * 60 * ticks_per_second
    trend_period = duration_minutes * 60  # full cycle over the session

    for i in range(total_ticks):
        seconds_elapsed = i / ticks_per_second
        ts = start_time + timedelta(seconds=seconds_elapsed)

        # Trend: sine wave creating crossover opportunities
        trend = math.sin(2 * math.pi * seconds_elapsed / trend_period) * 2.0

        # Noise: small random walk
        noise = random.gauss(0, 0.05)
        price = start_price + trend + noise * math.sqrt(seconds_elapsed / 60)

        # Keep price positive
        price = max(price, start_price * 0.8)

        # Volume varies
        volume = random.uniform(50, 500)

        ticks.append(
            Tick(
                symbol=symbol,
                timestamp=ts,
                price=round(price, 2),
                volume=round(volume, 1),
            )
        )

    return ticks


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_header() -> None:
    print("\n" + "=" * 70)
    print("  PAPER TRADING DEMO — SMA Crossover Strategy")
    print("=" * 70)


def print_bar(bar_num: int, symbol: str, o: float, h: float, l: float, c: float, v: float) -> None:
    direction = "+" if c >= o else "-"
    change_pct = ((c - o) / o) * 100 if o != 0 else 0
    print(
        f"  Bar {bar_num:>3} | {symbol} | "
        f"O={o:>8.2f} H={h:>8.2f} L={l:>8.2f} C={c:>8.2f} | "
        f"V={v:>8.0f} | {direction}{abs(change_pct):.2f}%"
    )


def print_trade(side: str, symbol: str, qty: int, price: float, commission: float) -> None:
    print(
        f"  >>> TRADE: {side} {qty} {symbol} @ ${price:.2f} "
        f"(commission: ${commission:.2f})"
    )


def print_portfolio(cash: float, equity: float, positions: dict) -> None:
    print(f"\n  Portfolio: Equity=${equity:,.2f}  Cash=${cash:,.2f}")
    if positions:
        for sym, pos in positions.items():
            qty = pos.get("quantity", 0)
            if qty != 0:
                print(
                    f"    {sym}: {qty:.0f} shares @ avg ${pos.get('avg_cost', 0):.2f} "
                    f"(value: ${pos.get('market_value', 0):,.2f})"
                )
    else:
        print("    No open positions")


def print_summary(stats: dict, broker: PaperBroker) -> None:
    print("\n" + "=" * 70)
    print("  SESSION SUMMARY")
    print("=" * 70)
    print(f"  Bars processed:    {stats['bars_processed']}")
    print(f"  Events processed:  {stats['events_processed']}")
    print(f"  Orders submitted:  {stats['orders_submitted']}")
    print(f"  Orders rejected:   {stats['orders_rejected']}")
    print(f"  Fills executed:    {stats['fills_processed']}")
    print(f"  Final equity:      ${stats['equity']:,.2f}")

    fills = broker.fills
    if fills:
        print(f"\n  Trade log ({len(fills)} fills):")
        for f in fills:
            print(
                f"    {f.timestamp.strftime('%H:%M:%S')} | "
                f"{f.side:>4} {f.quantity:>4} {f.symbol} @ ${f.price:.2f} "
                f"(comm: ${f.commission:.2f})"
            )
    else:
        print("\n  No trades executed.")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_demo() -> None:
    print_header()

    symbol = "AAPL"
    start_price = 150.0
    initial_capital = 100_000.0
    bar_interval_seconds = 60  # 1-minute bars
    session_minutes = 120  # 2-hour session
    # Use short SMA windows so crossovers happen with fewer bars
    short_window = 5
    long_window = 15

    print(f"\n  Symbol:          {symbol}")
    print(f"  Start price:     ${start_price:.2f}")
    print(f"  Initial capital: ${initial_capital:,.2f}")
    print(f"  Bar interval:    {bar_interval_seconds}s")
    print(f"  Session length:  {session_minutes} minutes")
    print(f"  SMA windows:     {short_window} / {long_window}")

    # --- Generate ticks ---
    print(f"\n  Generating {session_minutes} minutes of synthetic tick data...")
    start_time = datetime(2025, 3, 11, 9, 30, 0)
    ticks = generate_ticks(
        symbol=symbol,
        start_price=start_price,
        start_time=start_time,
        duration_minutes=session_minutes,
        ticks_per_second=2,
    )
    print(f"  Generated {len(ticks)} ticks")

    # --- Set up components ---
    agg = BarAggregator(interval=timedelta(seconds=bar_interval_seconds))
    data_handler = LiveDataHandler(
        symbols=[symbol], bar_aggregator=agg
    )

    strategy = SMACrossoverStrategy(
        symbols=[symbol],
        parameters={"short_window": short_window, "long_window": long_window},
    )

    portfolio = Portfolio(
        initial_capital=initial_capital,
        symbols=[symbol],
        commission_pct=0.001,
        position_size_pct=0.10,
    )

    broker = PaperBroker(
        initial_capital=initial_capital,
        commission_pct=0.001,
        slippage_pct=0.0005,
    )
    await broker.connect()
    broker.set_price(symbol, start_price)

    order_manager = OrderManager(broker)

    engine = PaperTradingEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        order_manager=order_manager,
        risk_manager=None,
        event_poll_interval=0.001,
        bar_poll_interval=0.001,
    )

    # Track fills for printing
    fill_log: list[dict] = []

    def on_fill(fill) -> None:
        fill_log.append({
            "side": fill.side,
            "symbol": fill.symbol,
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
        })
        print_trade(fill.side, fill.symbol, fill.quantity, fill.price, fill.commission)

    broker.add_fill_callback(on_fill)

    # --- Feed ticks and run engine ---
    print("\n" + "-" * 70)
    print("  LIVE SESSION")
    print("-" * 70 + "\n")

    bar_count = 0
    prev_bar_count = 0

    async def feed_ticks_and_run() -> None:
        nonlocal bar_count, prev_bar_count

        # Start the engine in background
        engine_task = asyncio.create_task(engine.start())

        # Feed ticks in batches to simulate real-time
        batch_size = 120  # 1 minute of ticks at 2/sec
        for i in range(0, len(ticks), batch_size):
            batch = ticks[i : i + batch_size]
            for tick in batch:
                agg.on_tick(tick)
                # Update broker price for fills
                broker.set_price(tick.symbol, tick.price)

            # Let the engine process
            await asyncio.sleep(0.01)

            # Print new bars
            completed = agg.get_completed_bars(symbol)
            if len(completed) > prev_bar_count:
                for j in range(prev_bar_count, len(completed)):
                    bar = completed[j]
                    bar_count += 1
                    print_bar(
                        bar_count, bar.symbol,
                        bar.open, bar.high, bar.low, bar.close, bar.volume,
                    )
                prev_bar_count = len(completed)

        # Flush the last partial bar
        agg.flush()
        completed = agg.get_completed_bars(symbol)
        if len(completed) > prev_bar_count:
            bar = completed[-1]
            bar_count += 1
            print_bar(
                bar_count, bar.symbol,
                bar.open, bar.high, bar.low, bar.close, bar.volume,
            )

        # Let engine process remaining events
        await asyncio.sleep(0.1)

        # Stop engine
        await engine.stop()
        await asyncio.sleep(0.1)
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass

    await feed_ticks_and_run()

    # --- Print summary ---
    account = await broker.get_account_info()
    positions = await broker.get_positions()
    print_portfolio(account["cash"], account["equity"], positions)

    stats = engine.get_statistics()
    print_summary(stats, broker)


def main() -> None:
    random.seed(42)  # Reproducible results
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()

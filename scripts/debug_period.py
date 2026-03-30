#!/usr/bin/env python
"""Debug a specific period — show what the strategy is doing bar by bar."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import (
    build_data_handler,
    build_optimizer,
    build_position_sizer,
    build_risk_manager,
    build_strategy,
)
from src.backtest.engine import BacktestEngine
from src.config import load_config
from src.execution.execution_handler import ExecutionHandler
from src.performance.metrics import PerformanceTracker
from src.portfolio.portfolio import Portfolio
from src.storage.null_storage import NullStorage


def main() -> None:
    config = load_config("configs/live/alpaca_momentum_config.yaml")
    config.data.start_date = "2024-03-28"
    config.data.end_date = "2024-09-30"

    data_handler = build_data_handler(config)
    position_sizer = build_position_sizer(config)
    risk_manager = build_risk_manager(config)
    optimizer = build_optimizer(config)
    strategy = build_strategy(config, optimizer, data_handler)

    portfolio = Portfolio(
        initial_capital=config.execution.initial_capital,
        symbols=config.data.symbols,
        commission_pct=config.execution.commission_pct,
        position_sizer=position_sizer,
    )

    execution_handler = ExecutionHandler(
        commission_pct=config.execution.commission_pct,
        slippage_pct=config.execution.slippage_pct,
    )

    performance_tracker = PerformanceTracker(
        initial_capital=config.execution.initial_capital,
    )

    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        performance_tracker=performance_tracker,
        risk_manager=risk_manager,
        storage=NullStorage(),
    )

    results = engine.run()

    # Print trade history with details
    trades = results["trade_history"]
    print(f"Total trades: {len(trades)}")
    print()

    # Get equity curve
    equity_df = performance_tracker.get_equity_curve()
    if "equity" in equity_df.columns:
        eq_values = equity_df["equity"].tolist()
    else:
        eq_values = equity_df.iloc[:, 0].tolist()

    # Find the peak and trough
    peak_val = 0
    peak_idx = 0
    trough_val = float("inf")
    trough_idx = 0
    for i, v in enumerate(eq_values):
        v = float(v)
        if v > peak_val:
            peak_val = v
            peak_idx = i
        if i >= peak_idx and v < trough_val:
            trough_val = v
            trough_idx = i

    print(f"Equity start: ${eq_values[0]:,.0f}")
    print(f"Peak: ${peak_val:,.0f} at bar {peak_idx}")
    print(f"Trough: ${trough_val:,.0f} at bar {trough_idx}")
    print(f"Equity end: ${float(eq_values[-1]):,.0f}")
    print(f"Max DD: {(peak_val - trough_val) / peak_val * 100:.1f}%")
    print()

    # Print all trades sorted by timestamp
    print(f"{'Date':<12} {'Symbol':<8} {'Side':<6} {'Qty':>6} {'Price':>10} {'PnL':>12}")
    print("-" * 60)
    for t in sorted(trades, key=lambda x: x.timestamp):
        date_str = t.timestamp.strftime("%Y-%m-%d") if hasattr(t.timestamp, "strftime") else str(t.timestamp)[:10]
        side = "BUY" if t.quantity > 0 else "SELL"
        pnl = getattr(t, "realized_pnl", 0) or 0
        print(f"{date_str:<12} {t.symbol:<8} {side:<6} {abs(t.quantity):>6} {t.price:>10.2f} {pnl:>+12.2f}")

    # Show biggest losing trades
    print()
    print("=== BIGGEST LOSING TRADES ===")
    losing = sorted(trades, key=lambda x: getattr(x, "realized_pnl", 0) or 0)
    for t in losing[:15]:
        date_str = t.timestamp.strftime("%Y-%m-%d") if hasattr(t.timestamp, "strftime") else str(t.timestamp)[:10]
        pnl = getattr(t, "realized_pnl", 0) or 0
        print(f"  {date_str} {t.symbol:<8} qty={t.quantity:>6} price={t.price:>10.2f} PnL={pnl:>+12.2f}")


if __name__ == "__main__":
    main()

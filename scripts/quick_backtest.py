#!/usr/bin/env python
"""Quick backtest runner for parameter comparison.

Usage:
    python scripts/quick_backtest.py --config configs/live/alpaca_momentum_config.yaml
"""

import argparse
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


def run_backtest(config_path: str) -> dict:
    """Run backtest and return key metrics."""
    config = load_config(config_path)

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
    equity_df = performance_tracker.get_equity_curve()

    # Extract equity values
    if hasattr(equity_df, "values"):
        # DataFrame — get the equity column
        if "equity" in equity_df.columns:
            eq_values = equity_df["equity"].tolist()
        else:
            eq_values = equity_df.iloc[:, 0].tolist()
    else:
        eq_values = list(equity_df)

    # Compute key metrics
    initial = config.execution.initial_capital
    final = eq_values[-1] if eq_values else initial
    total_return = (final - initial) / initial * 100

    # Max drawdown
    peak = initial
    max_dd = 0.0
    for eq in eq_values:
        val = float(eq)
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Recent performance (last 20%)
    n = len(eq_values)
    recent_start = int(n * 0.8)
    if recent_start < n and float(eq_values[recent_start]) > 0:
        recent_return = (float(eq_values[-1]) - float(eq_values[recent_start])) / float(eq_values[recent_start]) * 100
    else:
        recent_return = 0.0

    trades = results["trade_history"]

    return {
        "final_equity": final,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd * 100,
        "num_trades": len(trades),
        "recent_return_pct": recent_return,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print(f"Running: {args.config}")
    metrics = run_backtest(args.config)

    print(f"\n{'='*50}")
    print(f"  Final Equity:     ${metrics['final_equity']:,.2f}")
    print(f"  Total Return:     {metrics['total_return_pct']:+.1f}%")
    print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Recent Return:    {metrics['recent_return_pct']:+.1f}%")
    print(f"  Num Trades:       {metrics['num_trades']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

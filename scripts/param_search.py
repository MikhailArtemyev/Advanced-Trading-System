#!/usr/bin/env python
"""Parameter search for momentum strategy.

Tests different configurations and reports results.
"""

import copy
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


def run_one(config_path: str, overrides: dict | None = None) -> dict:
    """Run a single backtest with optional config overrides."""
    config = load_config(config_path)

    # Apply overrides to strategy/sizing parameters
    if overrides:
        for key, val in overrides.items():
            if key.startswith("strategy."):
                param_name = key.split(".", 1)[1]
                config.strategy.parameters[param_name] = val
            elif key.startswith("sizing."):
                param_name = key.split(".", 1)[1]
                config.sizing.parameters[param_name] = val
            elif key == "start_date":
                config.data.start_date = val
            elif key == "end_date":
                config.data.end_date = val

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

    if hasattr(equity_df, "values"):
        if "equity" in equity_df.columns:
            eq_values = equity_df["equity"].tolist()
        else:
            eq_values = equity_df.iloc[:, 0].tolist()
    else:
        eq_values = list(equity_df)

    initial = config.execution.initial_capital
    final = float(eq_values[-1]) if eq_values else initial
    total_return = (final - initial) / initial * 100

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

    return {
        "final_equity": final,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd * 100,
        "num_trades": len(results["trade_history"]),
        "recent_return_pct": recent_return,
    }


def main() -> None:
    cfg = "configs/live/alpaca_momentum_config.yaml"

    # Round 4: focused on best combinations
    b = {"strategy.top_n": 7, "strategy.trailing_stop_pct": 0.10,
         "strategy.lookback_period": 20, "sizing.max_position_fraction": 0.12}
    tests = {
        # Baselines
        "BASELINE (no filters)":  {**b},
        # SMA100 was best regime filter (SMA50 kills 5yr)
        "sma100":                 {**b, "strategy.regime_sma_period": 100},
        "sma100 mr1%":            {**b, "strategy.regime_sma_period": 100, "strategy.min_return": 0.01},
        "sma100 mr2%":            {**b, "strategy.regime_sma_period": 100, "strategy.min_return": 0.02},
        "sma100 cd10":            {**b, "strategy.regime_sma_period": 100, "strategy.crash_lookback": 10, "strategy.crash_threshold": -0.05, "strategy.recovery_bars": 3},
        "sma100 mr1% cd10":      {**b, "strategy.regime_sma_period": 100, "strategy.min_return": 0.01, "strategy.crash_lookback": 10, "strategy.crash_threshold": -0.05, "strategy.recovery_bars": 3},
        # mr1% + cd10 (no regime)
        "mr1% cd10":             {**b, "strategy.min_return": 0.01, "strategy.crash_lookback": 10, "strategy.crash_threshold": -0.05, "strategy.recovery_bars": 3},
        # SMA 75 (between 50 and 100)
        "sma75":                  {**b, "strategy.regime_sma_period": 75},
        "sma75 mr1%":             {**b, "strategy.regime_sma_period": 75, "strategy.min_return": 0.01},
        # Test with tighter stop + sma100
        "s8% sma100":             {**b, "strategy.trailing_stop_pct": 0.08, "strategy.regime_sma_period": 100},
        "s8% sma100 mr1%":        {**b, "strategy.trailing_stop_pct": 0.08, "strategy.regime_sma_period": 100, "strategy.min_return": 0.01},
        # lb30 + sma100
        "lb30 sma100":            {**b, "strategy.lookback_period": 30, "strategy.regime_sma_period": 100},
        "lb30 sma100 mr1%":       {**b, "strategy.lookback_period": 30, "strategy.regime_sma_period": 100, "strategy.min_return": 0.01},
    }

    windows = [
        ("1yr RECENT", "2025-03-28", "2026-03-28"),
        ("2yr CRASH", "2024-03-28", "2026-03-28"),
        ("5yr FULL", "2021-03-28", "2026-03-28"),
    ]

    for label, start, end in windows:
        print("=" * 80)
        print(f"{label} ({start} to {end})")
        print("=" * 80)
        print(f"{'Config':<30} {'Return':>8} {'MaxDD':>8} {'Recent':>8} {'Trades':>7}")
        print("-" * 80)

        for name, overrides in tests.items():
            ov = {**overrides, "start_date": start, "end_date": end}
            try:
                m = run_one(cfg, ov)
                print(f"{name:<30} {m['total_return_pct']:>+7.1f}% {m['max_drawdown_pct']:>7.1f}% {m['recent_return_pct']:>+7.1f}% {m['num_trades']:>7}")
            except Exception as e:
                print(f"{name:<30} ERROR: {e}")
        print()


if __name__ == "__main__":
    main()

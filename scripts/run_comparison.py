#!/usr/bin/env python
"""Run multiple backtest configurations and compare results side-by-side.

Usage:
    python scripts/run_comparison.py
    python scripts/run_comparison.py --no-plots
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_backtest import (
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

CONFIGS = [
    ("Baseline (Fixed Sizing, No Risk)", "configs/backtest_baseline.yaml"),
    ("Vol Sizing + Risk Mgmt", "configs/backtest_phase2_vol.yaml"),
    ("Mean-Variance Optimized", "configs/backtest_phase2_meanvar.yaml"),
    ("Risk Parity Optimized", "configs/backtest_phase2_riskparity.yaml"),
]


def run_single_config(config_path: str) -> dict:
    """Run a single backtest and return key metrics."""
    config = load_config(config_path)

    data_handler = build_data_handler(config)
    position_sizer = build_position_sizer(config)
    risk_manager = build_risk_manager(config)
    optimizer = build_optimizer(config)
    strategy = build_strategy(config, optimizer)

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
    )

    results = engine.run()

    # Compute metrics
    equity_curve = performance_tracker.get_equity_curve()
    trades = results["trade_history"]
    metrics = performance_tracker.calculate_metrics()
    trade_metrics = performance_tracker.calculate_trade_metrics(trades)

    summary = portfolio.get_portfolio_summary()
    exec_stats = execution_handler.get_statistics()

    return {
        "final_equity": summary["equity"],
        "total_return_pct": (summary["equity"] / config.execution.initial_capital - 1) * 100,
        "total_trades": len(trades),
        "rejected_orders": results.get("rejected_orders", 0),
        "total_commission": exec_stats["total_commission"],
        "total_slippage": exec_stats["total_slippage_cost"],
        "metrics": {**metrics, **trade_metrics},
        "equity_curve": equity_curve,
        "strategy_type": type(strategy).__name__,
        "sizing_method": config.sizing.method,
        "optimization": config.optimization.method,
        "risk_enabled": config.risk.enabled,
    }


def format_pct(val: float) -> str:
    """Format percentage with sign."""
    return f"{val:+.2f}%"


def format_ratio(val: float) -> str:
    """Format a ratio."""
    return f"{val:.3f}"


def format_money(val: float) -> str:
    """Format dollar amount."""
    return f"${val:,.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backtest configurations")
    parser.add_argument("--no-plots", action="store_true", help="Skip equity curve plots")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 2 BACKTEST COMPARISON")
    print("=" * 80)
    print()

    all_results: list[tuple[str, dict]] = []

    for name, config_path in CONFIGS:
        print(f"Running: {name} ({config_path})")
        try:
            result = run_single_config(config_path)
            all_results.append((name, result))
            print(f"  -> Final equity: {format_money(result['final_equity'])}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
        print()

    if not all_results:
        print("No successful runs to compare.")
        return

    # Print comparison table
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()

    # Header
    col_width = 28
    header = f"{'Metric':<30}"
    for name, _ in all_results:
        # Truncate long names
        short = name[:col_width]
        header += f"{short:>{col_width}}"
    print(header)
    print("-" * (30 + col_width * len(all_results)))

    # Rows
    rows = [
        ("Strategy", lambda r: r["strategy_type"]),
        ("Sizing", lambda r: r["sizing_method"]),
        ("Optimization", lambda r: r["optimization"]),
        ("Risk Manager", lambda r: "ON" if r["risk_enabled"] else "OFF"),
        ("", lambda r: ""),
        ("Final Equity", lambda r: format_money(r["final_equity"])),
        ("Total Return", lambda r: format_pct(r["total_return_pct"])),
        ("Total Trades", lambda r: str(r["total_trades"])),
        ("Rejected Orders", lambda r: str(r["rejected_orders"])),
        ("Commission", lambda r: format_money(r["total_commission"])),
        ("Slippage", lambda r: format_money(r["total_slippage"])),
    ]

    # Add metrics rows if available
    metric_rows = [
        ("", lambda r: ""),
        ("Ann. Return", lambda r: format_pct(r["metrics"].get("annualized_return_pct", 0))),
        ("Volatility", lambda r: format_pct(r["metrics"].get("volatility_pct", 0))),
        ("Sharpe Ratio", lambda r: format_ratio(r["metrics"].get("sharpe_ratio", 0))),
        ("Sortino Ratio", lambda r: format_ratio(r["metrics"].get("sortino_ratio", 0))),
        ("Calmar Ratio", lambda r: format_ratio(r["metrics"].get("calmar_ratio", 0))),
        ("Max Drawdown", lambda r: format_pct(r["metrics"].get("max_drawdown_pct", 0))),
        ("Win Rate", lambda r: format_pct(r["metrics"].get("win_rate_pct", 0))),
        ("Profit Factor", lambda r: format_ratio(r["metrics"].get("profit_factor", 0))),
    ]

    for label, fn in rows + metric_rows:
        if label == "":
            print()
            continue
        line = f"{label:<30}"
        for _, result in all_results:
            val = fn(result)
            line += f"{val:>{col_width}}"
        print(line)

    print()
    print("=" * 80)

    # Equity curve comparison plot
    if not args.no_plots and len(all_results) > 1:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(14, 7))
            for name, result in all_results:
                eq = result["equity_curve"]
                if eq is not None and not eq.empty:
                    ax.plot(eq["timestamp"], eq["equity"], label=name, linewidth=1.5)

            ax.set_title("Phase 2 Strategy Comparison — Equity Curves", fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

            plt.tight_layout()
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / "phase2_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Equity curve comparison saved to {save_path}")
            plt.close()
        except ImportError:
            print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()

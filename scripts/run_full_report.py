#!/usr/bin/env python
"""Run all backtest configurations and generate a comprehensive comparison report.

Runs every config in configs/, computes performance metrics, applies
Deflated Sharpe Ratio analysis, ranks strategies, and outputs:
  - Console summary
  - Text report at output/strategy_report.txt
  - Equity curve chart at output/equity_comparison.png
  - Drawdown chart at output/drawdown_comparison.png

Usage:
    python scripts/run_full_report.py
    python scripts/run_full_report.py --no-plots
    make report
"""

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
from src.validation.deflated_sharpe import deflated_sharpe_ratio

# All configs to compare, auto-discovered from configs/ directory
# Each entry: (display_name, config_path)
CONFIGS = [
    ("Baseline (Default)", "configs/backtest_config.yaml"),
    ("Baseline (No Risk)", "configs/backtest_baseline.yaml"),
    ("Conservative Risk", "configs/conservative_risk.yaml"),
    ("Volatility Sizing", "configs/volatility_sizing.yaml"),
    ("Kelly Sizing", "configs/kelly_sizing.yaml"),
    ("Vol + Risk Mgmt", "configs/backtest_phase2_vol.yaml"),
    ("Mean-Var Optimized", "configs/backtest_phase2_meanvar.yaml"),
    ("Risk Parity Optimized", "configs/backtest_phase2_riskparity.yaml"),
    ("ML Strategy (XGBoost)", "configs/ml_backtest_config.yaml"),
]

OUTPUT_DIR = Path("output")


def run_single_config(config_path: str) -> dict[str, Any]:
    """Run a single backtest and return all metrics."""
    config = load_config(config_path)

    # Suppress verbose output from data handler loading and ML training
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        data_handler = build_data_handler(config)
        position_sizer = build_position_sizer(config)
        risk_manager = build_risk_manager(config)
        optimizer = build_optimizer(config)
        strategy = build_strategy(config, optimizer, data_handler)
    finally:
        sys.stdout = old_stdout

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

    # Suppress verbose fill/bar logs from the engine
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        results = engine.run()
    finally:
        sys.stdout = old_stdout

    equity_curve = performance_tracker.get_equity_curve()
    trades = results["trade_history"]
    metrics = performance_tracker.calculate_metrics()
    trade_metrics = performance_tracker.calculate_trade_metrics(trades)

    # Compute return series for DSR
    returns_skew = 0.0
    returns_kurt = 3.0
    n_observations = 0
    if not equity_curve.empty and len(equity_curve) > 2:
        returns = equity_curve["equity"].pct_change().dropna()
        n_observations = len(returns)
        if n_observations > 2:
            returns_skew = float(returns.skew())
            returns_kurt = float(returns.kurtosis()) + 3.0

    summary = portfolio.get_portfolio_summary()
    exec_stats = execution_handler.get_statistics()

    return {
        "config_path": config_path,
        "initial_capital": config.execution.initial_capital,
        "final_equity": summary["equity"],
        "total_return_pct": (
            (summary["equity"] / config.execution.initial_capital - 1) * 100
        ),
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
        "symbols": config.data.symbols,
        "date_range": f"{config.data.start_date} to {config.data.end_date}",
        "n_observations": n_observations,
        "returns_skew": returns_skew,
        "returns_kurt": returns_kurt,
    }


def build_report(
    all_results: list[tuple[str, dict[str, Any]]],
) -> str:
    """Build the full text report."""
    lines: list[str] = []
    w = 90

    lines.append("=" * w)
    lines.append("STRATEGY COMPARISON REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * w)

    # --- Data summary ---
    first = all_results[0][1]
    lines.append("")
    lines.append("DATA")
    lines.append("-" * w)
    lines.append(f"  Symbols:      {', '.join(first['symbols'])}")
    lines.append(f"  Date Range:   {first['date_range']}")
    lines.append(f"  Capital:      ${first['initial_capital']:,.0f}")
    lines.append(f"  Configs Run:  {len(all_results)}")

    # --- Comparison table ---
    lines.append("")
    lines.append("=" * w)
    lines.append("SIDE-BY-SIDE COMPARISON")
    lines.append("=" * w)
    lines.append("")

    # Build table in groups of 5 to fit width
    group_size = 5
    for group_start in range(0, len(all_results), group_size):
        group = all_results[group_start : group_start + group_size]
        col_w = 18
        label_w = 22

        # Header
        header = f"{'Metric':<{label_w}}"
        for name, _ in group:
            short = name[:col_w]
            header += f"{short:>{col_w}}"
        lines.append(header)
        lines.append("-" * (label_w + col_w * len(group)))

        rows: list[tuple[str, Any]] = [
            ("Strategy", lambda r, w=col_w: r["strategy_type"][:w]),
            ("Sizing", lambda r, w=col_w: r["sizing_method"][:w]),
            ("Optimization", lambda r, w=col_w: r["optimization"][:w]),
            ("Risk Mgr", lambda r: "ON" if r["risk_enabled"] else "OFF"),
            ("", None),
            ("Final Equity", lambda r: f"${r['final_equity']:,.0f}"),
            ("Total Return", lambda r: f"{r['total_return_pct']:+.2f}%"),
            (
                "Ann. Return",
                lambda r: f"{r['metrics'].get('annualized_return_pct', 0):+.2f}%",
            ),
            ("Volatility", lambda r: f"{r['metrics'].get('volatility_pct', 0):.2f}%"),
            ("Sharpe", lambda r: f"{r['metrics'].get('sharpe_ratio', 0):.3f}"),
            ("Sortino", lambda r: f"{r['metrics'].get('sortino_ratio', 0):.3f}"),
            ("Calmar", lambda r: f"{r['metrics'].get('calmar_ratio', 0):.3f}"),
            (
                "Max Drawdown",
                lambda r: f"{r['metrics'].get('max_drawdown_pct', 0):.2f}%",
            ),
            ("", None),
            ("Trades", lambda r: str(r["total_trades"])),
            ("Win Rate", lambda r: f"{r['metrics'].get('win_rate_pct', 0):.1f}%"),
            ("Profit Factor", lambda r: f"{r['metrics'].get('profit_factor', 0):.2f}"),
            ("Rejected", lambda r: str(r["rejected_orders"])),
            ("Commission", lambda r: f"${r['total_commission']:,.0f}"),
            ("Slippage", lambda r: f"${r['total_slippage']:,.0f}"),
        ]

        for label, fn in rows:
            if label == "":
                lines.append("")
                continue
            line = f"{label:<{label_w}}"
            for _, result in group:
                val = fn(result)
                line += f"{val:>{col_w}}"
            lines.append(line)

        lines.append("")

    # --- Rankings ---
    lines.append("=" * w)
    lines.append("RANKINGS")
    lines.append("=" * w)
    lines.append("")

    rankings = [
        (
            "By Sharpe Ratio (risk-adjusted return)",
            lambda r: r["metrics"].get("sharpe_ratio", 0),
            True,
        ),
        (
            "By Total Return",
            lambda r: r["total_return_pct"],
            True,
        ),
        (
            "By Max Drawdown (least negative = best)",
            lambda r: r["metrics"].get("max_drawdown_pct", -100),
            True,
        ),
        (
            "By Calmar Ratio (return / drawdown)",
            lambda r: r["metrics"].get("calmar_ratio", 0),
            True,
        ),
        (
            "By Win Rate",
            lambda r: r["metrics"].get("win_rate_pct", 0),
            True,
        ),
    ]

    for title, key_fn, reverse in rankings:
        sorted_results = sorted(
            all_results, key=lambda x: key_fn(x[1]), reverse=reverse
        )
        lines.append(f"  {title}:")
        for rank, (name, result) in enumerate(sorted_results, 1):
            val = key_fn(result)
            if isinstance(val, float):
                lines.append(f"    {rank}. {name:<35} {val:>10.3f}")
            else:
                lines.append(f"    {rank}. {name:<35} {val:>10}")
        lines.append("")

    # --- Deflated Sharpe Ratio analysis ---
    lines.append("=" * w)
    lines.append("DEFLATED SHARPE RATIO ANALYSIS")
    lines.append("=" * w)
    lines.append("")
    lines.append(
        "  Tests whether each strategy's Sharpe ratio is statistically significant"
    )
    lines.append(
        f"  after adjusting for multiple testing ({len(all_results)} configurations tested)."
    )
    lines.append(
        "  A strategy is significant if its Sharpe survives the selection bias"
    )
    lines.append("  of picking the best from N trials.")
    lines.append("")

    n_trials = len(all_results)
    dsr_header = (
        f"  {'Strategy':<35} {'Sharpe':>8} {'E[maxSR]':>10} "
        f"{'Deflated':>10} {'p-value':>9} {'Sig?':>6}"
    )
    lines.append(dsr_header)
    lines.append("  " + "-" * (len(dsr_header) - 2))

    for name, result in all_results:
        observed_sr = result["metrics"].get("sharpe_ratio", 0.0)
        n_obs = result["n_observations"]

        if n_obs < 2:
            lines.append(f"  {name:<35} {'N/A':>8} (insufficient data)")
            continue

        dsr_result = deflated_sharpe_ratio(
            observed_sr=observed_sr,
            n_trials=n_trials,
            n_observations=n_obs,
            skewness=result["returns_skew"],
            kurtosis=result["returns_kurt"],
        )

        sig_marker = "YES" if dsr_result.is_significant else "no"
        lines.append(
            f"  {name:<35} {observed_sr:>8.3f} "
            f"{dsr_result.expected_max_sharpe:>10.3f} "
            f"{dsr_result.deflated_sharpe:>10.3f} "
            f"{dsr_result.p_value:>9.4f} "
            f"{sig_marker:>6}"
        )

    lines.append("")
    lines.append(f"  E[max SR] = expected best Sharpe from {n_trials} random trials")
    lines.append("  Deflated  = Observed Sharpe - E[max SR]")
    lines.append("  p-value   = P(true SR > E[max SR]) accounting for skew/kurtosis")
    lines.append("  Sig?      = significant at 5% level (p > 0.95)")

    # --- Per-strategy detail ---
    lines.append("")
    lines.append("=" * w)
    lines.append("PER-STRATEGY DETAIL")
    lines.append("=" * w)

    for name, result in all_results:
        lines.append("")
        lines.append(f"  {name}")
        lines.append(f"  {'=' * len(name)}")
        lines.append(f"  Config:         {result['config_path']}")
        lines.append(f"  Strategy:       {result['strategy_type']}")
        lines.append(f"  Sizing:         {result['sizing_method']}")
        lines.append(f"  Optimization:   {result['optimization']}")
        lines.append(f"  Risk Manager:   {'ON' if result['risk_enabled'] else 'OFF'}")
        lines.append("")

        m = result["metrics"]
        lines.append(
            f"  Capital:        ${result['initial_capital']:,.0f} -> ${result['final_equity']:,.0f}"
        )
        lines.append(f"  Total Return:   {result['total_return_pct']:+.2f}%")
        lines.append(f"  Ann. Return:    {m.get('annualized_return_pct', 0):+.2f}%")
        lines.append(f"  Volatility:     {m.get('volatility_pct', 0):.2f}%")
        lines.append(f"  Sharpe:         {m.get('sharpe_ratio', 0):.3f}")
        lines.append(f"  Sortino:        {m.get('sortino_ratio', 0):.3f}")
        lines.append(f"  Calmar:         {m.get('calmar_ratio', 0):.3f}")
        lines.append(f"  Max Drawdown:   {m.get('max_drawdown_pct', 0):.2f}%")
        lines.append("")
        lines.append(f"  Trades:         {result['total_trades']}")
        lines.append(f"  Win Rate:       {m.get('win_rate_pct', 0):.1f}%")
        lines.append(f"  Profit Factor:  {m.get('profit_factor', 0):.2f}")
        lines.append(f"  Rejected:       {result['rejected_orders']}")
        lines.append(f"  Commission:     ${result['total_commission']:,.2f}")
        lines.append(f"  Slippage:       ${result['total_slippage']:,.2f}")

    # --- Footer ---
    lines.append("")
    lines.append("=" * w)
    lines.append("END OF REPORT")
    lines.append("=" * w)

    return "\n".join(lines)


def generate_plots(all_results: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Generate comparison charts. Returns list of saved file paths."""
    saved: list[str] = []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return saved

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Equity curve comparison ---
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, result in all_results:
        eq = result["equity_curve"]
        if eq is not None and not eq.empty:
            ax.plot(eq["timestamp"], eq["equity"], label=name, linewidth=1.2)

    ax.set_title("Strategy Comparison — Equity Curves", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.tight_layout()

    path = str(OUTPUT_DIR / "equity_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(path)

    # --- Drawdown comparison ---
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, result in all_results:
        eq = result["equity_curve"]
        if eq is not None and not eq.empty:
            equity = eq["equity"]
            running_max = equity.cummax()
            drawdown = (equity - running_max) / running_max * 100
            ax.plot(eq["timestamp"], drawdown, label=name, linewidth=1.0, alpha=0.8)

    ax.set_title("Strategy Comparison — Drawdowns", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = str(OUTPUT_DIR / "drawdown_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(path)

    # --- Bar chart: Sharpe ratios ---
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [n for n, _ in all_results]
    sharpes = [r["metrics"].get("sharpe_ratio", 0) for _, r in all_results]
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in sharpes]
    bars = ax.barh(names, sharpes, color=colors)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Strategy Comparison — Sharpe Ratios", fontsize=14)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, sharpes, strict=True):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    path = str(OUTPUT_DIR / "sharpe_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all configs and generate comparison report"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    print("=" * 70)
    print("FULL STRATEGY COMPARISON")
    print("=" * 70)
    print()

    all_results: list[tuple[str, dict[str, Any]]] = []

    for name, config_path in CONFIGS:
        print(f"  Running: {name:<35} ", end="", flush=True)
        try:
            result = run_single_config(config_path)
            all_results.append((name, result))
            sharpe = result["metrics"].get("sharpe_ratio", 0)
            ret = result["total_return_pct"]
            print(f"Return: {ret:+6.1f}%  Sharpe: {sharpe:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")
    print()

    if not all_results:
        print("No successful runs.")
        return

    # Build and save report
    report = build_report(all_results)

    OUTPUT_DIR.mkdir(exist_ok=True)
    report_path = OUTPUT_DIR / "strategy_report.txt"
    report_path.write_text(report)

    # Print report to console
    print(report)
    print()
    print(f"Report saved to: {report_path}")

    # Generate plots
    if not args.no_plots:
        saved_plots = generate_plots(all_results)
        for p in saved_plots:
            print(f"Chart saved to:  {p}")


if __name__ == "__main__":
    main()

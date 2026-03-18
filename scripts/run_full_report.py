#!/usr/bin/env python
"""Run all backtest configurations and generate a comprehensive comparison report.

Runs every config in configs/, computes performance metrics, applies
Deflated Sharpe Ratio analysis, ranks strategies, and outputs:
  - Console summary
  - Markdown report at output/strategy_report.md
  - Equity curve chart at output/equity_comparison.png
  - Drawdown chart at output/drawdown_comparison.png
  - Sharpe bar chart at output/sharpe_comparison.png

Usage:
    python scripts/run_full_report.py
    python scripts/run_full_report.py --no-plots
    make report
"""

import argparse
import io
import logging
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

# All configs to compare
# Each entry: (display_name, config_path)
CONFIGS = [
    ("Baseline (Default)", "configs/backtest_config.yaml"),
    ("Baseline (No Risk)", "configs/backtest_baseline.yaml"),
    ("Conservative Risk", "configs/conservative_risk.yaml"),
    ("Kelly Sizing", "configs/kelly_sizing.yaml"),
    ("Volatility Sizing", "configs/backtest_phase2_vol.yaml"),
    ("Mean-Var Optimized", "configs/backtest_phase2_meanvar.yaml"),
    ("Risk Parity Optimized", "configs/backtest_phase2_riskparity.yaml"),
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


def _fmt_pct(val: float, sign: bool = False) -> str:
    """Format a percentage value."""
    if sign:
        return f"{val:+.2f}%"
    return f"{val:.2f}%"


def _fmt_dollar(val: float) -> str:
    """Format a dollar value."""
    return f"${val:,.0f}"


def build_report(
    all_results: list[tuple[str, dict[str, Any]]],
    chart_paths: list[str] | None = None,
) -> str:
    """Build the full Markdown report."""
    lines: list[str] = []

    lines.append("# Strategy Comparison Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    # --- Data summary ---
    first = all_results[0][1]
    lines.append("## Data")
    lines.append("")
    lines.append(f"- **Symbols:** {', '.join(first['symbols'])}")
    lines.append(f"- **Date Range:** {first['date_range']}")
    lines.append(f"- **Initial Capital:** {_fmt_dollar(first['initial_capital'])}")
    lines.append(f"- **Configs Run:** {len(all_results)}")
    lines.append("")

    # --- Comparison table ---
    lines.append("## Side-by-Side Comparison")
    lines.append("")

    # Build a single Markdown table with all strategies
    header = "| Metric |"
    separator = "| --- |"
    for name, _ in all_results:
        header += f" {name} |"
        separator += " ---: |"
    lines.append(header)
    lines.append(separator)

    table_rows: list[tuple[str, Any]] = [
        ("Strategy", lambda r: r["strategy_type"]),
        ("Sizing", lambda r: r["sizing_method"]),
        ("Optimization", lambda r: r["optimization"]),
        ("Risk Mgr", lambda r: "ON" if r["risk_enabled"] else "OFF"),
        ("**Final Equity**", lambda r: _fmt_dollar(r["final_equity"])),
        ("**Total Return**", lambda r: _fmt_pct(r["total_return_pct"], sign=True)),
        (
            "Ann. Return",
            lambda r: _fmt_pct(r["metrics"].get("annualized_return_pct", 0), sign=True),
        ),
        ("Volatility", lambda r: _fmt_pct(r["metrics"].get("volatility_pct", 0))),
        ("**Sharpe**", lambda r: f"{r['metrics'].get('sharpe_ratio', 0):.3f}"),
        ("Sortino", lambda r: f"{r['metrics'].get('sortino_ratio', 0):.3f}"),
        ("Calmar", lambda r: f"{r['metrics'].get('calmar_ratio', 0):.3f}"),
        (
            "**Max Drawdown**",
            lambda r: _fmt_pct(r["metrics"].get("max_drawdown_pct", 0)),
        ),
        ("Trades", lambda r: str(r["total_trades"])),
        ("Win Rate", lambda r: f"{r['metrics'].get('win_rate_pct', 0):.1f}%"),
        ("Profit Factor", lambda r: f"{r['metrics'].get('profit_factor', 0):.2f}"),
        ("Rejected", lambda r: str(r["rejected_orders"])),
        ("Commission", lambda r: _fmt_dollar(r["total_commission"])),
        ("Slippage", lambda r: _fmt_dollar(r["total_slippage"])),
    ]

    for label, fn in table_rows:
        row = f"| {label} |"
        for _, result in all_results:
            row += f" {fn(result)} |"
        lines.append(row)

    lines.append("")

    # --- Equity curves chart ---
    if chart_paths and len(chart_paths) >= 1:
        lines.append("## Equity Curves")
        lines.append("")
        lines.append(f"![Equity Curves]({chart_paths[0]})")
        lines.append("")
        # Find best and worst performers
        sorted_by_return = sorted(
            all_results, key=lambda x: x[1]["total_return_pct"], reverse=True
        )
        best_name, best_r = sorted_by_return[0]
        worst_name, worst_r = sorted_by_return[-1]
        lines.append(
            f"The chart above shows portfolio value over time for all "
            f"{len(all_results)} strategies. "
            f"**{best_name}** delivered the highest total return "
            f"({best_r['total_return_pct']:+.1f}%), "
            f"while **{worst_name}** had the lowest "
            f"({worst_r['total_return_pct']:+.1f}%). "
        )

        lines.append("")

    # --- Drawdown chart ---
    if chart_paths and len(chart_paths) >= 2:
        lines.append("## Drawdowns")
        lines.append("")
        lines.append(f"![Drawdowns]({chart_paths[1]})")
        lines.append("")

        sorted_by_dd = sorted(
            all_results,
            key=lambda x: x[1]["metrics"].get("max_drawdown_pct", -100),
            reverse=True,
        )
        best_dd_name, best_dd_r = sorted_by_dd[0]
        worst_dd_name, worst_dd_r = sorted_by_dd[-1]
        best_dd = best_dd_r["metrics"].get("max_drawdown_pct", 0)
        worst_dd = worst_dd_r["metrics"].get("max_drawdown_pct", 0)

        lines.append(
            f"Drawdown measures peak-to-trough decline as a percentage of peak equity. "
            f"**{best_dd_name}** had the shallowest drawdown ({best_dd:.1f}%), "
            f"while **{worst_dd_name}** experienced the deepest ({worst_dd:.1f}%). "
        )

        # Note risk management effect
        risk_on = [(n, r) for n, r in all_results if r["risk_enabled"]]
        risk_off = [(n, r) for n, r in all_results if not r["risk_enabled"]]
        if risk_on and risk_off:
            avg_dd_on = sum(
                r["metrics"].get("max_drawdown_pct", 0) for _, r in risk_on
            ) / len(risk_on)
            avg_dd_off = sum(
                r["metrics"].get("max_drawdown_pct", 0) for _, r in risk_off
            ) / len(risk_off)
            if avg_dd_on > avg_dd_off:  # less negative = better
                lines.append(
                    f"Risk management improves average max drawdown "
                    f"({avg_dd_on:.1f}% vs {avg_dd_off:.1f}% without)."
                )
        lines.append("")

    # --- Sharpe chart ---
    if chart_paths and len(chart_paths) >= 3:
        lines.append("## Sharpe Ratios")
        lines.append("")
        lines.append(f"![Sharpe Ratios]({chart_paths[2]})")
        lines.append("")

        positive_sharpe = [
            (n, r) for n, r in all_results if r["metrics"].get("sharpe_ratio", 0) > 0
        ]
        negative_sharpe = [
            (n, r) for n, r in all_results if r["metrics"].get("sharpe_ratio", 0) <= 0
        ]
        lines.append(
            f"The Sharpe ratio measures risk-adjusted return "
            f"(excess return per unit of volatility). "
            f"{len(positive_sharpe)} of {len(all_results)} strategies "
            f"achieved positive Sharpe ratios"
        )
        if negative_sharpe:
            neg_names = ", ".join(n for n, _ in negative_sharpe)
            lines.append(
                f"while {len(negative_sharpe)} had negative ratios ({neg_names}), "
                f"meaning their returns did not compensate for risk taken."
            )
        else:
            lines.append(".")
        lines.append("")

    # --- Rankings ---
    lines.append("## Rankings")
    lines.append("")

    rankings = [
        (
            "By Sharpe Ratio",
            "risk-adjusted return",
            lambda r: r["metrics"].get("sharpe_ratio", 0),
            True,
            ".3f",
        ),
        (
            "By Total Return",
            "absolute performance",
            lambda r: r["total_return_pct"],
            True,
            ".2f",
        ),
        (
            "By Max Drawdown",
            "least negative is best",
            lambda r: r["metrics"].get("max_drawdown_pct", -100),
            True,
            ".2f",
        ),
        (
            "By Calmar Ratio",
            "return / drawdown",
            lambda r: r["metrics"].get("calmar_ratio", 0),
            True,
            ".3f",
        ),
        (
            "By Win Rate",
            "percentage of profitable trades",
            lambda r: r["metrics"].get("win_rate_pct", 0),
            True,
            ".1f",
        ),
    ]

    for title, desc, key_fn, reverse, fmt in rankings:
        sorted_results = sorted(
            all_results, key=lambda x, kf=key_fn: kf(x[1]), reverse=reverse
        )
        lines.append(f"### {title}")
        lines.append(f"*{desc}*")
        lines.append("")
        lines.append("| Rank | Strategy | Value |")
        lines.append("| ---: | --- | ---: |")
        for rank, (name, result) in enumerate(sorted_results, 1):
            val = key_fn(result)
            lines.append(f"| {rank} | {name} | {val:{fmt}} |")
        lines.append("")

    # --- Deflated Sharpe Ratio analysis ---
    lines.append("## Deflated Sharpe Ratio Analysis")
    lines.append("")
    n_trials = len(all_results)
    lines.append(
        f"Tests whether each strategy's Sharpe ratio is statistically significant "
        f"after adjusting for multiple testing ({n_trials} configurations tested). "
        f"A strategy is significant if its Sharpe survives the selection bias "
        f"of picking the best from N trials."
    )
    lines.append("")

    lines.append("| Strategy | Sharpe | E[max SR] | Deflated | p-value | Significant |")
    lines.append("| --- | ---: | ---: | ---: | ---: | :---: |")

    any_significant = False
    for name, result in all_results:
        observed_sr = result["metrics"].get("sharpe_ratio", 0.0)
        n_obs = result["n_observations"]

        if n_obs < 2:
            lines.append(f"| {name} | N/A | — | — | — | — |")
            continue

        dsr_result = deflated_sharpe_ratio(
            observed_sr=observed_sr,
            n_trials=n_trials,
            n_observations=n_obs,
            skewness=result["returns_skew"],
            kurtosis=result["returns_kurt"],
        )

        sig = "**YES**" if dsr_result.is_significant else "no"
        if dsr_result.is_significant:
            any_significant = True
        lines.append(
            f"| {name} | {observed_sr:.3f} "
            f"| {dsr_result.expected_max_sharpe:.3f} "
            f"| {dsr_result.deflated_sharpe:.3f} "
            f"| {dsr_result.p_value:.4f} "
            f"| {sig} |"
        )

    lines.append("")
    lines.append("> **Legend:**")
    lines.append(
        f"> - **E[max SR]** = expected best Sharpe from {n_trials} random trials"
    )
    lines.append("> - **Deflated** = Observed Sharpe - E[max SR]")
    lines.append(
        "> - **p-value** = P(true SR > E[max SR]) accounting for skew/kurtosis"
    )
    lines.append("> - **Significant** = p-value > 0.95 (5% significance level)")
    lines.append("")

    # DSR interpretation
    if any_significant:
        sig_names = [
            name
            for name, result in all_results
            if result["n_observations"] >= 2
            and deflated_sharpe_ratio(
                observed_sr=result["metrics"].get("sharpe_ratio", 0.0),
                n_trials=n_trials,
                n_observations=result["n_observations"],
                skewness=result["returns_skew"],
                kurtosis=result["returns_kurt"],
            ).is_significant
        ]
        lines.append(
            f"**{', '.join(sig_names)}** survived the multiple testing adjustment. "
            f"However, strategies trained and tested on the same data (in-sample) "
            f"will show inflated significance. Use CPCV for true out-of-sample evaluation."
        )
    else:
        lines.append(
            "**No strategy achieved statistical significance** after adjusting for "
            "multiple testing. The observed Sharpe ratios are consistent with random "
            "variation from testing multiple configurations."
        )
    lines.append("")

    # --- Per-strategy detail ---
    lines.append("## Per-Strategy Detail")
    lines.append("")

    for name, result in all_results:
        m = result["metrics"]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- **Config:** `{result['config_path']}`")
        lines.append(f"- **Strategy:** {result['strategy_type']}")
        lines.append(f"- **Sizing:** {result['sizing_method']}")
        lines.append(f"- **Optimization:** {result['optimization']}")
        lines.append(f"- **Risk Manager:** {'ON' if result['risk_enabled'] else 'OFF'}")
        lines.append("")

        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(
            f"| Capital | {_fmt_dollar(result['initial_capital'])} "
            f"-> {_fmt_dollar(result['final_equity'])} |"
        )
        lines.append(
            f"| Total Return | {_fmt_pct(result['total_return_pct'], sign=True)} |"
        )
        lines.append(
            f"| Ann. Return | "
            f"{_fmt_pct(m.get('annualized_return_pct', 0), sign=True)} |"
        )
        lines.append(f"| Volatility | {_fmt_pct(m.get('volatility_pct', 0))} |")
        lines.append(f"| Sharpe | {m.get('sharpe_ratio', 0):.3f} |")
        lines.append(f"| Sortino | {m.get('sortino_ratio', 0):.3f} |")
        lines.append(f"| Calmar | {m.get('calmar_ratio', 0):.3f} |")
        lines.append(f"| Max Drawdown | {_fmt_pct(m.get('max_drawdown_pct', 0))} |")
        lines.append(f"| Trades | {result['total_trades']} |")
        lines.append(f"| Win Rate | {m.get('win_rate_pct', 0):.1f}% |")
        lines.append(f"| Profit Factor | {m.get('profit_factor', 0):.2f} |")
        lines.append(f"| Rejected | {result['rejected_orders']} |")
        lines.append(f"| Commission | ${result['total_commission']:,.2f} |")
        lines.append(f"| Slippage | ${result['total_slippage']:,.2f} |")
        lines.append("")

    # --- Footer ---
    lines.append("---")
    lines.append(
        f"*Report generated by `make report` on "
        f"{datetime.now().strftime('%Y-%m-%d')}.*"
    )

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
        except Exception:
            logging.exception("Config %s failed", name)
    print()

    if not all_results:
        print("No successful runs.")
        return

    # Generate plots first so we can embed them in the report
    chart_paths: list[str] = []
    if not args.no_plots:
        chart_paths = generate_plots(all_results)
        for p in chart_paths:
            print(f"Chart saved to:  {p}")

    # Build and save report
    report = build_report(all_results, chart_paths)

    OUTPUT_DIR.mkdir(exist_ok=True)
    report_path = OUTPUT_DIR / "strategy_report.md"
    report_path.write_text(report)

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

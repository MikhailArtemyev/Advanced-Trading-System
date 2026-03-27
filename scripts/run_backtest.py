#!/usr/bin/env python
"""Main script to run backtests.

Usage:
    python -m scripts.run_backtest --config configs/backtest_config.yaml
    python scripts/run_backtest.py --config configs/backtest_config.yaml
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path so imports work when run directly
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
from src.performance.visualization import create_full_report
from src.portfolio.portfolio import Portfolio
from src.storage.null_storage import NullStorage
from src.storage.sql_storage import SQLStorage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visual reports",
    )
    args = parser.parse_args()

    # Load configuration
    if not Path(args.config).exists():
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Build components from config
    print("\nInitializing components...")

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

    # Storage — use SQLite if enabled in config, otherwise NullStorage
    db_cfg = config.live.database
    if db_cfg.enabled:
        storage = SQLStorage(db_url=db_cfg.db_url)
    else:
        storage = NullStorage()

    # Create and run backtest engine
    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        performance_tracker=performance_tracker,
        risk_manager=risk_manager,
        storage=storage,
    )

    results = engine.run()

    # Close storage
    storage.close()

    # Print performance report
    performance_tracker.print_report(
        trades=results["trade_history"],
        portfolio_equity_history=portfolio.equity_history,
    )

    # Risk summary
    if risk_manager is not None:
        print("\nRisk Summary:")
        print(f"  Rejected Orders:  {results['rejected_orders']}")
        print(f"  Halted:           {risk_manager.is_halted}")
        if risk_manager.is_halted:
            print(f"  Halt Reason:      {risk_manager.halt_reason}")

    # Portfolio summary
    summary = portfolio.get_portfolio_summary()
    print("\nPortfolio Summary:")
    print(f"  Equity:           ${summary['equity']:,.2f}")
    print(f"  Cash:             ${summary['cash']:,.2f}")
    print(f"  Long Exposure:    ${summary['long_exposure']:,.2f}")
    print(f"  Short Exposure:   ${summary['short_exposure']:,.2f}")
    print(f"  Net Exposure:     ${summary['net_exposure']:,.2f}")
    print(f"  Positions:        {summary['num_positions']}")

    # Execution statistics
    exec_stats = execution_handler.get_statistics()
    print("\nExecution Statistics:")
    print(f"  Orders Processed: {exec_stats['orders_processed']}")
    print(f"  Total Commission: ${exec_stats['total_commission']:.2f}")
    print(f"  Total Slippage:   ${exec_stats['total_slippage_cost']:.2f}")

    # Config info
    print(f"\nStrategy:           {type(strategy).__name__}")
    print(f"Position Sizing:    {config.sizing.method}")
    print(f"Risk Manager:       {'enabled' if config.risk.enabled else 'disabled'}")
    opt_method = config.optimization.method
    if opt_method != "none":
        print(f"Optimization:       {opt_method}")
        print(f"Rebalance Freq:     {config.optimization.rebalance_frequency} bars")
    else:
        print("Optimization:       none")

    # Generate visual reports
    if not args.no_plots:
        print("\nGenerating visual reports...")
        equity_curve = performance_tracker.get_equity_curve()
        create_full_report(equity_curve, output_dir=args.output, show=False)


if __name__ == "__main__":
    main()

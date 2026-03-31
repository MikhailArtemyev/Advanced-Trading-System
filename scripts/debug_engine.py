#!/usr/bin/env python
"""Trace position updates through the engine during crash period."""

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

    # Monkey-patch update_position to log calls
    orig_update = strategy.update_position

    def traced_update(symbol: str, quantity: int) -> None:
        print(f"  [UPDATE_POS] {symbol} -> qty={quantity}")
        orig_update(symbol, quantity)

    strategy.update_position = traced_update

    # Monkey-patch calculate_signals to log
    orig_calc = strategy.calculate_signals

    def traced_calc(timestamp, dh):
        signals = orig_calc(timestamp, dh)
        if signals:
            open_pos = {k: v for k, v in strategy.current_positions.items() if v != 0}
            date = str(timestamp)[:10]
            print(
                f"Bar {strategy._bar_count} ({date}): {len(signals)} signals, positions={open_pos}"
            )
            for s in signals:
                print(f"  {s.signal_type.name:<6} {s.symbol}")
        return signals

    strategy.calculate_signals = traced_calc

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
    print(f"\nTotal trades: {len(results['trade_history'])}")
    print(f"Final equity: ${portfolio.get_equity():,.0f}")


if __name__ == "__main__":
    main()

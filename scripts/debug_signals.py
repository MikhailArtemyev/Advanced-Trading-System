#!/usr/bin/env python
"""Debug what signals the strategy generates during the crash period."""

import sys
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import build_data_handler, build_optimizer, build_strategy
from src.config import load_config


def main() -> None:
    config = load_config("configs/live/alpaca_momentum_config.yaml")
    config.data.start_date = "2024-03-28"
    config.data.end_date = "2024-09-30"

    data_handler = build_data_handler(config)
    optimizer = build_optimizer(config)
    strategy = build_strategy(config, optimizer, data_handler)

    # Manually step through bars and check signals
    bar_count = 0
    while data_handler.continue_backtest:
        data_handler.update_bars()
        bar_count += 1

        # Get timestamp from any symbol
        ts = None
        for sym in config.data.symbols:
            bars = data_handler.get_latest_bars(sym, 1)
            if not bars.empty:
                ts = bars.index[-1]
                break

        if ts is None:
            continue

        signals = strategy.calculate_signals(ts, data_handler)

        if signals:
            date_str = str(ts)[:10]
            positions = dict(strategy.current_positions)
            open_pos = {k: v for k, v in positions.items() if v != 0}
            print(f"Bar {bar_count} ({date_str}): {len(signals)} signals, "
                  f"{len(open_pos)} open positions")
            for s in signals:
                print(f"  {s.signal_type.name:<6} {s.symbol:<8} strength={s.strength:.2f}")

        # Print position summary every 25 bars
        if bar_count % 25 == 0:
            positions = dict(strategy.current_positions)
            open_pos = {k: v for k, v in positions.items() if v != 0}
            date_str = str(ts)[:10]
            print(f"\n--- Bar {bar_count} ({date_str}) position check ---")
            print(f"  Open positions: {open_pos}")
            print(f"  High water marks: {dict(strategy._high_water)}")

            # Check crash detector
            if strategy.crash_lookback > 0:
                crash = strategy._detect_crash(data_handler)
                print(f"  Crash detected: {crash}")
                print(f"  Cooldown until: bar {strategy._cooldown_until}")
            print()

    print(f"\nTotal bars processed: {bar_count}")
    print(f"Final positions: {dict(strategy.current_positions)}")


if __name__ == "__main__":
    main()

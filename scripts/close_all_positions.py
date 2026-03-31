#!/usr/bin/env python3
"""Close all open positions on Alpaca.

Connects to the broker, fetches all positions, and submits
market orders to close each one. Supports both long and short positions.

Usage:
    python scripts/close_all_positions.py
    python scripts/close_all_positions.py --config configs/live/alpaca_paper_config.yaml
    python scripts/close_all_positions.py --dry-run   # Preview without executing
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.broker.alpaca_broker import AlpacaBroker
from src.broker.base_broker import BrokerOrder, OrderStatus
from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("close_all")


async def close_all_positions(broker: AlpacaBroker, dry_run: bool = False) -> None:
    """Fetch all positions and submit market orders to close them."""
    await broker.connect()

    try:
        account = await broker.get_account_info()
        print(f"\nAccount equity: ${account['equity']:,.2f}")
        print(f"Account cash:   ${account['cash']:,.2f}\n")

        positions = await broker.get_positions()

        if not positions:
            print("No open positions. Nothing to close.")
            return

        print(f"Found {len(positions)} open position(s):\n")
        print(f"  {'Symbol':<8} {'Qty':>8} {'Side':<6} {'Market Value':>14}")
        print(f"  {'-'*8} {'-'*8} {'-'*6} {'-'*14}")

        for symbol, pos in sorted(positions.items()):
            qty = pos["quantity"]
            side = "LONG" if qty > 0 else "SHORT"
            print(f"  {symbol:<8} {qty:>8.0f} {side:<6} ${pos['market_value']:>13,.2f}")

        print()

        if dry_run:
            print("[DRY RUN] No orders submitted.")
            return

        # Submit closing orders
        for symbol, pos in positions.items():
            qty = pos["quantity"]
            if qty == 0:
                continue

            # Long position → sell; short position → buy to cover
            if qty > 0:
                side = "SELL"
                close_qty = int(qty)
            else:
                side = "BUY"
                close_qty = int(abs(qty))

            order = BrokerOrder(
                order_id=f"close-{symbol}",
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=close_qty,
            )

            result = await broker.submit_order(order)

            if result.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED):
                logger.info(
                    "Closing %s: %s %d shares → order %s",
                    symbol,
                    side,
                    close_qty,
                    result.broker_order_id,
                )
            elif result.status == OrderStatus.FILLED:
                logger.info(
                    "Closed %s: %s %d @ $%.2f",
                    symbol,
                    side,
                    result.filled_quantity,
                    result.filled_avg_price,
                )
            else:
                logger.error(
                    "Failed to close %s: status=%s", symbol, result.status.name
                )

        # Wait briefly and check final state
        await asyncio.sleep(2)
        remaining = await broker.get_positions()
        if remaining:
            print(f"\n⚠ {len(remaining)} position(s) still open (may be filling):")
            for sym, p in remaining.items():
                print(f"  {sym}: {p['quantity']:.0f} shares")
        else:
            print("\nAll positions closed.")

        account = await broker.get_account_info()
        print(f"\nFinal equity: ${account['equity']:,.2f}")
        print(f"Final cash:   ${account['cash']:,.2f}")

    finally:
        await broker.disconnect()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Close all open positions")
    parser.add_argument(
        "--config",
        default="configs/live/alpaca_paper_config.yaml",
        help="Path to config YAML (for broker credentials)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show positions without closing them",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    live_cfg = config.live

    api_key = live_cfg.broker.api_key
    api_secret = live_cfg.broker.api_secret
    if not api_key or not api_secret:
        print("Error: ALPACA_API_KEY and ALPACA_API_SECRET required")
        sys.exit(1)

    broker = AlpacaBroker(
        api_key=api_key,
        api_secret=api_secret,
        base_url=live_cfg.broker.base_url,
        paper_mode=live_cfg.broker.paper_mode,
    )

    asyncio.run(close_all_positions(broker, dry_run=args.dry_run))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Download historical OHLCV data for backtesting.

Usage:
    python scripts/download_data.py --symbols AAPL MSFT --start 2020-01-01 --end 2023-12-31
    python scripts/download_data.py  # Uses defaults: AAPL, MSFT, GOOGL
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_symbol(symbol: str, start: str, end: str, output_dir: str) -> bool:
    """Download OHLCV data for a single symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        output_dir: Directory to save CSV files

    Returns:
        True if download successful, False otherwise
    """
    try:
        print(f"Downloading {symbol}...", end=" ")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)

        if df.empty:
            print(f"No data found for {symbol}")
            return False

        # Standardize columns
        df = df.reset_index()
        df.columns = df.columns.str.lower()

        # Keep only OHLCV columns
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in required_cols if c in df.columns]

        if len(available_cols) < len(required_cols):
            missing = set(required_cols) - set(available_cols)
            print(f"Warning: Missing columns {missing}")

        df = df[available_cols]

        # Ensure date is in proper format
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # Save to CSV
        output_path = Path(output_dir) / f"{symbol}.csv"
        df.to_csv(output_path, index=False)

        print(f"OK - {len(df)} bars saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return False


def main() -> None:
    """Main entry point for data download script."""
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data for backtesting"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL"],
        help="Stock symbols to download (default: AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date in YYYY-MM-DD format (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2023-12-31",
        help="End date in YYYY-MM-DD format (default: 2023-12-31)",
    )
    parser.add_argument(
        "--output",
        default="data/sample",
        help="Output directory for CSV files (default: data/sample)",
    )
    args = parser.parse_args()

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"Downloading data from {args.start} to {args.end}")
    print(f"Output directory: {args.output}")
    print("-" * 50)

    # Download each symbol
    success_count = 0
    for symbol in args.symbols:
        if download_symbol(symbol, args.start, args.end, args.output):
            success_count += 1

    print("-" * 50)
    print(f"Downloaded {success_count}/{len(args.symbols)} symbols successfully")


if __name__ == "__main__":
    main()

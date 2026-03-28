#!/usr/bin/env python3
"""Walk-forward backtest for ML-filtered strategies.

Eliminates look-ahead bias by retraining the ML model periodically
using only data available up to that point.

Downloads ALL data once, then slices from CSVs for each period —
no repeated API calls.

For each test window:
  1. Train model on the preceding N months (training window)
  2. Run backtest on the test window with that model
  3. Carry over final equity to the next period

Usage:
    python scripts/run_walkforward.py \
        --config configs/ml/ml_filtered_mean_reversion.yaml \
        --train-months 6 --test-months 1
"""

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # type: ignore[import-untyped]
import pandas as pd
from xgboost import XGBClassifier  # type: ignore[import-untyped]

from scripts.utils import (
    build_optimizer,
    build_position_sizer,
    build_risk_manager,
    build_strategy,
)
from src.backtest.engine import BacktestEngine
from src.config import load_config
from src.data.alpaca_handler import AlpacaBarDataHandler
from src.data.data_handler import HistoricalCSVDataHandler
from src.execution.execution_handler import ExecutionHandler
from src.ml.features import FEATURE_NAMES
from src.ml.labeling import label_trades
from src.portfolio.portfolio import Portfolio
from src.storage.null_storage import NullStorage

FULL_CACHE_DIR = "data/walkforward_full"


def _download_full_range(
    symbols: list[str],
    start_date: str,
    end_date: str,
    timeframe: str,
    api_key: str,
    api_secret: str,
) -> None:
    """Download the entire date range once into FULL_CACHE_DIR."""
    cache = Path(FULL_CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)

    # Use AlpacaBarDataHandler — it skips symbols already cached
    AlpacaBarDataHandler(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        api_key=api_key,
        api_secret=api_secret,
        feed="iex",
        cache_dir=FULL_CACHE_DIR,
    )
    print(f"  All data cached in {FULL_CACHE_DIR}/")


def _slice_csvs(
    symbols: list[str],
    start_date: str,
    end_date: str,
    dest_dir: str,
) -> None:
    """Slice the full cached CSVs to a date range, write to dest_dir."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    for symbol in symbols:
        src = Path(FULL_CACHE_DIR) / f"{symbol}.csv"
        if not src.exists():
            continue
        df = pd.read_csv(src, parse_dates=["date"])
        mask = (df["date"] >= start) & (df["date"] <= end)
        sliced = df[mask]
        if not sliced.empty:
            sliced.to_csv(Path(dest_dir) / f"{symbol}.csv", index=False)


def _train_model_from_csvs(
    symbols: list[str],
    train_start: str,
    train_end: str,
    params: dict,
    cache_dir: str,
) -> object:
    """Train an XGBoost model on pre-sliced CSV data."""
    all_labels: list[pd.DataFrame] = []
    for symbol in symbols:
        csv_path = Path(cache_dir) / f"{symbol}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        if df.empty:
            continue
        labeled = label_trades(
            df=df,
            symbol=symbol,
            lookback_period=params.get("lookback_period", 20),
            entry_threshold=params.get("entry_threshold", 2.0),
            exit_threshold=params.get("exit_threshold", 0.5),
            max_holding_period=params.get("max_holding_period", 10),
            cost_bps=30,
            stop_loss_atr=params.get("stop_loss_atr", 1.5),
            take_profit_atr=params.get("take_profit_atr", 2.5),
            atr_period=params.get("atr_period", 14),
        )
        if not labeled.empty:
            all_labels.append(labeled)

    if not all_labels:
        return None

    dataset = pd.concat(all_labels, ignore_index=True)
    x_train = dataset[FEATURE_NAMES].values
    y_train = dataset["label"].values.astype(int)

    if len(x_train) < 50:
        return None

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(x_train, y_train)

    return {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": params.get("threshold", 0.65),
        "train_end": train_end,
        "strategy_params": params,
        "symbols": symbols,
    }


def _run_period_backtest(
    config_path: str,
    model_path: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    cache_dir: str,
    timeframe: str,
) -> tuple[float, int]:
    """Run backtest for a single period using pre-sliced CSV data.

    Returns (final_equity, n_trades).
    """
    config = load_config(config_path)

    config.data.start_date = start_date
    config.data.end_date = end_date
    config.data.data_path = cache_dir
    config.execution.initial_capital = initial_capital
    config.strategy.parameters["model_path"] = model_path

    # Use CSV handler directly — data already sliced in cache_dir
    data_handler = HistoricalCSVDataHandler(
        data_path=cache_dir,
        symbols=config.data.symbols,
        start_date=start_date,
        end_date=end_date,
    )

    position_sizer = build_position_sizer(config)
    risk_manager = build_risk_manager(config)
    optimizer = build_optimizer(config)
    strategy = build_strategy(config, optimizer, data_handler)

    portfolio = Portfolio(
        initial_capital=initial_capital,
        symbols=config.data.symbols,
        commission_pct=config.execution.commission_pct,
        position_sizer=position_sizer,
    )

    execution_handler = ExecutionHandler(
        commission_pct=config.execution.commission_pct,
        slippage_pct=config.execution.slippage_pct,
    )

    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        risk_manager=risk_manager,
        storage=NullStorage(),
    )

    results = engine.run()
    return results["final_equity"], len(results["trade_history"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward ML backtest")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument(
        "--train-months", type=int, default=6, help="Training window in months"
    )
    parser.add_argument(
        "--test-months", type=int, default=1, help="Test window in months"
    )
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    symbols = config.data.symbols
    params = dict(config.strategy.parameters)

    backtest_start = datetime.strptime(config.data.start_date, "%Y-%m-%d")
    backtest_end = datetime.strptime(config.data.end_date, "%Y-%m-%d")
    initial_capital = config.execution.initial_capital

    # Determine timeframe
    bar_secs = config.live.data.bar_interval_seconds
    tf_map = {60: "1Min", 300: "5Min", 900: "15Min", 3600: "1Hour"}
    timeframe = tf_map.get(bar_secs, "5Min")

    print("=" * 60)
    print("WALK-FORWARD BACKTEST")
    print("=" * 60)
    print(f"Backtest period: {config.data.start_date} to {config.data.end_date}")
    print(f"Train window:    {args.train_months} months")
    print(f"Test window:     {args.test_months} months")
    print(f"Timeframe:       {timeframe}")
    print(f"Symbols:         {len(symbols)} stocks")
    print(f"Initial capital: ${initial_capital:,.0f}")
    print()

    # Step 1: Download ALL data once (training start of first period to backtest end)
    earliest_train_start = backtest_start - timedelta(days=args.train_months * 30)
    full_start = str(earliest_train_start.date())
    full_end = str(backtest_end.date())

    print(f"Downloading full data range: {full_start} to {full_end}")
    _download_full_range(
        symbols=symbols,
        start_date=full_start,
        end_date=full_end,
        timeframe=timeframe,
        api_key=config.live.broker.api_key,
        api_secret=config.live.broker.api_secret,
    )
    print()

    # Step 2: Generate test periods
    periods: list[tuple[datetime, datetime]] = []
    cursor = backtest_start
    while cursor < backtest_end:
        period_end = min(cursor + timedelta(days=args.test_months * 30), backtest_end)
        periods.append((cursor, period_end))
        cursor = period_end

    equity = initial_capital
    milestones: list[tuple[datetime, float]] = [(backtest_start, initial_capital)]
    total_trades = 0

    model_dir = Path("models/walkforward")
    model_dir.mkdir(parents=True, exist_ok=True)
    slice_dir = Path("data/walkforward_slices")

    # Step 3: Walk forward through each period
    for i, (test_start, test_end) in enumerate(periods):
        train_start = test_start - timedelta(days=args.train_months * 30)
        train_start_str = str(train_start.date())
        train_end_str = str(test_start.date())
        test_start_str = str(test_start.date())
        test_end_str = str(test_end.date())

        print(f"--- Period {i + 1}/{len(periods)} ---")
        print(f"  Train: {train_start_str} to {train_end_str}")
        print(f"  Test:  {test_start_str} to {test_end_str}")
        print(f"  Equity: ${equity:,.2f}")

        # Slice training data
        train_slice = str(slice_dir / f"train_{i}")
        _slice_csvs(symbols, train_start_str, train_end_str, train_slice)

        # Train model
        print("  Training model...")
        artifact = _train_model_from_csvs(
            symbols=symbols,
            train_start=train_start_str,
            train_end=train_end_str,
            params=params,
            cache_dir=train_slice,
        )

        if artifact is None:
            print("  WARNING: Not enough training data, skipping period")
            continue

        model_path = str(model_dir / f"model_{i}.joblib")
        joblib.dump(artifact, model_path)

        # Slice test data
        test_slice = str(slice_dir / f"test_{i}")
        _slice_csvs(symbols, test_start_str, test_end_str, test_slice)

        # Run backtest
        print("  Running backtest...")
        final_eq, n_trades = _run_period_backtest(
            config_path=args.config,
            model_path=model_path,
            start_date=test_start_str,
            end_date=test_end_str,
            initial_capital=equity,
            cache_dir=test_slice,
            timeframe=timeframe,
        )

        pnl = final_eq - equity
        pnl_pct = 100 * pnl / equity
        total_trades += n_trades
        print(f"  Result: ${final_eq:,.2f} ({pnl:+,.2f}, {pnl_pct:+.2f}%)")
        print(f"  Trades: {n_trades}")

        equity = final_eq
        milestones.append((test_end, equity))

    # Clean up slices
    if slice_dir.exists():
        shutil.rmtree(slice_dir)

    # Final summary
    total_return = 100 * (equity - initial_capital) / initial_capital
    print()
    print("=" * 60)
    print("WALK-FORWARD RESULTS")
    print("=" * 60)
    print(f"Initial Capital:  ${initial_capital:,.2f}")
    print(f"Final Equity:     ${equity:,.2f}")
    print(f"Total Return:     {total_return:+.2f}%")
    print(f"Total Trades:     {total_trades}")
    print(f"Periods:          {len(periods)}")

    # Save equity curve chart
    if len(milestones) > 1:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt

            timestamps = [m[0] for m in milestones]
            equities = [m[1] for m in milestones]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(timestamps, equities, linewidth=1.5, marker="o", markersize=4)
            ax.set_title("Walk-Forward Equity Curve (Out-of-Sample)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            fig.autofmt_xdate()
            ax.grid(True, alpha=0.3)
            ax.axhline(
                y=initial_capital,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Start",
            )
            ax.legend()

            chart_path = output_dir / "walkforward_equity.png"
            fig.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\nEquity curve saved to {chart_path}")
        except ImportError:
            print("\nmatplotlib not available — skipping chart")


if __name__ == "__main__":
    main()

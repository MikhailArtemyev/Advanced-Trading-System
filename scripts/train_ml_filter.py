#!/usr/bin/env python3
"""Train the ML signal filter for mean-reversion trading.

Downloads historical intraday bars from Alpaca, simulates mean-reversion
entries, labels them as profitable/unprofitable, extracts features, and
trains an XGBoost classifier.

Usage:
    python scripts/train_ml_filter.py \\
        --config configs/backtest/backtest_intraday_5min.yaml \\
        --train-months 6 \\
        --output models/ml_filter.joblib
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from xgboost import XGBClassifier  # type: ignore[import-untyped]

from src.config import load_config
from src.data.alpaca_handler import AlpacaBarDataHandler
from src.ml.features import FEATURE_NAMES
from src.ml.labeling import label_trades


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML signal filter")
    parser.add_argument(
        "--config",
        default="configs/backtest/backtest_intraday_5min.yaml",
        help="Config YAML (for symbols, strategy params, Alpaca credentials)",
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=6,
        help="Months of historical data for training (default: 6)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=14,
        help="Days of recent data for testing (default: 14)",
    )
    parser.add_argument(
        "--output",
        default="models/ml_filter.joblib",
        help="Output path for trained model",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    symbols = config.data.symbols
    params = dict(config.strategy.parameters)

    # Date ranges
    end = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    test_start = end - timedelta(days=args.test_days)
    train_start = test_start - timedelta(days=args.train_months * 30)

    print(f"Training period: {train_start.date()} to {test_start.date()}")
    print(f"Testing period:  {test_start.date()} to {end.date()}")
    print(f"Symbols: {symbols}")

    # Determine timeframe from config
    bar_secs = config.live.data.bar_interval_seconds
    tf_map = {60: "1Min", 300: "5Min", 900: "15Min", 3600: "1Hour"}
    timeframe = tf_map.get(bar_secs, "5Min")
    print(f"Timeframe: {timeframe}")

    # Download data
    print("\nDownloading historical bars...")
    handler = AlpacaBarDataHandler(
        symbols=symbols,
        start_date=str(train_start.date()),
        end_date=str(end.date()),
        timeframe=timeframe,
        api_key=config.live.broker.api_key,
        api_secret=config.live.broker.api_secret,
        feed="iex",
        cache_dir="data/ml_training_cache",
    )

    # Label trades for each symbol
    print("\nLabeling trades...")
    all_labels: list[pd.DataFrame] = []

    for symbol in symbols:
        csv_path = Path("data/ml_training_cache") / f"{symbol}.csv"
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

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
            wins = int(labeled["label"].sum())
            total = len(labeled)
            print(
                f"  {symbol}: {total} signals, "
                f"{wins} profitable ({100 * wins / total:.0f}%)"
            )

    if not all_labels:
        print("ERROR: No labeled trades found. Check date range and parameters.")
        sys.exit(1)

    dataset = pd.concat(all_labels, ignore_index=True)
    print(f"\nTotal dataset: {len(dataset)} samples")
    print(
        f"  Profitable: {int(dataset['label'].sum())} ({100 * dataset['label'].mean():.1f}%)"
    )
    print(
        f"  Unprofitable: {int((1 - dataset['label']).sum())} ({100 * (1 - dataset['label'].mean()):.1f}%)"
    )

    # Chronological train/test split
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
    test_cutoff = pd.Timestamp(test_start).tz_localize(None)

    train_mask = dataset["timestamp"] < test_cutoff
    test_mask = dataset["timestamp"] >= test_cutoff

    train_df = dataset[train_mask]
    test_df = dataset[test_mask]

    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Test set:  {len(test_df)} samples")

    if len(train_df) < 50:
        print("ERROR: Not enough training samples. Try a longer training period.")
        sys.exit(1)

    # Prepare features and labels
    x_train = train_df[FEATURE_NAMES].values
    y_train = train_df["label"].values.astype(int)
    x_test = test_df[FEATURE_NAMES].values if len(test_df) > 0 else np.array([])
    y_test = test_df["label"].values.astype(int) if len(test_df) > 0 else np.array([])

    # Handle class imbalance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Train XGBoost
    print("\nTraining XGBoost classifier...")
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

    # Feature importance
    print("\nFeature importance:")
    importances = model.feature_importances_
    for name, imp in sorted(
        zip(FEATURE_NAMES, importances, strict=True), key=lambda x: -x[1]
    ):
        print(f"  {name:20s} {imp:.4f}")

    # Evaluate on test set
    metrics: dict[str, float] = {}
    if len(x_test) > 0:
        from sklearn.metrics import (  # type: ignore[import-untyped]
            accuracy_score,
            classification_report,
            confusion_matrix,
        )

        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)

        print("\nTest set results:")
        print(classification_report(y_test, y_pred, target_names=["Skip", "Take"]))
        print(f"Confusion matrix:\n{cm}")

        # Filter impact analysis
        take_mask = y_prob >= 0.5
        if take_mask.any():
            taken_returns = test_df[take_mask]["return_pct"].values
            blocked_returns = test_df[~take_mask]["return_pct"].values
            print("\nFilter impact (test set):")
            print(
                f"  Trades taken:   {take_mask.sum()} "
                f"(avg return: {np.mean(taken_returns):+.3f}%)"
            )
            print(
                f"  Trades blocked: {(~take_mask).sum()} "
                f"(avg return: {np.mean(blocked_returns):+.3f}%)"
                if len(blocked_returns) > 0
                else "  Trades blocked: 0"
            )
    else:
        print("\nNo test data — skipping evaluation")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
        "train_end": str(test_start.date()),
        "metrics": metrics,
        "strategy_params": params,
        "symbols": symbols,
        "timeframe": timeframe,
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    joblib.dump(artifact, output_path)
    print(f"\nModel saved to {output_path}")

    # Keep reference to handler to avoid unused-variable lint
    _ = handler


if __name__ == "__main__":
    main()

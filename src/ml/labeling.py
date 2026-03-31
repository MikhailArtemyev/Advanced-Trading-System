"""Trade labeling for ML training data.

Simulates MeanReversionStrategy entries on historical bar data and
labels each entry as profitable (1) or unprofitable (0) based on
the actual future price movement.
"""

import numpy as np
import pandas as pd

from .features import extract_features


def label_trades(
    df: pd.DataFrame,
    symbol: str,
    lookback_period: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    max_holding_period: int = 10,
    cost_bps: float = 30,
    stop_loss_atr: float = 1.5,
    take_profit_atr: float = 2.5,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Simulate mean-reversion entries and label outcomes.

    For each bar where z-score crosses the entry threshold, look forward
    to determine if the trade would have been profitable after costs.

    Args:
        df: OHLCV DataFrame (columns: open, high, low, close, volume).
        symbol: Symbol name (added to output).
        lookback_period: Rolling window for z-score.
        entry_threshold: Z-score threshold for entry.
        exit_threshold: Z-score threshold for exit.
        max_holding_period: Max bars to hold before force-exit.
        cost_bps: Round-trip cost in basis points (e.g. 30 = 0.30%).
        stop_loss_atr: Stop-loss distance in ATR multiples (0 to disable).
        take_profit_atr: Take-profit distance in ATR multiples (0 to disable).
        atr_period: Lookback for ATR computation.

    Returns:
        DataFrame with columns: [timestamp, symbol, signal_type, label]
        plus all feature columns. Only rows with entry signals.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    cost_pct = cost_bps / 10000.0

    records: list[dict[str, object]] = []

    for i in range(lookback_period, n - max_holding_period):
        window = close[i - lookback_period + 1 : i + 1]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        if std == 0:
            continue

        z = (close[i] - mean) / std

        signal_type: str | None = None
        if z <= -entry_threshold:
            signal_type = "LONG"
        elif z >= entry_threshold:
            signal_type = "SHORT"

        if signal_type is None:
            continue

        # Compute exit price: simulate the strategy's exit logic
        entry_price = close[i]
        exit_price = _simulate_exit(
            close=close,
            entry_idx=i,
            signal_type=signal_type,
            lookback_period=lookback_period,
            exit_threshold=exit_threshold,
            max_holding_period=max_holding_period,
            high=high,
            low=low,
            stop_loss_atr=stop_loss_atr,
            take_profit_atr=take_profit_atr,
            atr_period=atr_period,
        )

        # Label: profitable after costs?
        if signal_type == "LONG":
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price

        label = 1 if ret > cost_pct else 0

        # Extract features (backward-looking only)
        bar_window = df.iloc[i - lookback_period + 1 : i + 1]
        feats = extract_features(bar_window)
        if feats is None:
            continue

        record: dict[str, object] = {
            "timestamp": df.index[i] if hasattr(df.index, "dtype") else i,
            "symbol": symbol,
            "signal_type": signal_type,
            "label": label,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": ret * 100,
        }
        record.update(feats)
        records.append(record)

    return pd.DataFrame(records)


def _simulate_exit(
    close: np.ndarray,
    entry_idx: int,
    signal_type: str,
    lookback_period: int,
    exit_threshold: float,
    max_holding_period: int,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    stop_loss_atr: float = 0.0,
    take_profit_atr: float = 0.0,
    atr_period: int = 14,
) -> float:
    """Simulate the mean-reversion exit logic to find exit price.

    Mirrors MeanReversionStrategy._evaluate_symbol exit conditions
    including ATR-based stop-loss and take-profit.
    """
    entry_price = close[entry_idx]

    # Compute ATR at entry if stop/take-profit enabled
    atr = 0.0
    if (
        (stop_loss_atr > 0 or take_profit_atr > 0)
        and high is not None
        and low is not None
    ):
        n = min(atr_period, entry_idx)
        if n >= 1:
            tr = np.maximum(
                high[entry_idx - n + 1 : entry_idx + 1]
                - low[entry_idx - n + 1 : entry_idx + 1],
                np.maximum(
                    np.abs(
                        high[entry_idx - n + 1 : entry_idx + 1]
                        - close[entry_idx - n : entry_idx]
                    ),
                    np.abs(
                        low[entry_idx - n + 1 : entry_idx + 1]
                        - close[entry_idx - n : entry_idx]
                    ),
                ),
            )
            atr = float(np.mean(tr))

    for offset in range(1, max_holding_period + 1):
        j = entry_idx + offset
        if j >= len(close):
            return float(close[-1])

        price = float(close[j])

        # Stop-loss check
        if stop_loss_atr > 0 and atr > 0:
            stop_dist = atr * stop_loss_atr
            if signal_type == "LONG" and price <= entry_price - stop_dist:
                return price
            if signal_type == "SHORT" and price >= entry_price + stop_dist:
                return price

        # Take-profit check
        if take_profit_atr > 0 and atr > 0:
            tp_dist = atr * take_profit_atr
            if signal_type == "LONG" and price >= entry_price + tp_dist:
                return price
            if signal_type == "SHORT" and price <= entry_price - tp_dist:
                return price

        # Recompute z-score at bar j
        window = close[j - lookback_period + 1 : j + 1]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        if std == 0:
            continue

        z = (close[j] - mean) / std

        # Check exit condition (same as strategy)
        if signal_type == "LONG" and z >= -exit_threshold:
            return float(close[j])
        if signal_type == "SHORT" and z <= exit_threshold:
            return float(close[j])

    # Max holding period reached — force exit
    exit_idx = min(entry_idx + max_holding_period, len(close) - 1)
    return float(close[exit_idx])

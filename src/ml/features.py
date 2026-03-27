"""Feature extraction for ML signal filter.

Computes 8 features from OHLCV bar data for the XGBoost trade filter.
All features are backward-looking — no future data is used.
"""

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "z_score",
    "z_score_velocity",
    "atr_ratio",
    "volume_ratio",
    "rsi_14",
    "bar_range_ratio",
    "close_vs_vwap",
    "return_5",
]

MIN_BARS_REQUIRED = 20


def extract_features(bars: pd.DataFrame) -> dict[str, float] | None:
    """Extract ML features from a DataFrame of recent bars.

    Args:
        bars: DataFrame with columns [open, high, low, close, volume],
              at least MIN_BARS_REQUIRED rows.

    Returns:
        Dict of feature_name -> value, or None if insufficient data.
    """
    if len(bars) < MIN_BARS_REQUIRED:
        return None

    close = bars["close"].values
    high = bars["high"].values
    low = bars["low"].values
    volume = bars["volume"].values

    # 1. Z-score (20-bar)
    mean = float(np.mean(close))
    std = float(np.std(close, ddof=1))
    if std == 0:
        return None
    z_score = (close[-1] - mean) / std

    # 2. Z-score velocity (change over last 3 bars)
    if len(close) >= 5:
        z_prev = (close[-4] - np.mean(close[:-3])) / max(
            np.std(close[:-3], ddof=1), 1e-8
        )
        z_score_velocity = z_score - z_prev
    else:
        z_score_velocity = 0.0

    # 3. ATR ratio (10-bar ATR / close)
    n_atr = min(10, len(close) - 1)
    tr = np.maximum(
        high[-n_atr:] - low[-n_atr:],
        np.maximum(
            np.abs(high[-n_atr:] - close[-(n_atr + 1) : -1]),
            np.abs(low[-n_atr:] - close[-(n_atr + 1) : -1]),
        ),
    )
    atr = float(np.mean(tr))
    atr_ratio = atr / close[-1] if close[-1] != 0 else 0.0

    # 4. Volume ratio (current / 20-bar SMA)
    vol_mean = float(np.mean(volume))
    volume_ratio = float(volume[-1]) / vol_mean if vol_mean > 0 else 1.0

    # 5. RSI (14-bar)
    rsi_14 = _compute_rsi(close, 14)

    # 6. Bar range ratio ((high - low) / close)
    bar_range_ratio = (high[-1] - low[-1]) / close[-1] if close[-1] != 0 else 0.0

    # 7. Close vs VWAP approximation
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    vwap = cum_tp_vol[-1] / cum_vol[-1] if cum_vol[-1] > 0 else close[-1]
    close_vs_vwap = (close[-1] - vwap) / close[-1] if close[-1] != 0 else 0.0

    # 8. 5-bar return
    if len(close) >= 6:
        return_5 = (close[-1] - close[-6]) / close[-6] if close[-6] != 0 else 0.0
    else:
        return_5 = 0.0

    return {
        "z_score": float(z_score),
        "z_score_velocity": float(z_score_velocity),
        "atr_ratio": float(atr_ratio),
        "volume_ratio": float(volume_ratio),
        "rsi_14": float(rsi_14),
        "bar_range_ratio": float(bar_range_ratio),
        "close_vs_vwap": float(close_vs_vwap),
        "return_5": float(return_5),
    }


def extract_features_batch(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute feature columns for every row in df (for training).

    Args:
        df: DataFrame with columns [open, high, low, close, volume].
        lookback: Window size for feature computation.

    Returns:
        DataFrame with feature columns aligned to df's index.
        Rows with insufficient history have NaN values.
    """
    n = len(df)
    results: dict[str, list[float | None]] = {f: [] for f in FEATURE_NAMES}

    for i in range(n):
        if i < lookback - 1:
            for f in FEATURE_NAMES:
                results[f].append(None)
            continue

        window = df.iloc[i - lookback + 1 : i + 1]
        feats = extract_features(window)
        if feats is None:
            for f in FEATURE_NAMES:
                results[f].append(None)
        else:
            for f in FEATURE_NAMES:
                results[f].append(feats[f])

    return pd.DataFrame(results, index=df.index)


def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
    """Compute RSI for the last bar."""
    if len(close) < period + 1:
        return 50.0  # neutral default

    deltas = np.diff(close[-(period + 1) :])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)

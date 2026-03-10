"""Technical indicator feature generators.

Provides standard technical analysis indicators as FeatureGenerator subclasses:
- SMAFeature: Simple Moving Average at multiple windows
- RSIFeature: Relative Strength Index
- MACDFeature: Moving Average Convergence Divergence
- BollingerBandFeature: Bollinger Band width and %B
- ATRFeature: Average True Range (raw and normalised)
"""

import numpy as np
import pandas as pd

from .base_feature import FeatureGenerator, FeatureResult


class SMAFeature(FeatureGenerator):
    """Simple Moving Average features at multiple windows.

    Produces two columns per window:
        sma_{window}:       Raw SMA value
        sma_{window}_ratio: Close / SMA (price relative to moving average)

    Args:
        windows: List of lookback windows. Defaults to [5, 10, 20, 50].
    """

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows: list[int] = windows or [5, 10, 20, 50]

    @property
    def lookback(self) -> int:
        return max(self.windows)

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        close = ohlcv["close"]
        features = pd.DataFrame(index=ohlcv.index)

        for w in self.windows:
            sma = close.rolling(w).mean()
            features[f"sma_{w}"] = sma
            features[f"sma_{w}_ratio"] = close / sma

        names = list(features.columns)
        return FeatureResult(features=features, feature_names=names, method="sma")


class RSIFeature(FeatureGenerator):
    """Relative Strength Index.

    RSI = 100 − 100 / (1 + RS)
    RS  = avg_gain / avg_loss over `period` bars (simple rolling mean).

    Edge case: when avg_loss == 0 (all gains), RSI is defined as 100.

    Produces:
        rsi_{period}: RSI value in [0, 100]

    Args:
        period: Lookback period. Must be >= 1. Defaults to 14.

    Raises:
        ValueError: If period < 1.
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self.period = period

    @property
    def lookback(self) -> int:
        return self.period + 1

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        close = ohlcv["close"]
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        # When avg_loss == 0 and avg_gain > 0, RSI is conventionally 100
        rsi = rsi.where(avg_loss != 0, 100.0)

        col = f"rsi_{self.period}"
        features = pd.DataFrame({col: rsi}, index=ohlcv.index)
        return FeatureResult(features=features, feature_names=[col], method="rsi")


class MACDFeature(FeatureGenerator):
    """Moving Average Convergence Divergence.

    Produces:
        macd:           MACD line  = fast EMA − slow EMA
        macd_signal:    Signal line = EMA(MACD, signal_period)
        macd_histogram: MACD − Signal

    Args:
        fast_period:   Fast EMA period. Defaults to 12.
        slow_period:   Slow EMA period. Defaults to 26.
        signal_period: Signal EMA period. Defaults to 9.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    @property
    def lookback(self) -> int:
        return self.slow_period + self.signal_period

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        close = ohlcv["close"]

        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        names = ["macd", "macd_signal", "macd_histogram"]
        features = pd.DataFrame(
            {
                "macd": macd_line,
                "macd_signal": signal_line,
                "macd_histogram": histogram,
            },
            index=ohlcv.index,
        )
        return FeatureResult(features=features, feature_names=names, method="macd")


class BollingerBandFeature(FeatureGenerator):
    """Bollinger Band features.

    Produces:
        bb_width:  (Upper − Lower) / Middle — normalised band width
        bb_pct_b:  (Close − Lower) / (Upper − Lower) — position within band (%B)

    %B = 0 → price at lower band, %B = 0.5 → price at SMA, %B = 1 → upper band.
    Both features are NaN when the rolling window has zero variance (std = 0).

    Args:
        window:  Rolling window for SMA and std. Defaults to 20.
        num_std: Number of standard deviations for band width. Defaults to 2.0.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window = window
        self.num_std = num_std

    @property
    def lookback(self) -> int:
        return self.window

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        close = ohlcv["close"]
        sma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()

        upper = sma + self.num_std * std
        lower = sma - self.num_std * std

        band_range = upper - lower
        bb_width = band_range / sma
        bb_pct_b = (close - lower) / band_range.replace(0, np.nan)

        features = pd.DataFrame(
            {"bb_width": bb_width, "bb_pct_b": bb_pct_b},
            index=ohlcv.index,
        )
        return FeatureResult(
            features=features,
            feature_names=["bb_width", "bb_pct_b"],
            method="bollinger",
        )


class ATRFeature(FeatureGenerator):
    """Average True Range (raw and normalised).

    True Range = max(High−Low, |High−PrevClose|, |Low−PrevClose|)
    ATR        = rolling mean of True Range over `period` bars.

    Produces:
        atr_{period}:     Raw ATR value
        atr_{period}_pct: ATR / Close (volatility as fraction of price)

    Args:
        period: ATR smoothing period. Defaults to 14.
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period

    @property
    def lookback(self) -> int:
        return self.period + 1

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(self.period).mean()

        atr_col = f"atr_{self.period}"
        pct_col = f"atr_{self.period}_pct"
        features = pd.DataFrame(
            {atr_col: atr, pct_col: atr / close},
            index=ohlcv.index,
        )
        return FeatureResult(
            features=features,
            feature_names=[atr_col, pct_col],
            method="atr",
        )

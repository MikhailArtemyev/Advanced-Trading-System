"""Statistical feature generators.

Provides features based on statistical properties of price and returns:
- ReturnFeature:       Log returns at multiple horizons
- ZScoreFeature:       Rolling z-score of close price
- HigherMomentFeature: Rolling skewness and kurtosis of returns
- HurstExponentFeature: Hurst exponent (mean-reversion vs trend detector)
- VolatilityFeature:   Annualised realised volatility at multiple windows
"""

import numpy as np
import pandas as pd

from .base_feature import FeatureGenerator, FeatureResult


class ReturnFeature(FeatureGenerator):
    """Log return features at multiple horizons.

    Produces:
        return_{h}d: Log return over h days — ln(close[t] / close[t-h])

    Args:
        horizons: List of lookback horizons in days. Defaults to [1, 5, 10, 20].
    """

    def __init__(self, horizons: list[int] | None = None) -> None:
        self.horizons: list[int] = horizons or [1, 5, 10, 20]

    @property
    def lookback(self) -> int:
        return max(self.horizons) + 1

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        close = ohlcv["close"]
        features = pd.DataFrame(index=ohlcv.index)

        for h in self.horizons:
            features[f"return_{h}d"] = np.log(close / close.shift(h))

        names = list(features.columns)
        return FeatureResult(features=features, feature_names=names, method="returns")


class ZScoreFeature(FeatureGenerator):
    """Rolling z-score of close price.

    z = (close − rolling_mean) / rolling_std

    Extreme z-scores suggest mean-reversion opportunity.
    Returns NaN when rolling std == 0 (constant price series).

    Produces:
        zscore_{window}: Rolling z-score

    Args:
        window: Rolling window for mean and std. Defaults to 20.
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    @property
    def lookback(self) -> int:
        return self.window

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        close = ohlcv["close"]
        mean = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        zscore = (close - mean) / std.replace(0, np.nan)

        col = f"zscore_{self.window}"
        features = pd.DataFrame({col: zscore}, index=ohlcv.index)
        return FeatureResult(features=features, feature_names=[col], method="zscore")


class HigherMomentFeature(FeatureGenerator):
    """Rolling skewness and kurtosis of daily returns.

    Produces:
        skew_{window}: Rolling skewness of simple daily returns
        kurt_{window}: Rolling excess kurtosis of simple daily returns

    For constant returns (zero std), pandas returns skew=0.0 and kurt=-3.0.

    Args:
        window: Rolling window. Defaults to 20.
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    @property
    def lookback(self) -> int:
        return self.window + 1

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        returns = ohlcv["close"].pct_change()
        skew = returns.rolling(self.window).skew()
        kurt = returns.rolling(self.window).kurt()

        skew_col = f"skew_{self.window}"
        kurt_col = f"kurt_{self.window}"
        features = pd.DataFrame({skew_col: skew, kurt_col: kurt}, index=ohlcv.index)
        return FeatureResult(
            features=features,
            feature_names=[skew_col, kurt_col],
            method="higher_moments",
        )


class HurstExponentFeature(FeatureGenerator):
    """Hurst Exponent estimation via rescaled range (R/S) analysis.

    Interpretation:
        H < 0.5: Mean-reverting process
        H = 0.5: Random walk (no memory)
        H > 0.5: Trending / persistent process

    Output is clipped to [0.0, 1.0]. NaN for bars before the first full window.

    Produces:
        hurst_{window}: Estimated Hurst exponent

    Args:
        window: Rolling window for R/S analysis. Must be >= 20. Defaults to 100.

    Raises:
        ValueError: If window < 20.
    """

    def __init__(self, window: int = 100) -> None:
        if window < 20:
            raise ValueError(f"window must be >= 20, got {window}")
        self.window = window

    @property
    def lookback(self) -> int:
        return self.window

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        close = ohlcv["close"]
        hurst_values = pd.Series(np.nan, index=ohlcv.index, dtype=float)

        for i in range(self.window, len(close)):
            window_data = close.iloc[i - self.window : i].values
            hurst_values.iloc[i] = self._estimate_hurst(window_data)

        col = f"hurst_{self.window}"
        features = pd.DataFrame({col: hurst_values}, index=ohlcv.index)
        return FeatureResult(features=features, feature_names=[col], method="hurst")

    def _estimate_hurst(self, prices: np.ndarray) -> float:
        """Estimate Hurst exponent using R/S analysis.

        Args:
            prices: Array of price values (length == window).

        Returns:
            Estimated Hurst exponent in [0.0, 1.0].
            Returns 0.5 (neutral) when estimation is not possible.
        """
        returns = np.diff(np.log(prices))
        n = len(returns)

        if n < 10:
            return 0.5

        max_k = min(n // 2, 50)
        sizes: list[int] = []
        rs_values: list[float] = []

        for k in range(10, max_k + 1):
            num_chunks = n // k
            if num_chunks < 1:
                continue

            rs_chunk: list[float] = []
            for c in range(num_chunks):
                chunk = returns[c * k : (c + 1) * k]
                mean_chunk = chunk.mean()
                deviate = np.cumsum(chunk - mean_chunk)
                r = deviate.max() - deviate.min()
                s = chunk.std(ddof=1)
                if s > 0:
                    rs_chunk.append(r / s)

            if rs_chunk:
                sizes.append(k)
                rs_values.append(float(np.mean(rs_chunk)))

        if len(sizes) < 2:
            return 0.5

        slope = np.polyfit(np.log(sizes), np.log(rs_values), 1)[0]
        return float(np.clip(slope, 0.0, 1.0))


class VolatilityFeature(FeatureGenerator):
    """Annualised realised volatility at multiple windows.

    Computes rolling std of simple daily returns, annualised by sqrt(252).

    Produces:
        vol_{window}:             Annualised realised volatility for each window
        vol_ratio_{short}_{long}: vol_short / vol_long (regime indicator)
            Only produced when len(windows) >= 2.

    Args:
        windows: List of rolling windows. Defaults to [5, 20, 60].
    """

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows: list[int] = windows or [5, 20, 60]

    @property
    def lookback(self) -> int:
        return max(self.windows) + 1

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        returns = ohlcv["close"].pct_change()
        features = pd.DataFrame(index=ohlcv.index)

        for w in self.windows:
            features[f"vol_{w}"] = returns.rolling(w).std() * np.sqrt(252)

        if len(self.windows) >= 2:
            short_w = min(self.windows)
            long_w = max(self.windows)
            long_vol = features[f"vol_{long_w}"].replace(0, np.nan)
            features[f"vol_ratio_{short_w}_{long_w}"] = features[f"vol_{short_w}"] / long_vol

        names = list(features.columns)
        return FeatureResult(features=features, feature_names=names, method="volatility")

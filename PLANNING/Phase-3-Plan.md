# Phase 3: Machine Learning & Advanced Analytics
## Step-by-Step Implementation Guide

**Duration:** 8 Weeks (Months 5-6)
**Goal:** ML-driven signal generation with proper validation (CPCV), feature engineering, regime detection, experiment tracking, and deflated performance metrics

**Prerequisites:** Phase 2 complete — working multi-asset backtester with position sizing, risk management, portfolio optimization, walk-forward analysis, and 600+ tests

---

## Overview

```
Week 1-2: Feature Engineering Pipeline (Technical Indicators & Statistical Features)
Week 3-4: ML Model Integration (XGBoost/LightGBM Signal Generation)
Week 5:   CPCV Validation Framework (Combinatorial Purged Cross-Validation)
Week 6:   Deflated Sharpe Ratio & Advanced Performance Metrics
Week 7:   Regime Detection (Hidden Markov Model) & Experiment Tracking (MLflow)
Week 8:   Integration Testing, Validation & Documentation
```

---

## What Changes From Phase 2

Phase 2 established multi-asset portfolio management with risk controls and optimization. Phase 3 adds machine learning signal generation with proper financial ML validation:

| Component | Phase 2 State | Phase 3 Target |
|-----------|--------------|----------------|
| Strategy | SMA crossover (rule-based only) | ML model signals (XGBoost/LightGBM) alongside rule-based |
| Features | Raw OHLCV prices only | 30+ engineered features (technical indicators, statistical, cross-asset) |
| Validation | Walk-forward (single-path) | CPCV — combinatorial purged cross-validation (multi-path) |
| Performance | Sharpe, Sortino, Calmar, drawdown | + Deflated Sharpe Ratio, probabilistic Sharpe, trial-aware metrics |
| Regime | None | HMM-based regime detection (bull/bear/sideways) |
| Tracking | Print to console | MLflow experiment tracking with parameter/metric logging |
| Config | YAML for sizing/risk/optimization | + ML model config, feature config, regime config sections |
| Testing | 604 tests | Target 800+ with ML, feature, and validation coverage |

### New Project Structure (additions to Phase 2)

```
src/
├── ... (existing Phase 1-2 modules)
├── features/
│   ├── __init__.py
│   ├── base_feature.py          # Abstract feature interface
│   ├── technical.py             # Technical indicator features (SMA, RSI, MACD, BB, ATR)
│   ├── statistical.py           # Statistical features (z-score, rolling skew/kurt, hurst)
│   └── pipeline.py              # Feature pipeline that composes feature sets
├── ml/
│   ├── __init__.py
│   ├── base_model.py            # Abstract ML model interface
│   ├── xgboost_model.py         # XGBoost signal model
│   ├── lightgbm_model.py        # LightGBM signal model
│   └── ml_strategy.py           # Strategy subclass using ML predictions
├── validation/
│   ├── __init__.py
│   ├── cpcv.py                  # Combinatorial Purged Cross-Validation
│   └── deflated_sharpe.py       # Deflated Sharpe Ratio
├── regime/
│   ├── __init__.py
│   └── hmm.py                   # Hidden Markov Model regime detector
├── tracking/
│   ├── __init__.py
│   └── mlflow_tracker.py        # MLflow experiment tracking wrapper
tests/
├── ... (existing Phase 1-2 tests)
├── test_features.py
├── test_ml_model.py
├── test_ml_strategy.py
├── test_cpcv.py
├── test_deflated_sharpe.py
├── test_regime.py
├── test_mlflow_tracker.py
├── test_phase3_integration.py
configs/
├── ... (existing configs)
├── ml_backtest_config.yaml      # Full ML-enabled backtest config
└── feature_config.yaml          # Feature engineering configuration
```

---

## Week 1: Feature Engineering — Technical Indicators

### Step 1.1: Feature Base Class & Technical Indicators
**Time:** Day 1-3

**File: `src/features/base_feature.py`**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class FeatureResult:
    """Result of feature computation for one symbol.

    Attributes:
        features: DataFrame with feature columns, indexed by date
        feature_names: List of column names produced
        method: Name of the feature computation method
    """

    features: pd.DataFrame
    feature_names: list[str]
    method: str


class FeatureGenerator(ABC):
    """Abstract base class for feature generators.

    Each generator produces one or more feature columns from OHLCV data.
    Generators must be stateless — all state lives in the input data.
    """

    @abstractmethod
    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        """Compute features from OHLCV data.

        Args:
            ohlcv: DataFrame with columns: open, high, low, close, volume.
                   Index must be DatetimeIndex.

        Returns:
            FeatureResult with computed feature columns.
            Rows with NaN (from lookback requirements) should be retained
            — the pipeline handles trimming.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def lookback(self) -> int:
        """Minimum number of bars required before first valid output."""
        raise NotImplementedError
```

**File: `src/features/technical.py`**
```python
"""Technical indicator feature generators.

Provides standard technical analysis indicators as features:
- SMAFeature: Simple Moving Average at multiple windows
- RSIFeature: Relative Strength Index
- MACDFeature: Moving Average Convergence Divergence
- BollingerBandFeature: Bollinger Band width and %B
- ATRFeature: Average True Range (normalized)
"""

import numpy as np
import pandas as pd

from .base_feature import FeatureGenerator, FeatureResult


class SMAFeature(FeatureGenerator):
    """Simple Moving Average features at multiple windows.

    Produces:
        - sma_{window}: Raw SMA value
        - sma_{window}_ratio: Close / SMA (mean-reversion signal)
    """

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [5, 10, 20, 50]

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

    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss over `period` bars.

    Produces:
        - rsi_{period}: RSI value (0-100)
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
        rsi = 100 - 100 / (1 + rs)

        features = pd.DataFrame(
            {f"rsi_{self.period}": rsi}, index=ohlcv.index
        )
        return FeatureResult(
            features=features,
            feature_names=[f"rsi_{self.period}"],
            method="rsi",
        )


class MACDFeature(FeatureGenerator):
    """Moving Average Convergence Divergence.

    Produces:
        - macd: MACD line (fast EMA - slow EMA)
        - macd_signal: Signal line (EMA of MACD)
        - macd_histogram: MACD - Signal
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

        features = pd.DataFrame(
            {
                "macd": macd_line,
                "macd_signal": signal_line,
                "macd_histogram": histogram,
            },
            index=ohlcv.index,
        )
        return FeatureResult(
            features=features,
            feature_names=["macd", "macd_signal", "macd_histogram"],
            method="macd",
        )


class BollingerBandFeature(FeatureGenerator):
    """Bollinger Band features.

    Produces:
        - bb_width: (Upper - Lower) / Middle — band width
        - bb_pct_b: (Close - Lower) / (Upper - Lower) — %B position
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
    """Normalized Average True Range.

    Produces:
        - atr_{period}: ATR value
        - atr_{period}_pct: ATR / Close (volatility as % of price)
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
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(self.period).mean()

        features = pd.DataFrame(
            {
                f"atr_{self.period}": atr,
                f"atr_{self.period}_pct": atr / close,
            },
            index=ohlcv.index,
        )
        return FeatureResult(
            features=features,
            feature_names=[f"atr_{self.period}", f"atr_{self.period}_pct"],
            method="atr",
        )
```

**Tasks:**
- [ ] Create `src/features/__init__.py`
- [ ] Implement `FeatureGenerator` abstract base class with `compute()` and `lookback`
- [ ] Implement `FeatureResult` frozen dataclass
- [ ] Implement `SMAFeature` with configurable windows
- [ ] Implement `RSIFeature` with Wilder smoothing
- [ ] Implement `MACDFeature` with EMA-based MACD/signal/histogram
- [ ] Implement `BollingerBandFeature` with width and %B
- [ ] Implement `ATRFeature` with normalized ATR
- [ ] Write unit tests for each feature with known inputs → verify formula output
- [ ] Test RSI boundaries (0 and 100)
- [ ] Test with insufficient data (< lookback) → expect NaNs

**Deliverable:** Five technical indicator feature generators with full test coverage

---

## Week 2: Feature Engineering — Statistical Features & Pipeline

### Step 2.1: Statistical Feature Generators
**Time:** Day 1-3

**File: `src/features/statistical.py`**
```python
"""Statistical feature generators.

Provides features based on statistical properties of returns:
- ReturnFeature: Log returns at multiple horizons
- ZScoreFeature: Rolling z-score of price
- HigherMomentFeature: Rolling skewness and kurtosis
- HurstExponentFeature: Hurst exponent (mean-reversion vs trend)
- VolatilityFeature: Realized volatility at multiple windows
"""

import numpy as np
import pandas as pd

from .base_feature import FeatureGenerator, FeatureResult


class ReturnFeature(FeatureGenerator):
    """Log return features at multiple horizons.

    Produces:
        - return_{h}d: Log return over h days
    """

    def __init__(self, horizons: list[int] | None = None) -> None:
        self.horizons = horizons or [1, 5, 10, 20]

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

    z = (close - rolling_mean) / rolling_std

    Mean-reversion signal: extreme z-scores suggest reversion.

    Produces:
        - zscore_{window}: Rolling z-score
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

        features = pd.DataFrame(
            {f"zscore_{self.window}": zscore}, index=ohlcv.index
        )
        return FeatureResult(
            features=features,
            feature_names=[f"zscore_{self.window}"],
            method="zscore",
        )


class HigherMomentFeature(FeatureGenerator):
    """Rolling skewness and kurtosis of returns.

    Produces:
        - skew_{window}: Rolling skewness of daily returns
        - kurt_{window}: Rolling excess kurtosis of daily returns
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

        features = pd.DataFrame(
            {f"skew_{self.window}": skew, f"kurt_{self.window}": kurt},
            index=ohlcv.index,
        )
        return FeatureResult(
            features=features,
            feature_names=[f"skew_{self.window}", f"kurt_{self.window}"],
            method="higher_moments",
        )


class HurstExponentFeature(FeatureGenerator):
    """Hurst Exponent estimation via rescaled range (R/S) analysis.

    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending

    Produces:
        - hurst_{window}: Estimated Hurst exponent
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
        hurst_values = pd.Series(np.nan, index=ohlcv.index)

        for i in range(self.window, len(close)):
            window_data = close.iloc[i - self.window : i].values
            h = self._estimate_hurst(window_data)
            hurst_values.iloc[i] = h

        features = pd.DataFrame(
            {f"hurst_{self.window}": hurst_values}, index=ohlcv.index
        )
        return FeatureResult(
            features=features,
            feature_names=[f"hurst_{self.window}"],
            method="hurst",
        )

    def _estimate_hurst(self, prices: np.ndarray) -> float:
        """Estimate Hurst exponent using R/S analysis."""
        returns = np.diff(np.log(prices))
        n = len(returns)

        if n < 10:
            return 0.5

        max_k = min(n // 2, 50)
        sizes = []
        rs_values = []

        for k in range(10, max_k + 1):
            num_chunks = n // k
            if num_chunks < 1:
                continue

            rs_chunk = []
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
                rs_values.append(np.mean(rs_chunk))

        if len(sizes) < 2:
            return 0.5

        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)

        slope = np.polyfit(log_sizes, log_rs, 1)[0]
        return float(np.clip(slope, 0.0, 1.0))


class VolatilityFeature(FeatureGenerator):
    """Realized volatility at multiple windows.

    Produces:
        - vol_{window}: Annualized realized volatility
        - vol_ratio_{short}_{long}: Volatility ratio (short / long)
    """

    def __init__(
        self, windows: list[int] | None = None
    ) -> None:
        self.windows = windows or [5, 20, 60]

    @property
    def lookback(self) -> int:
        return max(self.windows) + 1

    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        returns = ohlcv["close"].pct_change()
        features = pd.DataFrame(index=ohlcv.index)

        for w in self.windows:
            vol = returns.rolling(w).std() * np.sqrt(252)
            features[f"vol_{w}"] = vol

        # Volatility ratio: short / long
        if len(self.windows) >= 2:
            short_w = min(self.windows)
            long_w = max(self.windows)
            features[f"vol_ratio_{short_w}_{long_w}"] = (
                features[f"vol_{short_w}"]
                / features[f"vol_{long_w}"].replace(0, np.nan)
            )

        names = list(features.columns)
        return FeatureResult(features=features, feature_names=names, method="volatility")
```

**Tasks:**
- [ ] Implement `ReturnFeature` with configurable horizons
- [ ] Implement `ZScoreFeature` with rolling z-score
- [ ] Implement `HigherMomentFeature` (skewness & kurtosis)
- [ ] Implement `HurstExponentFeature` with R/S analysis
- [ ] Implement `VolatilityFeature` with multiple windows and vol ratio
- [ ] Write unit tests: verify return calculation against manual
- [ ] Test z-score with known mean/std series → verify output
- [ ] Test Hurst exponent: trending series → H > 0.5, mean-reverting → H < 0.5
- [ ] Test edge cases: constant prices, single bar

**Deliverable:** Five statistical feature generators with full test coverage

---

### Step 2.2: Feature Pipeline
**Time:** Day 3-5

**File: `src/features/pipeline.py`**
```python
"""Feature pipeline that composes multiple feature generators.

Orchestrates feature computation, handles NaN trimming from lookback
windows, and produces a clean feature matrix ready for ML models.
"""

from dataclasses import dataclass, field

import pandas as pd

from .base_feature import FeatureGenerator, FeatureResult


@dataclass
class PipelineResult:
    """Result of running the full feature pipeline.

    Attributes:
        features: Clean feature DataFrame (NaN rows trimmed)
        feature_names: List of all feature column names
        target: Optional target series (forward returns)
        rows_dropped: Number of rows trimmed due to NaN from lookback
    """

    features: pd.DataFrame
    feature_names: list[str]
    target: pd.Series | None = None
    rows_dropped: int = 0


class FeaturePipeline:
    """Composes multiple FeatureGenerators into a single pipeline.

    Usage:
        pipeline = FeaturePipeline([SMAFeature(), RSIFeature()])
        result = pipeline.run(ohlcv_data, target_horizon=5)

    The pipeline:
    1. Runs each generator on the input OHLCV data
    2. Concatenates all feature columns
    3. Optionally creates a forward-return target column
    4. Drops rows with NaN (from lookback windows)
    5. Returns clean features + target ready for model training
    """

    def __init__(self, generators: list[FeatureGenerator]) -> None:
        if not generators:
            raise ValueError("At least one FeatureGenerator is required")
        self.generators = generators

    @property
    def total_lookback(self) -> int:
        """Maximum lookback across all generators."""
        return max(g.lookback for g in self.generators)

    def run(
        self,
        ohlcv: pd.DataFrame,
        target_horizon: int = 5,
        target_type: str = "return",
    ) -> PipelineResult:
        """Run all generators and produce clean feature matrix.

        Args:
            ohlcv: OHLCV DataFrame with DatetimeIndex
            target_horizon: Forward-looking period for target variable (days)
            target_type: Target type — "return" for forward log return,
                        "direction" for binary up/down (1/0)

        Returns:
            PipelineResult with clean features and target
        """
        all_features = pd.DataFrame(index=ohlcv.index)
        all_names: list[str] = []

        for gen in self.generators:
            result = gen.compute(ohlcv)
            all_features = pd.concat(
                [all_features, result.features], axis=1
            )
            all_names.extend(result.feature_names)

        # Create target (forward return)
        close = ohlcv["close"]
        if target_type == "return":
            target = close.shift(-target_horizon) / close - 1
        elif target_type == "direction":
            forward_return = close.shift(-target_horizon) / close - 1
            target = (forward_return > 0).astype(int)
        else:
            msg = f"Unknown target_type: {target_type}"
            raise ValueError(msg)

        target.name = "target"

        # Drop rows where any feature or target is NaN
        combined = pd.concat([all_features, target], axis=1)
        initial_rows = len(combined)
        combined = combined.dropna()
        rows_dropped = initial_rows - len(combined)

        clean_features = combined[all_names]
        clean_target = combined["target"]

        return PipelineResult(
            features=clean_features,
            feature_names=all_names,
            target=clean_target,
            rows_dropped=rows_dropped,
        )

    def compute_features_only(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute features without target (for live prediction).

        Returns the latest row of features after dropping NaN.
        """
        all_features = pd.DataFrame(index=ohlcv.index)

        for gen in self.generators:
            result = gen.compute(ohlcv)
            all_features = pd.concat(
                [all_features, result.features], axis=1
            )

        return all_features.dropna()
```

**Tasks:**
- [ ] Implement `PipelineResult` dataclass
- [ ] Implement `FeaturePipeline` with `run()` and `compute_features_only()`
- [ ] Support "return" and "direction" target types
- [ ] Handle NaN trimming correctly (no look-ahead bias in target)
- [ ] Verify `total_lookback` returns max across generators
- [ ] Write tests: pipeline with 3 generators → verify column count
- [ ] Test forward return target against manual calculation
- [ ] Test direction target: 1 for up, 0 for down
- [ ] Test empty pipeline raises ValueError
- [ ] Test `compute_features_only` returns features without target

**Deliverable:** Complete feature pipeline that produces ML-ready feature matrix

---

## Week 3: ML Model Integration — Base & XGBoost

### Step 3.1: ML Model Base Class & XGBoost Implementation
**Time:** Day 1-3

**File: `src/ml/base_model.py`**
```python
"""Abstract base class for ML signal models.

ML models in this system:
1. Are trained on historical feature+target data
2. Produce signal predictions (probability or direction)
3. Must support serialization (save/load)
4. Report feature importance for interpretability
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ModelPrediction:
    """Result of a model prediction.

    Attributes:
        signal: Predicted signal strength (-1.0 to 1.0)
            Positive = long, negative = short, near 0 = no action
        confidence: Model confidence (0.0 to 1.0)
        features_used: Number of features in the input
    """

    signal: float
    confidence: float
    features_used: int


@dataclass
class TrainResult:
    """Result of model training.

    Attributes:
        train_score: In-sample score (accuracy or AUC)
        val_score: Validation score (if validation set provided)
        feature_importance: Feature name → importance mapping
        n_train_samples: Number of training samples
        n_features: Number of features
    """

    train_score: float
    val_score: float | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    n_train_samples: int = 0
    n_features: int = 0


class MLModel(ABC):
    """Abstract base class for ML signal generation models."""

    @abstractmethod
    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | None = None,
    ) -> TrainResult:
        """Train the model on feature+target data.

        Args:
            features: Training feature matrix
            target: Training target series
            val_features: Optional validation features
            val_target: Optional validation target

        Returns:
            TrainResult with scores and feature importance
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> list[ModelPrediction]:
        """Generate predictions for each row of features.

        Args:
            features: Feature matrix (one or more rows)

        Returns:
            List of ModelPrediction, one per row
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save trained model to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load trained model from disk."""
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance mapping."""
        raise NotImplementedError
```

**File: `src/ml/xgboost_model.py`**
```python
"""XGBoost-based signal generation model.

Uses gradient boosted trees for binary classification
(up/down direction) or regression (forward return prediction).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .base_model import MLModel, ModelPrediction, TrainResult


class XGBoostSignalModel(MLModel):
    """XGBoost signal model for trading.

    Supports both classification (direction prediction) and
    regression (return prediction) modes.

    Attributes:
        params: XGBoost parameters
        n_estimators: Number of boosting rounds
        mode: "classification" or "regression"
        model: Trained XGBoost model (None before training)
        feature_names: Feature names from training data
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        mode: str = "classification",
        random_state: int = 42,
        **extra_params: Any,
    ) -> None:
        if mode not in ("classification", "regression"):
            raise ValueError(f"mode must be classification or regression, got {mode}")

        self.mode = mode
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.params: dict[str, Any] = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            **extra_params,
        }

        if mode == "classification":
            self.params["objective"] = "binary:logistic"
            self.params["eval_metric"] = "auc"
        else:
            self.params["objective"] = "reg:squarederror"
            self.params["eval_metric"] = "rmse"

        self.model: xgb.XGBClassifier | xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | None = None,
    ) -> TrainResult:
        self.feature_names = list(features.columns)

        if self.mode == "classification":
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators, **self.params
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators, **self.params
            )

        eval_set = None
        if val_features is not None and val_target is not None:
            eval_set = [(val_features, val_target)]

        self.model.fit(
            features,
            target,
            eval_set=eval_set,
            verbose=False,
        )

        train_score = float(self.model.score(features, target))
        val_score = None
        if val_features is not None and val_target is not None:
            val_score = float(self.model.score(val_features, val_target))

        importance = self.get_feature_importance()

        return TrainResult(
            train_score=train_score,
            val_score=val_score,
            feature_importance=importance,
            n_train_samples=len(features),
            n_features=len(self.feature_names),
        )

    def predict(self, features: pd.DataFrame) -> list[ModelPrediction]:
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        predictions = []

        if self.mode == "classification":
            probas = self.model.predict_proba(features)
            for i in range(len(features)):
                prob_up = float(probas[i, 1])
                signal = (prob_up - 0.5) * 2  # Map [0,1] -> [-1,1]
                confidence = abs(prob_up - 0.5) * 2
                predictions.append(
                    ModelPrediction(
                        signal=signal,
                        confidence=confidence,
                        features_used=features.shape[1],
                    )
                )
        else:
            raw_preds = self.model.predict(features)
            for i in range(len(features)):
                pred = float(raw_preds[i])
                signal = float(np.clip(pred * 100, -1.0, 1.0))
                confidence = min(abs(pred) * 50, 1.0)
                predictions.append(
                    ModelPrediction(
                        signal=signal,
                        confidence=confidence,
                        features_used=features.shape[1],
                    )
                )

        return predictions

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("No trained model to save")
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        if self.mode == "classification":
            self.model = xgb.XGBClassifier()
        else:
            self.model = xgb.XGBRegressor()
        self.model.load_model(str(path))

    def get_feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance, strict=True))
```

**Tasks:**
- [ ] Implement `MLModel` abstract base class with train/predict/save/load
- [ ] Implement `ModelPrediction` and `TrainResult` dataclasses
- [ ] Implement `XGBoostSignalModel` with classification and regression modes
- [ ] Signal mapping: probability → [-1, 1] range
- [ ] Write tests: train on synthetic data → predict → verify signal range
- [ ] Test save/load round-trip
- [ ] Test feature importance returns correct feature names
- [ ] Test untrained model raises RuntimeError on predict
- [ ] Add `xgboost` to `requirements.txt`

**Deliverable:** XGBoost signal model with training, prediction, and serialization

---

## Week 4: ML Strategy & LightGBM

### Step 4.1: LightGBM Model
**Time:** Day 1-2

**File: `src/ml/lightgbm_model.py`**
```python
"""LightGBM-based signal generation model.

Same interface as XGBoostSignalModel but uses LightGBM backend.
LightGBM is often faster for large datasets and handles
categorical features natively.
"""

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base_model import MLModel, ModelPrediction, TrainResult


class LightGBMSignalModel(MLModel):
    """LightGBM signal model for trading.

    Attributes:
        params: LightGBM parameters
        n_estimators: Number of boosting rounds
        mode: "classification" or "regression"
        model: Trained LightGBM model (None before training)
        feature_names: Feature names from training data
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        mode: str = "classification",
        random_state: int = 42,
        **extra_params: Any,
    ) -> None:
        if mode not in ("classification", "regression"):
            raise ValueError(f"mode must be classification or regression, got {mode}")

        self.mode = mode
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.params: dict[str, Any] = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbose": -1,
            **extra_params,
        }

        self.model: lgb.LGBMClassifier | lgb.LGBMRegressor | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | None = None,
    ) -> TrainResult:
        self.feature_names = list(features.columns)

        if self.mode == "classification":
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators, **self.params
            )
        else:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators, **self.params
            )

        eval_set = None
        if val_features is not None and val_target is not None:
            eval_set = [(val_features, val_target)]

        self.model.fit(
            features,
            target,
            eval_set=eval_set,
        )

        train_score = float(self.model.score(features, target))
        val_score = None
        if val_features is not None and val_target is not None:
            val_score = float(self.model.score(val_features, val_target))

        importance = self.get_feature_importance()

        return TrainResult(
            train_score=train_score,
            val_score=val_score,
            feature_importance=importance,
            n_train_samples=len(features),
            n_features=len(self.feature_names),
        )

    def predict(self, features: pd.DataFrame) -> list[ModelPrediction]:
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        predictions = []

        if self.mode == "classification":
            probas = self.model.predict_proba(features)
            for i in range(len(features)):
                prob_up = float(probas[i, 1])
                signal = (prob_up - 0.5) * 2
                confidence = abs(prob_up - 0.5) * 2
                predictions.append(
                    ModelPrediction(
                        signal=signal,
                        confidence=confidence,
                        features_used=features.shape[1],
                    )
                )
        else:
            raw_preds = self.model.predict(features)
            for i in range(len(features)):
                pred = float(raw_preds[i])
                signal = float(np.clip(pred * 100, -1.0, 1.0))
                confidence = min(abs(pred) * 50, 1.0)
                predictions.append(
                    ModelPrediction(
                        signal=signal,
                        confidence=confidence,
                        features_used=features.shape[1],
                    )
                )

        return predictions

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("No trained model to save")
        self.model.booster_.save_model(str(path))

    def load(self, path: str | Path) -> None:
        if self.mode == "classification":
            self.model = lgb.LGBMClassifier()
        else:
            self.model = lgb.LGBMRegressor()
        self.model.booster_ = lgb.Booster(model_file=str(path))

    def get_feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance, strict=True))
```

**Tasks:**
- [ ] Implement `LightGBMSignalModel` mirroring XGBoost interface
- [ ] Classification and regression modes
- [ ] Write tests: train/predict/save/load round-trip
- [ ] Test feature importance output
- [ ] Add `lightgbm` to `requirements.txt`

**Deliverable:** LightGBM signal model with identical interface to XGBoost

---

### Step 4.2: ML-Based Strategy
**Time:** Day 2-5

**File: `src/ml/ml_strategy.py`**
```python
"""ML-based trading strategy.

Uses a trained MLModel + FeaturePipeline to generate signals.
At each bar, computes features from recent data and passes them
to the model for prediction. The model's signal and confidence
are used to generate LONG/SHORT/EXIT events.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from ..data.data_handler import DataHandler
from ..events.event import SignalEvent, SignalType
from ..features.pipeline import FeaturePipeline
from ..strategy.base_strategy import Strategy
from .base_model import MLModel


class MLStrategy(Strategy):
    """Strategy that generates signals from ML model predictions.

    The strategy:
    1. Collects enough bars for feature computation (lookback)
    2. Computes features using the FeaturePipeline
    3. Passes features to the MLModel for prediction
    4. Maps model prediction → SignalEvent

    Signal Mapping:
        prediction.signal > threshold  → LONG (strength = confidence)
        prediction.signal < -threshold → SHORT (strength = confidence)
        Otherwise, if in position       → EXIT

    Attributes:
        pipeline: Feature computation pipeline
        model: Trained ML model for prediction
        signal_threshold: Minimum |signal| to generate entry
        exit_threshold: |signal| below which to exit
        lookback_bars: Number of bars to fetch for features
    """

    def __init__(
        self,
        symbols: list[str],
        pipeline: FeaturePipeline,
        model: MLModel,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(symbols, parameters)
        self.pipeline = pipeline
        self.model = model

        self.signal_threshold: float = self.parameters.get("signal_threshold", 0.1)
        self.exit_threshold: float = self.parameters.get("exit_threshold", 0.0)
        self.lookback_bars: int = self.parameters.get(
            "lookback_bars", pipeline.total_lookback + 50
        )

    def calculate_signals(
        self,
        timestamp: datetime,
        data_handler: DataHandler,
    ) -> list[SignalEvent]:
        signals = []

        for symbol in self.symbols:
            signal = self._generate_ml_signal(timestamp, symbol, data_handler)
            if signal is not None:
                signals.append(signal)

        return signals

    def _generate_ml_signal(
        self,
        timestamp: datetime,
        symbol: str,
        data_handler: DataHandler,
    ) -> SignalEvent | None:
        bars = data_handler.get_latest_bars(symbol, self.lookback_bars)

        if len(bars) < self.pipeline.total_lookback + 1:
            return None

        # Compute features for the latest bar
        feature_df = self.pipeline.compute_features_only(bars)

        if feature_df.empty:
            return None

        # Predict using the latest row
        latest_features = feature_df.iloc[[-1]]
        predictions = self.model.predict(latest_features)
        pred = predictions[0]

        current_pos = self.current_positions.get(symbol, 0)

        # Signal logic
        if pred.signal > self.signal_threshold and current_pos <= 0:
            return self._create_signal(
                timestamp, symbol, SignalType.LONG, strength=pred.confidence
            )
        elif pred.signal < -self.signal_threshold and current_pos >= 0:
            if current_pos > 0:
                return self._create_signal(
                    timestamp, symbol, SignalType.EXIT, strength=1.0
                )
            # SHORT signals only if enabled
            return self._create_signal(
                timestamp, symbol, SignalType.SHORT, strength=pred.confidence
            )
        elif abs(pred.signal) < self.exit_threshold and current_pos != 0:
            return self._create_signal(
                timestamp, symbol, SignalType.EXIT, strength=1.0
            )

        return None
```

**Tasks:**
- [ ] Implement `MLStrategy` extending `Strategy` base class
- [ ] Feature computation at each bar using pipeline
- [ ] Signal threshold logic for LONG/SHORT/EXIT
- [ ] Configurable via parameters dict
- [ ] Write tests: mock model → verify signal generation
- [ ] Test with real pipeline + trained model end-to-end
- [ ] Test insufficient data (< lookback) → no signal
- [ ] Test position awareness (no duplicate entries)

**Deliverable:** ML strategy that integrates feature pipeline and model predictions into event-driven backtest

---

## Week 5: CPCV Validation Framework

### Step 5.1: Combinatorial Purged Cross-Validation
**Time:** Day 1-5

CPCV (from Marcos López de Prado, "Advances in Financial Machine Learning") provides proper validation for financial ML by:
1. Generating multiple train/test paths from combinatorial splits
2. Purging overlapping samples between train/test to prevent leakage
3. Adding an embargo period after each purged boundary

**File: `src/validation/cpcv.py`**
```python
"""Combinatorial Purged Cross-Validation (CPCV).

Implements the CPCV method from López de Prado (2018) for
proper validation of financial ML models.

Key differences from standard k-fold:
1. Combinatorial: generates C(n_splits, n_test_splits) paths
2. Purged: removes samples near train/test boundaries
3. Embargo: extra gap after purge to prevent information leakage

Reference: Advances in Financial Machine Learning, Chapter 12.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class CPCVSplit:
    """One train/test split from CPCV.

    Attributes:
        train_indices: Array of training sample indices
        test_indices: Array of test sample indices
        path_id: Identifier for this combinatorial path
    """

    train_indices: np.ndarray
    test_indices: np.ndarray
    path_id: int


@dataclass
class CPCVResult:
    """Result of running CPCV validation.

    Attributes:
        scores: Score per split (e.g., accuracy, Sharpe)
        mean_score: Average score across all paths
        std_score: Standard deviation of scores
        n_paths: Number of combinatorial paths tested
        predictions: Concatenated out-of-sample predictions (optional)
    """

    scores: list[float]
    mean_score: float
    std_score: float
    n_paths: int
    predictions: pd.Series | None = None


class CPCVValidator:
    """Combinatorial Purged Cross-Validation.

    Generates combinatorial train/test splits with purging and embargo
    for proper financial ML validation.

    Attributes:
        n_splits: Number of groups to split data into
        n_test_splits: Number of groups to use as test in each path
        purge_window: Number of samples to remove at train/test boundary
        embargo_pct: Fraction of test size to embargo after purge
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be < n_splits")
        if n_splits < 3:
            raise ValueError("n_splits must be >= 3")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def get_splits(self, n_samples: int) -> list[CPCVSplit]:
        """Generate all combinatorial purged splits.

        Args:
            n_samples: Total number of samples

        Returns:
            List of CPCVSplit objects
        """
        group_size = n_samples // self.n_splits
        groups: list[np.ndarray] = []

        for i in range(self.n_splits):
            start = i * group_size
            end = start + group_size if i < self.n_splits - 1 else n_samples
            groups.append(np.arange(start, end))

        # Generate all combinations of test groups
        test_combos = list(combinations(range(self.n_splits), self.n_test_splits))

        splits = []
        for path_id, test_group_ids in enumerate(test_combos):
            train_groups = [
                i for i in range(self.n_splits) if i not in test_group_ids
            ]

            test_indices = np.concatenate([groups[i] for i in test_group_ids])
            train_indices = np.concatenate([groups[i] for i in train_groups])

            # Purge: remove train samples near test boundaries
            train_indices = self._purge(train_indices, test_indices, n_samples)

            # Embargo: remove extra train samples after purge
            train_indices = self._embargo(train_indices, test_indices)

            splits.append(
                CPCVSplit(
                    train_indices=train_indices,
                    test_indices=test_indices,
                    path_id=path_id,
                )
            )

        return splits

    def _purge(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Remove train samples within purge_window of test boundaries."""
        test_set = set(test_idx)
        purge_set: set[int] = set()

        for t_idx in test_idx:
            for offset in range(1, self.purge_window + 1):
                purge_set.add(t_idx - offset)
                purge_set.add(t_idx + offset)

        purge_set -= test_set
        return np.array([i for i in train_idx if i not in purge_set])

    def _embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> np.ndarray:
        """Apply embargo: remove train samples right after test blocks."""
        if len(test_idx) == 0:
            return train_idx

        embargo_size = max(1, int(len(test_idx) * self.embargo_pct))
        test_max = int(test_idx.max())
        embargo_indices = set(range(test_max + 1, test_max + 1 + embargo_size))

        return np.array([i for i in train_idx if i not in embargo_indices])

    @property
    def n_paths(self) -> int:
        """Number of combinatorial paths = C(n_splits, n_test_splits)."""
        from math import comb

        return comb(self.n_splits, self.n_test_splits)
```

**Tasks:**
- [ ] Implement `CPCVValidator` with configurable n_splits, n_test_splits
- [ ] Implement purging to remove train samples near test boundaries
- [ ] Implement embargo after test blocks
- [ ] Implement `get_splits()` returning all combinatorial paths
- [ ] Verify C(n_splits, n_test_splits) paths are generated
- [ ] Write tests: verify no train/test overlap
- [ ] Test purge removes correct indices
- [ ] Test embargo removes correct indices
- [ ] Test with known data → verify split boundaries
- [ ] Test edge cases: small n_samples, large purge_window

**Deliverable:** CPCV validator producing properly purged combinatorial splits

---

## Week 6: Deflated Sharpe Ratio & Advanced Metrics

### Step 6.1: Deflated Sharpe Ratio
**Time:** Day 1-3

The Deflated Sharpe Ratio (López de Prado, 2014) adjusts the Sharpe ratio for the number of trials conducted. It answers: "What is the probability that this Sharpe ratio is a false positive?"

**File: `src/validation/deflated_sharpe.py`**
```python
"""Deflated Sharpe Ratio implementation.

Adjusts the Sharpe ratio for multiple testing (selection bias).
When you test N strategies and pick the best, the observed Sharpe
is inflated. DSR estimates the probability that the best Sharpe
is a false discovery.

Reference: Bailey & López de Prado (2014),
"The Deflated Sharpe Ratio: Correcting for Selection Bias,
Backtest Overfitting, and Non-Normality"
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DSRResult:
    """Result of Deflated Sharpe Ratio calculation.

    Attributes:
        observed_sharpe: The raw Sharpe ratio being tested
        deflated_sharpe: Sharpe adjusted for multiple testing
        p_value: Probability that observed Sharpe is a false positive
        expected_max_sharpe: Expected maximum Sharpe under null hypothesis
        n_trials: Number of independent trials (strategies tested)
        is_significant: True if p_value < significance_level
    """

    observed_sharpe: float
    deflated_sharpe: float
    p_value: float
    expected_max_sharpe: float
    n_trials: int
    is_significant: bool


def expected_max_sharpe(
    n_trials: int,
    sharpe_std: float = 1.0,
) -> float:
    """Expected maximum Sharpe ratio under null hypothesis.

    E[max(SR)] ≈ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e))
    where γ ≈ 0.5772 (Euler-Mascheroni constant)

    Args:
        n_trials: Number of independent trials
        sharpe_std: Standard deviation of Sharpe ratios under null

    Returns:
        Expected maximum Sharpe ratio
    """
    if n_trials <= 1:
        return 0.0

    euler_mascheroni = 0.5772156649

    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))

    e_max = (1 - euler_mascheroni) * z1 + euler_mascheroni * z2
    return float(e_max * sharpe_std)


def probabilistic_sharpe(
    observed_sr: float,
    benchmark_sr: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio — Prob(SR > benchmark).

    PSR = Φ((SR - SR*) / σ(SR))
    where σ(SR) accounts for non-normality of returns.

    Args:
        observed_sr: Observed Sharpe ratio
        benchmark_sr: Benchmark Sharpe ratio to beat
        n_observations: Number of return observations
        skewness: Skewness of returns
        kurtosis: Kurtosis of returns (3.0 = normal)

    Returns:
        Probability that true Sharpe exceeds benchmark
    """
    if n_observations < 2:
        return 0.5

    # Standard error of Sharpe ratio (Lo, 2002) with non-normal adjustment
    se_sr = np.sqrt(
        (1 + 0.5 * observed_sr**2 - skewness * observed_sr
         + ((kurtosis - 3) / 4) * observed_sr**2)
        / (n_observations - 1)
    )

    if se_sr <= 0:
        return 0.5

    z = (observed_sr - benchmark_sr) / se_sr
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_observations: int,
    sharpe_std: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    significance_level: float = 0.05,
) -> DSRResult:
    """Calculate the Deflated Sharpe Ratio.

    Tests whether the observed Sharpe ratio is significant after
    accounting for the number of strategies tried (selection bias).

    Args:
        observed_sr: Observed annualized Sharpe ratio
        n_trials: Number of independent strategies/configurations tested
        n_observations: Number of return observations
        sharpe_std: Std dev of Sharpe ratios under null
        skewness: Return series skewness
        kurtosis: Return series kurtosis (3.0 = normal)
        significance_level: p-value threshold for significance

    Returns:
        DSRResult with deflated Sharpe, p-value, and significance
    """
    e_max_sr = expected_max_sharpe(n_trials, sharpe_std)
    p_value = probabilistic_sharpe(
        observed_sr, e_max_sr, n_observations, skewness, kurtosis
    )

    deflated_sr = observed_sr - e_max_sr

    return DSRResult(
        observed_sharpe=observed_sr,
        deflated_sharpe=deflated_sr,
        p_value=p_value,
        expected_max_sharpe=e_max_sr,
        n_trials=n_trials,
        is_significant=p_value > (1 - significance_level),
    )
```

**Tasks:**
- [ ] Implement `expected_max_sharpe` with Euler-Mascheroni approximation
- [ ] Implement `probabilistic_sharpe` with non-normality adjustment
- [ ] Implement `deflated_sharpe_ratio` combining both
- [ ] Write tests: 1 trial → no deflation
- [ ] Test: 1000 trials → significant deflation
- [ ] Test PSR with known values from Bailey & López de Prado paper
- [ ] Test edge cases: n_observations < 2, n_trials = 1
- [ ] Test skewness and kurtosis adjustments affect result
- [ ] Verify is_significant flag at 5% level

**Deliverable:** Deflated Sharpe Ratio with probabilistic Sharpe ratio calculation

---

## Week 7: Regime Detection & Experiment Tracking

### Step 7.1: Hidden Markov Model Regime Detection
**Time:** Day 1-3

**File: `src/regime/hmm.py`**
```python
"""Hidden Markov Model regime detection.

Identifies market regimes (bull/bear/sideways) from return
and volatility characteristics. Useful for:
1. Regime-conditional strategy selection
2. Position sizing adjustment by regime
3. Risk management (tighter limits in bear regimes)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class RegimeResult:
    """Result of regime detection.

    Attributes:
        regimes: Series mapping dates → regime labels (0, 1, 2, ...)
        regime_names: Mapping from regime number to descriptive name
        transition_matrix: Regime transition probability matrix
        regime_means: Mean return per regime
        regime_vols: Volatility per regime
        current_regime: Most recent detected regime
    """

    regimes: pd.Series
    regime_names: dict[int, str] = field(default_factory=dict)
    transition_matrix: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    regime_means: dict[int, float] = field(default_factory=dict)
    regime_vols: dict[int, float] = field(default_factory=dict)
    current_regime: int = 0


class RegimeDetector:
    """HMM-based market regime detection.

    Fits a Gaussian HMM to return and volatility observations,
    then labels each period with a regime.

    Regimes are automatically labeled by their mean return:
        - Highest mean → "bull"
        - Lowest mean → "bear"
        - Middle → "sideways"

    Attributes:
        n_regimes: Number of regimes to detect (default 3)
        vol_window: Window for rolling volatility feature
        n_iter: Maximum EM iterations for HMM fitting
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        n_regimes: int = 3,
        vol_window: int = 20,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        if n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")
        self.n_regimes = n_regimes
        self.vol_window = vol_window
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: GaussianHMM | None = None

    def fit_predict(self, ohlcv: pd.DataFrame) -> RegimeResult:
        """Fit HMM and predict regimes for the entire series.

        Args:
            ohlcv: OHLCV DataFrame with DatetimeIndex

        Returns:
            RegimeResult with regime labels, statistics, and names
        """
        observations = self._build_observations(ohlcv)
        observations = observations.dropna()

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(observations.values)

        hidden_states = self.model.predict(observations.values)
        regimes = pd.Series(hidden_states, index=observations.index, name="regime")

        # Label regimes by mean return
        returns = ohlcv["close"].pct_change().reindex(observations.index)
        regime_means: dict[int, float] = {}
        regime_vols: dict[int, float] = {}

        for r in range(self.n_regimes):
            mask = regimes == r
            regime_means[r] = float(returns[mask].mean()) if mask.any() else 0.0
            regime_vols[r] = float(returns[mask].std()) if mask.any() else 0.0

        # Auto-label
        sorted_regimes = sorted(regime_means.keys(), key=lambda x: regime_means[x])
        names = {}
        if self.n_regimes == 2:
            names[sorted_regimes[0]] = "bear"
            names[sorted_regimes[1]] = "bull"
        elif self.n_regimes >= 3:
            names[sorted_regimes[0]] = "bear"
            names[sorted_regimes[-1]] = "bull"
            for i in range(1, len(sorted_regimes) - 1):
                names[sorted_regimes[i]] = "sideways"

        return RegimeResult(
            regimes=regimes,
            regime_names=names,
            transition_matrix=self.model.transmat_,
            regime_means=regime_means,
            regime_vols=regime_vols,
            current_regime=int(hidden_states[-1]),
        )

    def _build_observations(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Build observation matrix for HMM from OHLCV data."""
        close = ohlcv["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std()

        return pd.DataFrame({"return": returns, "volatility": vol})

    def predict_current(self, ohlcv: pd.DataFrame) -> int:
        """Predict the current regime for the latest bar.

        Requires the model to be fitted first.

        Args:
            ohlcv: Recent OHLCV data

        Returns:
            Regime label for the latest bar

        Raises:
            RuntimeError: If model has not been fitted
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        observations = self._build_observations(ohlcv).dropna()
        if observations.empty:
            return 0

        states = self.model.predict(observations.values)
        return int(states[-1])
```

**Tasks:**
- [ ] Implement `RegimeDetector` with configurable n_regimes
- [ ] Implement `fit_predict()` with Gaussian HMM
- [ ] Auto-label regimes by mean return (bull/bear/sideways)
- [ ] Return transition matrix and per-regime statistics
- [ ] Write tests: synthetic trending/mean-reverting data → verify regime detection
- [ ] Test 2-regime detection (bull/bear)
- [ ] Test `predict_current()` after fitting
- [ ] Test unfitted model raises RuntimeError
- [ ] Add `hmmlearn` to `requirements.txt`

**Deliverable:** HMM regime detector with automatic regime labeling

---

### Step 7.2: MLflow Experiment Tracking
**Time:** Day 3-5

**File: `src/tracking/mlflow_tracker.py`**
```python
"""MLflow experiment tracking wrapper.

Provides a clean interface for logging backtest and ML experiment
results to MLflow. Supports parameter, metric, and artifact logging.
"""

from pathlib import Path
from typing import Any

import mlflow


class ExperimentTracker:
    """MLflow experiment tracking wrapper.

    Wraps MLflow to provide a simplified interface for:
    1. Logging backtest parameters (strategy, sizing, risk limits)
    2. Logging performance metrics (Sharpe, drawdown, return)
    3. Logging ML model metrics (train/val scores, feature importance)
    4. Saving artifacts (trained models, equity curves, feature plots)

    Attributes:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking server URI
        run_id: Current active run ID (None if no run active)
    """

    def __init__(
        self,
        experiment_name: str = "trading_system",
        tracking_uri: str = "mlruns",
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_id: str | None = None

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None) -> str:
        """Start a new MLflow run.

        Args:
            run_name: Optional human-readable name for the run

        Returns:
            The run ID
        """
        run = mlflow.start_run(run_name=run_name)
        self.run_id = run.info.run_id
        return self.run_id

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        self.run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters (flattens nested dicts with dot notation)."""
        flat = self._flatten_dict(params)
        for key, value in flat.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log numeric metrics."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

    def log_artifact(self, path: str | Path) -> None:
        """Log a file as an artifact."""
        mlflow.log_artifact(str(path))

    def log_backtest_result(
        self,
        config: dict[str, Any],
        metrics: dict[str, Any],
        trade_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Convenience method to log a complete backtest result.

        Args:
            config: Backtest configuration dict
            metrics: Performance metrics from PerformanceTracker
            trade_metrics: Optional trade-level metrics
        """
        self.log_params(config)
        self.log_metrics(metrics)
        if trade_metrics:
            self.log_metrics({f"trade_{k}": v for k, v in trade_metrics.items()
                            if isinstance(v, (int, float))})

    def _flatten_dict(
        self, d: dict[str, Any], parent_key: str = ""
    ) -> dict[str, Any]:
        """Flatten nested dict with dot notation keys."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
```

**Tasks:**
- [ ] Implement `ExperimentTracker` wrapping MLflow
- [ ] Parameter, metric, and artifact logging
- [ ] `log_backtest_result` convenience method
- [ ] Nested dict flattening for config logging
- [ ] Write tests: start/end run, log params/metrics (use local tracking URI)
- [ ] Test dict flattening with nested configs
- [ ] Add `mlflow` to `requirements.txt`

**Deliverable:** MLflow experiment tracker for backtest and ML experiment logging

---

## Week 8: Integration Testing, Validation & Documentation

### Step 8.1: Comprehensive Test Suite
**Time:** Day 1-3

**New test files to create:**

| Test File | Coverage |
|-----------|----------|
| `tests/test_features.py` | All 10 feature generators with known inputs/outputs |
| `tests/test_ml_model.py` | XGBoost + LightGBM train/predict/save/load |
| `tests/test_ml_strategy.py` | MLStrategy signal generation with mock model |
| `tests/test_cpcv.py` | CPCV splits, purging, embargo, no overlap |
| `tests/test_deflated_sharpe.py` | DSR, PSR, expected max Sharpe formulas |
| `tests/test_regime.py` | HMM fit, regime labeling, prediction |
| `tests/test_mlflow_tracker.py` | Tracking start/end/log with local URI |
| `tests/test_phase3_integration.py` | Full pipeline: features → model → strategy → backtest |

**Key test scenarios:**
- [ ] Feature pipeline produces correct column count
- [ ] RSI bounded in [0, 100]
- [ ] Hurst < 0.5 for mean-reverting synthetic data
- [ ] CPCV produces C(6,2)=15 paths with no train/test overlap
- [ ] Purge removes correct number of boundary samples
- [ ] DSR: 1 trial → observed Sharpe unchanged
- [ ] DSR: 1000 trials → significant deflation
- [ ] PSR with known parameters matches reference values
- [ ] HMM detects 2 regimes in synthetic bull/bear data
- [ ] ML strategy emits LONG when model predicts positive signal
- [ ] ML strategy emits EXIT when signal drops below threshold
- [ ] Full pipeline: features → train → predict → backtest produces results
- [ ] Backwards compatibility: Phase 2 config still works

**Deliverable:** 200+ new tests, total suite 800+

---

### Step 8.2: Extended Configuration
**Time:** Day 3-4

**File: `configs/ml_backtest_config.yaml`**
```yaml
data:
  symbols:
    - AAPL
    - MSFT
    - GOOGL
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  data_source: "csv"
  data_path: "data/sample"

execution:
  initial_capital: 100000.0
  commission_pct: 0.001
  slippage_pct: 0.0005

strategy:
  name: "ml_strategy"
  parameters:
    signal_threshold: 0.1
    exit_threshold: 0.0
    lookback_bars: 200

features:
  technical:
    - type: "sma"
      windows: [5, 10, 20, 50]
    - type: "rsi"
      period: 14
    - type: "macd"
    - type: "bollinger"
      window: 20
    - type: "atr"
      period: 14
  statistical:
    - type: "returns"
      horizons: [1, 5, 10, 20]
    - type: "zscore"
      window: 20
    - type: "higher_moments"
      window: 20
    - type: "volatility"
      windows: [5, 20, 60]
  target:
    horizon: 5
    type: "direction"

ml:
  model: "xgboost"
  mode: "classification"
  parameters:
    n_estimators: 200
    max_depth: 4
    learning_rate: 0.05
    subsample: 0.8

validation:
  method: "cpcv"
  n_splits: 6
  n_test_splits: 2
  purge_window: 5
  embargo_pct: 0.01

regime:
  enabled: true
  n_regimes: 3
  vol_window: 20

sizing:
  method: "volatility"
  parameters:
    risk_fraction: 0.02

risk:
  enabled: true
  max_position_pct: 0.10
  max_drawdown_pct: 0.15

optimization:
  method: "none"

tracking:
  enabled: true
  experiment_name: "ml_backtest"
```

**Tasks:**
- [ ] Create ML backtest config with all new sections
- [ ] Add `features`, `ml`, `validation`, `regime`, `tracking` config models to `src/config.py`
- [ ] Update `scripts/run_backtest.py` with ML factory functions
- [ ] Add feature config example
- [ ] Update README with Phase 3 features

**Deliverable:** Config-driven ML backtest with all Phase 3 components

---

### Step 8.3: Documentation & Final Validation
**Time:** Day 4-5

**Validation runs:**
- [ ] Single asset ML backtest (AAPL) — XGBoost classification
- [ ] Multi-asset ML backtest (5 symbols) — LightGBM regression
- [ ] CPCV validation across full date range
- [ ] Deflated Sharpe Ratio with N=50 trials
- [ ] Regime detection on 4-year data — verify 3 regimes
- [ ] MLflow tracking — verify all metrics logged
- [ ] Backwards compatibility: Phase 2 SMA config still works

**Documentation:**
- [ ] Update README with Phase 3 features
- [ ] Document ML configuration options
- [ ] Document feature engineering pipeline usage
- [ ] Add NOTES-Phase3-Week8.md

**Deliverable:** Validated ML trading system with documentation

---

## Success Criteria Checklist

### Functional Requirements
- [ ] Feature pipeline: 30+ features from OHLCV data
- [ ] XGBoost model: train, predict, save, load
- [ ] LightGBM model: train, predict, save, load
- [ ] ML strategy: generates signals from model predictions
- [ ] CPCV: generates correct number of purged paths
- [ ] DSR: deflates Sharpe ratio proportional to trials
- [ ] HMM: detects bull/bear/sideways regimes
- [ ] MLflow: logs parameters, metrics, and artifacts
- [ ] Config-driven: all new components configurable via YAML

### Code Quality
- [ ] All tests pass (800+ total)
- [ ] Backwards compatible with Phase 1 and Phase 2 configurations
- [ ] Type hints on all new public methods
- [ ] Docstrings on all new classes and methods
- [ ] Code passes linting (ruff, black, mypy)

### Performance
- [ ] Feature pipeline runs < 1 second for 4 years of daily data
- [ ] Model training < 10 seconds for 4 years of data
- [ ] Full ML backtest (single asset, 4 years) < 60 seconds

---

## Risk Mitigation

| Risk | Mitigation | Status |
|------|------------|--------|
| Look-ahead bias in features | Features use only past data; target uses shift(-N) | ☐ |
| Overfitting ML model | CPCV validation, DSR for multiple testing correction | ☐ |
| Feature leakage in CV | Purging + embargo in CPCV | ☐ |
| HMM instability | Multiple random restarts, covariance regularization | ☐ |
| XGBoost/LightGBM version conflicts | Pin versions in requirements.txt | ☐ |
| MLflow overhead | Optional tracking (disabled by default) | ☐ |
| Breaking Phase 2 | All existing tests must pass, default params match Phase 2 | ☐ |

---

## New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `xgboost` | >=2.0 | Gradient boosted trees for signal generation |
| `lightgbm` | >=4.0 | Light gradient boosting for signal generation |
| `hmmlearn` | >=0.3 | Hidden Markov Model for regime detection |
| `mlflow` | >=2.10 | Experiment tracking |
| `scipy` | >=1.11 | Statistical functions (DSR, PSR) |

---

## Next Steps (Phase 4 Preview)

After completing Phase 3:
1. Alternative data integration (sentiment, options flow)
2. Reinforcement learning for execution optimization
3. Live paper trading bridge (Alpaca/IBKR)
4. Multi-timeframe analysis (intraday + daily)
5. Portfolio-level regime-conditional allocation
6. Distributed backtesting (Dask/Ray)

---

*Phase 3 Development Plan v1.0*

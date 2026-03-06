# Phase 3, Weeks 1-2 — Feature Engineering Pipeline

## What this is

A composable feature engineering pipeline that transforms raw OHLCV price data into
clean, ML-ready feature matrices. This is the foundation for all ML work in Phase 3 —
models consume these features, not raw prices.

**Before:**
```
Strategy sees raw OHLCV data (open, high, low, close, volume)
No feature engineering
No way to feed structured inputs to an ML model
```

**After:**
```
10 feature generators (5 technical, 5 statistical)
FeaturePipeline composes any combination into a clean matrix
Automatic NaN trimming from lookback windows
Forward-return target generation (log return or binary direction)
compute_features_only() for live prediction (no target)
```

## What was built

### 1. Base Classes (`src/features/base_feature.py`)

Two abstractions that all feature generators implement:

- **`FeatureResult`** — frozen dataclass with `features` (DataFrame), `feature_names` (list), `method` (str)
- **`FeatureGenerator`** — ABC requiring `compute(ohlcv) -> FeatureResult` and `lookback` property

Generators are stateless — no instance state mutated during `compute()`. NaN rows from
lookback periods are retained in the output; the pipeline handles trimming.

### 2. Technical Indicators (`src/features/technical.py`)

| Class | Output Columns | Lookback |
|-------|---------------|----------|
| `SMAFeature(windows=[5,10,20,50])` | `sma_{w}`, `sma_{w}_ratio` (close/SMA) | `max(windows)` |
| `RSIFeature(period=14)` | `rsi_{period}` (0-100) | `period + 1` |
| `MACDFeature(fast=12,slow=26,signal=9)` | `macd`, `macd_signal`, `macd_histogram` | `slow + signal` |
| `BollingerBandFeature(window=20,num_std=2)` | `bb_width`, `bb_pct_b` | `window` |
| `ATRFeature(period=14)` | `atr_{period}`, `atr_{period}_pct` (ATR/close) | `period + 1` |

**RSI edge case:** when `avg_loss == 0` (all gains), RSI returns 100.0 (not NaN). This
follows the standard financial convention — pure uptrend = maximum RSI.

**Bollinger edge case:** when `std == 0` (constant price), `bb_width = 0` and
`bb_pct_b = NaN` (division by zero band range replaced with NaN).

### 3. Statistical Features (`src/features/statistical.py`)

| Class | Output Columns | Lookback |
|-------|---------------|----------|
| `ReturnFeature(horizons=[1,5,10,20])` | `return_{h}d` (log return) | `max(horizons) + 1` |
| `ZScoreFeature(window=20)` | `zscore_{window}` | `window` |
| `HigherMomentFeature(window=20)` | `skew_{window}`, `kurt_{window}` | `window + 1` |
| `HurstExponentFeature(window=100)` | `hurst_{window}` (0-1) | `window` |
| `VolatilityFeature(windows=[5,20,60])` | `vol_{w}`, `vol_ratio_{short}_{long}` | `max(windows) + 1` |

**HurstExponentFeature** uses R/S (rescaled range) analysis to estimate the Hurst exponent:
- H < 0.5: mean-reverting
- H = 0.5: random walk
- H > 0.5: trending

Requires `window >= 20`, values clipped to [0, 1]. Uses Python for-loop per bar
(O(n × k²)) — acceptable for backtesting, would need vectorization for production.

**VolatilityFeature** annualizes by `sqrt(252)`. When `len(windows) >= 2`, produces a
`vol_ratio` column (short vol / long vol) as a regime indicator.

### 4. Feature Pipeline (`src/features/pipeline.py`)

**`FeaturePipeline`** composes generators and handles all the plumbing:

```python
pipeline = FeaturePipeline([
    SMAFeature([10, 20]),
    RSIFeature(14),
    ReturnFeature([1, 5]),
])
```

**`run(ohlcv, target_horizon=5, target_type="return")`** — training mode:
1. Runs all generators on the full OHLCV data
2. Concatenates all feature columns
3. Creates forward target: `ln(close[t+h] / close[t])` for "return", binary for "direction"
4. Drops all NaN rows (lookback at start + forward shift at end)
5. Returns `PipelineResult` with clean features, target, feature_names, rows_dropped

**`compute_features_only(ohlcv)`** — prediction mode:
- Features only, no target (for live/inference use)
- Returns clean DataFrame after `dropna()`

**Key properties:**
- `total_lookback` = `max(g.lookback for g in generators)`
- `rows_dropped` = `(total_lookback - 1) + target_horizon` (for rolling NaN + forward shift)
- Validates `target_horizon >= 1`, rejects unknown `target_type`
- Stores a defensive copy of the generators list

## How the pieces connect

```
OHLCV DataFrame (open, high, low, close, volume + DatetimeIndex)
    │
    ▼
FeaturePipeline
    ├── SMAFeature.compute(ohlcv)     → sma_10, sma_10_ratio, sma_20, ...
    ├── RSIFeature.compute(ohlcv)     → rsi_14
    ├── ReturnFeature.compute(ohlcv)  → return_1d, return_5d
    └── ... any combination ...
    │
    ▼
Concatenate all columns + create forward target
    │
    ▼
dropna() (removes lookback NaN rows + forward-shift NaN rows)
    │
    ▼
PipelineResult
    ├── features: pd.DataFrame  (clean, no NaN)
    ├── feature_names: list[str]
    ├── target: pd.Series       (forward log return or binary direction)
    └── rows_dropped: int
```

This feeds directly into the ML models built in Weeks 3-4.

## Tests

**46 tests** in `tests/test_features_technical.py`:
- `FeatureResult` creation and immutability (frozen dataclass)
- `FeatureGenerator` ABC cannot be instantiated
- `SMAFeature`: default windows, lookback, formula verification against known series, ratio = close/SMA, insufficient data → all NaN
- `RSIFeature`: lookback, range [0,100], all-gains → 100, all-losses → 0, NaN boundary at correct row
- `MACDFeature`: lookback, histogram = macd - signal, MACD line = fast EMA - slow EMA
- `BollingerBandFeature`: width > 0 for volatile data, width = 0 and pct_b = NaN for constant prices
- `ATRFeature`: non-negative, zero for flat OHLCV, pct = ATR/close, gap detection

**47 tests** in `tests/test_features_statistical.py`:
- `ReturnFeature`: log return formula verified for 1-day and multi-day, NaN for insufficient bars
- `ZScoreFeature`: formula verification, positive z-score for trending up, NaN for constant price
- `HigherMomentFeature`: skew/kurt columns, constant returns → skew=0, kurt=-3
- `HurstExponentFeature`: window < 20 raises, values in [0,1], trending (AR(1) ρ=0.7) → H > 0.5, mean-reverting → H < 0.5
- `VolatilityFeature`: annualization, zero vol for constant prices, vol_ratio = short/long, no ratio for single window

**30 tests** in `tests/test_features_pipeline.py`:
- `PipelineResult`: creation, defaults
- Construction: empty generators raises, total_lookback = max, defensive copy
- `run()`: returns PipelineResult, no NaN in output, correct column count, rows_dropped formula, return target formula verified row-by-row, direction target binary and consistent with return
- `compute_features_only()`: returns DataFrame, no NaN, no target column, consistent with run() features
- Integration: 3-generator pipeline on 500 bars, mixed technical+statistical, both direction classes present, 2000-bar performance test

**Total: 727 tests passing (604 existing + 123 new).**

## Files changed

| File | Action | What |
|------|--------|------|
| `src/features/__init__.py` | Created | Public exports for all 10 generators + pipeline classes |
| `src/features/base_feature.py` | Created | `FeatureResult` frozen dataclass, `FeatureGenerator` ABC |
| `src/features/technical.py` | Created | 5 technical indicator generators |
| `src/features/statistical.py` | Created | 5 statistical feature generators |
| `src/features/pipeline.py` | Created | `FeaturePipeline`, `PipelineResult` |
| `tests/test_features_technical.py` | Created | 46 tests |
| `tests/test_features_statistical.py` | Created | 47 tests |
| `tests/test_features_pipeline.py` | Created | 30 tests |
| `requirements.txt` | Modified | Added `scikit-learn>=1.3.0` |

## Usage examples

```python
# Build a feature matrix for model training
from src.features import FeaturePipeline, SMAFeature, RSIFeature, ReturnFeature

pipeline = FeaturePipeline([
    SMAFeature([10, 20, 50]),
    RSIFeature(14),
    ReturnFeature([1, 5, 10]),
])

result = pipeline.run(ohlcv_df, target_horizon=5, target_type="direction")
# result.features: 12 columns, no NaN
# result.target: binary 0/1 (price up or down in 5 days)
# result.rows_dropped: ~54 (lookback NaN + forward shift NaN)

X_train = result.features
y_train = result.target
```

```python
# Compute features for live prediction (no target)
latest_bars = data_handler.get_latest_bars("AAPL", 200)
features = pipeline.compute_features_only(latest_bars)
latest_row = features.iloc[[-1]]  # most recent feature vector
```

```python
# Custom generator combination
from src.features import (
    ATRFeature, BollingerBandFeature, HurstExponentFeature,
    VolatilityFeature, ZScoreFeature,
)

pipeline = FeaturePipeline([
    BollingerBandFeature(window=20),
    ATRFeature(14),
    ZScoreFeature(20),
    VolatilityFeature([5, 20]),
    HurstExponentFeature(window=100),
])
# total_lookback = 100 (from Hurst)
# 9 feature columns
```

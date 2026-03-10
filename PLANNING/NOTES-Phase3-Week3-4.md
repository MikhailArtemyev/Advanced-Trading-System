# Phase 3, Weeks 3-4 — ML Model Integration

## What this is

XGBoost and LightGBM signal generation models behind a common abstract interface,
plus an `MLStrategy` that plugs ML predictions into the event-driven backtester.
This closes the loop between the feature pipeline (Weeks 1-2) and the backtesting
engine (Phase 2).

**Before (Weeks 1-2):**
```
Feature pipeline produces clean feature matrices
No ML models
No way to use features in the backtester
Strategies are rule-based only (SMA crossover)
```

**After (Weeks 3-4):**
```
MLModel ABC with train/predict/save/load/feature_importance
XGBoostSignalModel — classification (direction) or regression (return)
LightGBMSignalModel — same interface, LightGBM backend
MLStrategy(Strategy) — feeds features to model, maps predictions to signals
Full event loop: OHLCV → features → model → signal → order → fill
```

## What was built

### 1. Base Classes (`src/ml/base_model.py`)

Three data structures and one ABC:

**`ModelPrediction`** — what the model returns per bar:
```python
signal: float       # -1.0 (strong short) to 1.0 (strong long)
confidence: float   # 0.0 to 1.0
features_used: int  # number of input features
```

**`TrainResult`** — what training returns:
```python
train_score: float                           # accuracy (clf) or R² (reg)
val_score: float | None                      # if validation set provided
feature_importance: dict[str, float]         # feature name → importance
n_train_samples: int
n_features: int
```

**`MLModel`** — ABC with 5 abstract methods + 1 property:

| Method | Purpose |
|--------|---------|
| `train(features, target, val_features?, val_target?)` | Train on feature matrix + target |
| `predict(features)` | Returns `list[ModelPrediction]`, one per row |
| `save(path)` / `load(path)` | Serialize to disk and restore |
| `get_feature_importance()` | Feature name → importance dict |
| `is_trained` (property) | Whether model is ready for prediction |

### 2. XGBoost Model (`src/ml/xgboost_model.py`)

```python
model = XGBoostSignalModel(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    mode="classification",    # or "regression"
    regression_signal_scale=100.0,
)
```

**Classification mode** (`binary:logistic`, eval metric `auc`):
- Predicts P(up) via `predict_proba`
- Signal: `(prob_up - 0.5) * 2` → maps [0, 1] to [-1, 1]
- Confidence: `abs(prob_up - 0.5) * 2` → maps [0.5, 1.0] to [0, 1]

**Regression mode** (`reg:squarederror`, eval metric `rmse`):
- Predicts raw forward return
- Signal: `clip(pred * regression_signal_scale, -1, 1)`
- Confidence: `min(abs(pred) * scale / 2, 1.0)`
- Scale is configurable — default 100.0 works for log returns (~0.001-0.01 magnitude)

**Save/load** uses `joblib.dump/load` for the sklearn model + a `{path}.meta.json`
sidecar file storing `feature_names`, `mode`, and `regression_signal_scale`. This
approach:
- Preserves feature names across save/load
- Validates mode on load (raises ValueError on mismatch)
- Creates parent directories automatically
- Works identically for both XGBoost and LightGBM

**Guards:**
- Predict before training → `RuntimeError`
- Invalid mode → `ValueError`
- Single-class target in classification → `ValueError`
- Feature column mismatch at predict time → `ValueError`
- Empty features → returns `[]`
- Save untrained model → `RuntimeError`

### 3. LightGBM Model (`src/ml/lightgbm_model.py`)

Same interface as XGBoost. Differences:
- Uses `lgb.LGBMClassifier` / `lgb.LGBMRegressor`
- `verbose=-1` in params to silence LightGBM output
- Same joblib + meta.json save/load (avoids fragile `booster_.save_model`)

Code duplication between XGBoost and LightGBM is accepted — both are ~120 lines,
differ in framework details, and a shared base would couple them unnecessarily. If a
third model is added (e.g., CatBoost), refactor into a shared base at that point.

### 4. ML Strategy (`src/ml/ml_strategy.py`)

```python
strategy = MLStrategy(
    symbols=["AAPL", "MSFT"],
    pipeline=pipeline,
    model=trained_model,
    parameters={
        "signal_threshold": 0.1,    # min |signal| for entry
        "exit_threshold": 0.0,      # |signal| below which to exit
        "lookback_bars": 200,       # bars to fetch for feature computation
        "shorts_enabled": False,    # allow SHORT signals
    },
)
```

Extends `Strategy` base class from Phase 2. Implements `calculate_signals()`:

For each symbol:
1. Fetch `lookback_bars` via `data_handler.get_latest_bars(symbol, n)`
2. If insufficient bars (< `pipeline.total_lookback + 1`): skip
3. Compute features: `pipeline.compute_features_only(bars)`
4. If empty features: skip
5. Predict on latest row: `model.predict(features.iloc[[-1]])[0]`
6. Signal decision:

| Condition | Position | Action |
|-----------|----------|--------|
| `signal > threshold` | flat or short | LONG (strength = confidence) |
| `signal < -threshold` | long | EXIT |
| `signal < -threshold` | flat, shorts enabled | SHORT (strength = confidence) |
| `signal < -threshold` | flat, shorts disabled | nothing |
| `|signal| < exit_threshold` | any position | EXIT |
| otherwise | any | nothing |

No duplicate entry signals — checks `current_positions` before emitting.
`lookback_bars` defaults to `pipeline.total_lookback + 50` (extra padding for safety).

## How the pieces connect

```
Historical OHLCV Data
    │
    ├─── TRAINING TIME ──────────────────────────────────────
    │
    │   FeaturePipeline.run(ohlcv, target_horizon=5, target_type="direction")
    │       │
    │       ▼
    │   PipelineResult (features + target)
    │       │
    │       ▼
    │   XGBoostSignalModel.train(features, target)
    │       │
    │       ▼
    │   TrainResult (score, feature importance)
    │   model.save("models/xgb_v1")
    │
    ├─── BACKTEST TIME ──────────────────────────────────────
    │
    │   MLStrategy(symbols, pipeline, trained_model)
    │       │
    │       ▼  (for each bar, for each symbol)
    │   data_handler.get_latest_bars(symbol, lookback_bars)
    │       │
    │       ▼
    │   pipeline.compute_features_only(bars)
    │       │
    │       ▼
    │   model.predict(latest_features)
    │       │
    │       ▼
    │   ModelPrediction(signal, confidence)
    │       │
    │       ▼
    │   SignalEvent(LONG/SHORT/EXIT, strength=confidence)
    │       │
    │       ▼
    │   Portfolio → OrderEvent → RiskManager → ExecutionHandler → FillEvent
```

The MLStrategy is a drop-in replacement for SMACrossoverStrategy — it plugs into
the same event loop, same portfolio, same risk manager. You just swap the strategy
in the backtest engine.

## Tests

**49 tests** in `tests/test_ml_model.py`:

*TestModelPrediction (2 tests):*
- Creation with positive/negative signals

*TestTrainResult (2 tests):*
- All fields, default values

*TestMLModelABC (1 test):*
- Cannot instantiate abstract class

*TestXGBoostClassification (19 tests):*
- Train returns TrainResult with correct fields
- Train score between 0 and 1
- Validation score returned when val set provided
- Predict returns list of ModelPrediction
- Signal always in [-1, 1], confidence in [0, 1]
- features_used matches input column count
- Untrained predict raises RuntimeError
- Empty features returns empty list
- Column mismatch raises ValueError
- Feature importance keys match training columns
- Untrained importance returns empty dict
- Save/load roundtrip — predictions match after reload
- Save untrained raises RuntimeError
- Load restores feature_names
- Load with mode mismatch raises ValueError
- Invalid mode in constructor raises ValueError
- is_trained property works
- Single-class target raises ValueError

*TestXGBoostRegression (4 tests):*
- Train, predict range, custom signal scale, save/load roundtrip

*TestLightGBMClassification (19 tests):*
- Mirror of XGBoost classification tests

*TestLightGBMRegression (3 tests):*
- Train, predict range, save/load roundtrip

**17 tests** in `tests/test_ml_strategy.py`:

Uses a hand-written `StubMLModel` (returns pre-configured predictions, no real ML
libraries needed) and `StubDataHandler` (synthetic OHLCV data).

*TestMLStrategyConstruction (4 tests):*
- Pipeline and model stored, default params, custom params, lookback from pipeline

*TestMLStrategySignals (12 tests):*
- LONG above threshold, no signal below threshold
- EXIT when long and signal negative
- SHORT when enabled, no SHORT when disabled
- EXIT on weak signal (below exit_threshold) when in position
- Signal strength equals model confidence
- No duplicate LONG when already long
- No signal with insufficient data
- Multiple symbols generate independent signals
- Correct timestamp and symbol on signals

*TestMLStrategyIntegration (1 test):*
- End-to-end: real FeaturePipeline + real XGBoostSignalModel trained on synthetic data,
  run through strategy, verify SignalEvent types and strength range

**Total: 793 tests passing (727 existing + 66 new).**

## Files changed

| File | Action | What |
|------|--------|------|
| `src/ml/__init__.py` | Created | Public exports for all ML classes |
| `src/ml/base_model.py` | Created | `MLModel` ABC, `ModelPrediction`, `TrainResult` |
| `src/ml/xgboost_model.py` | Created | `XGBoostSignalModel` (classification + regression) |
| `src/ml/lightgbm_model.py` | Created | `LightGBMSignalModel` (classification + regression) |
| `src/ml/ml_strategy.py` | Created | `MLStrategy(Strategy)` |
| `tests/test_ml_model.py` | Created | 49 tests |
| `tests/test_ml_strategy.py` | Created | 17 tests |
| `requirements.txt` | Modified | Added `xgboost>=2.0.0`, `lightgbm>=4.0.0` |

## Usage examples

```python
# Train an XGBoost classifier on features
from src.features import FeaturePipeline, SMAFeature, RSIFeature, ATRFeature
from src.ml import XGBoostSignalModel

pipeline = FeaturePipeline([SMAFeature([10, 20]), RSIFeature(14), ATRFeature(14)])
result = pipeline.run(ohlcv_df, target_horizon=5, target_type="direction")

model = XGBoostSignalModel(n_estimators=200, mode="classification")
train_result = model.train(result.features, result.target)
print(f"Accuracy: {train_result.train_score:.1%}")
print(f"Top features: {sorted(train_result.feature_importance.items(), key=lambda x: -x[1])[:3]}")
```

```python
# Save and reload
model.save("models/xgb_direction_v1")

model2 = XGBoostSignalModel(mode="classification")
model2.load("models/xgb_direction_v1")
preds = model2.predict(new_features)  # same predictions as original
```

```python
# Use in backtester
from src.ml import MLStrategy

strategy = MLStrategy(
    symbols=["AAPL", "MSFT"],
    pipeline=pipeline,
    model=model,
    parameters={"signal_threshold": 0.15, "shorts_enabled": True},
)

# Plug into BacktestEngine exactly like SMACrossoverStrategy
engine = BacktestEngine(
    data_handler=data_handler,
    strategy=strategy,
    portfolio=portfolio,
    execution_handler=execution_handler,
)
engine.run()
```

```python
# LightGBM — identical interface
from src.ml import LightGBMSignalModel

lgb_model = LightGBMSignalModel(n_estimators=300, mode="regression")
lgb_model.train(features, target)
preds = lgb_model.predict(latest_features)
# preds[0].signal in [-1, 1], preds[0].confidence in [0, 1]
```

# Phase 3, Week 8 — Integration Testing, Validation & Documentation

## What this is

The final integration week: wiring all Phase 3 components (features, ML models,
CPCV, DSR, regime detection, experiment tracking) into the main backtest runner,
extending the config system with backwards-compatible defaults, and verifying
everything works end-to-end.

**Before (Week 7):**
```
ML models exist but can't be run from the command line
Config system doesn't know about features, ML, or regime settings
No integration tests spanning features → model → backtest → metrics
Phase 2 configs might break with new code
```

**After (Week 8):**
```
`make run-ml` runs full ML backtest from config
Config-driven feature pipeline, model selection, and training
23 integration tests covering all Phase 3 component interactions
All Phase 2 configs load and run without changes (backwards compatible)
997 total tests passing, all checks green
```

## What was built

### 1. Config Extension (`src/config.py`)

Five new Pydantic config models added to `BacktestConfig`:

| Config | Key Fields | Defaults |
|--------|-----------|----------|
| `FeatureConfig` | `technical`, `statistical`, `target` | empty lists, `{"horizon": 5, "type": "direction"}` |
| `MLConfig` | `model`, `mode`, `parameters` | `"xgboost"`, `"classification"`, `{}` |
| `ValidationConfig` | `n_splits`, `n_test_splits`, `purge_window`, `embargo_pct` | 6, 2, 5, 0.01 |
| `RegimeConfig` | `enabled`, `n_regimes`, `vol_window` | `False`, 3, 20 |
| `TrackingConfig` | `enabled`, `experiment_name` | `False`, `"trading_system"` |

All have `Field(default_factory=...)` on `BacktestConfig`, so existing Phase 2
YAML files load without changes — missing sections use defaults.

**Validation rules:**
- `MLConfig.model` must be `"xgboost"` or `"lightgbm"`
- `MLConfig.mode` must be `"classification"` or `"regression"`
- `RegimeConfig.n_regimes` must be >= 2
- `ValidationConfig.n_splits` must be >= 3

### 2. ML Wiring in `scripts/run_backtest.py`

**Feature pipeline builder** — `build_feature_pipeline(config)`:
- Maps config `type` strings to feature constructors via `_FEATURE_BUILDERS` dict
- Falls back to a default set of 8 generators if none specified in config
- Supports: sma, rsi, macd, bollinger, atr, returns, zscore, higher_moments, hurst, volatility

**ML model builder** — `build_ml_model(config)`:
- Creates `XGBoostSignalModel` or `LightGBMSignalModel` from `config.ml`
- Passes through all parameters from config

**Strategy builder update** — `build_strategy(config, optimizer, data_handler)`:
- New `ml_strategy` path: builds pipeline, trains model on historical data, returns `MLStrategy`
- Training data read from `data_handler.data[symbol]` (bypasses `current_index` limit)
- Logs training score, feature count, and sample count
- Non-ML strategies unchanged

**Training data access pattern:**

```python
# data_handler.get_latest_bars() respects current_index (returns nothing at start)
# data_handler.data[symbol] has the full history (needed for pre-backtest training)
train_data = data_handler.data[first_symbol].copy()
pipeline_result = pipeline.run(train_data, target_horizon=horizon, target_type=target_type)
model.train(pipeline_result.features, pipeline_result.target)
```

This is the key design insight: the backtest engine's `update_bars()` advances
`current_index` one bar at a time, so `get_latest_bars()` returns nothing before
the first bar. Training needs all data upfront, so it reads directly from the
handler's internal `data` dict.

### 3. ML Backtest Config (`configs/ml_backtest_config.yaml`)

```yaml
strategy:
  name: "ml_strategy"
  parameters:
    signal_threshold: 0.1
    exit_threshold: 0.0
    lookback_bars: 200
    shorts_enabled: false

features:
  technical:
    - type: "sma"       (windows: [5, 10, 20, 50])
    - type: "rsi"       (period: 14)
    - type: "macd"
    - type: "bollinger" (window: 20)
    - type: "atr"       (period: 14)
  statistical:
    - type: "returns"        (horizons: [1, 5, 10, 20])
    - type: "zscore"         (window: 20)
    - type: "higher_moments" (window: 20)
    - type: "volatility"     (windows: [5, 20, 60])

ml:
  model: "xgboost"
  mode: "classification"
  parameters:
    n_estimators: 200
    max_depth: 4
    learning_rate: 0.05
    subsample: 0.8
```

Uses volatility-based sizing and risk management. Regime and tracking disabled
by default (can be turned on by setting `regime.enabled: true`).

### 4. Makefile Target

```makefile
run-ml: download-data
	$(PYTHON) ./scripts/run_backtest.py --config configs/ml_backtest_config.yaml
```

## ML Backtest Results

Running `make run-ml` on AAPL/MSFT/GOOGL (2020-2024):

```
Total Return:     58.75%
Sharpe Ratio:     1.88
Max Drawdown:     -6.50%
Win Rate:         80.2%
Profit Factor:    3.37
Calmar Ratio:     9.04
```

**Overfitting caveat:** These results are misleading. The model trains on the
same data it backtests — this is in-sample performance, not out-of-sample. The
high Sharpe (1.88) and win rate (80%) reflect memorization, not prediction.

To measure true performance:
1. Use CPCV (Week 5) for cross-validated accuracy
2. Apply DSR (Week 6) to adjust for multiple testing
3. Walk-forward analysis for realistic OOS estimates

The in-sample backtest exists to verify the pipeline works end-to-end, not to
measure strategy quality.

## Tests

**23 tests** in `tests/test_phase3_integration.py`:

### TestFeatureToModelPipeline (3 tests)

| Test | What |
|------|------|
| `test_xgboost_classification_pipeline` | Features → XGBoost train → predict, verify signal/confidence ranges |
| `test_lightgbm_regression_pipeline` | Features → LightGBM regression → predict |
| `test_xgboost_with_validation_split` | Train/val split, verify val_score returned |

### TestCPCVWithRealModel (2 tests)

| Test | What |
|------|------|
| `test_cpcv_xgboost` | CPCV with real XGBoost, verify C(5,2)=10 paths, scores in [0,1] |
| `test_cpcv_predictions_collected` | OOS predictions collected across paths |

### TestDSRWithBacktestMetrics (2 tests)

| Test | What |
|------|------|
| `test_dsr_on_ml_sharpe` | DSR on moderate Sharpe with negative skew |
| `test_single_trial_preserves_sharpe` | Single trial: no deflation applied |

### TestRegimeWithRealData (2 tests)

| Test | What |
|------|------|
| `test_regime_on_synthetic_ohlcv` | Regime detection on synthetic data, verify bull/bear labels |
| `test_regime_proba_on_recent` | Probability vector sums to 1 |

### TestMLStrategyBacktest (4 tests)

Uses `InMemoryHandler(DataHandler)` — a test-only data handler wrapping a DataFrame:

| Test | What |
|------|------|
| `test_xgboost_backtest_runs` | Full backtest with XGBoost, verify trade_history and sharpe_ratio |
| `test_lightgbm_backtest_runs` | Full backtest with LightGBM |
| `test_ml_backtest_produces_trades` | Verify trade history is populated |
| `test_ml_backtest_metrics_valid` | Verify metric types are correct |

### TestConfigBackwardsCompatibility (5 tests)

| Test | What |
|------|------|
| `test_phase2_config_loads` | Default config loads, new fields use defaults |
| `test_baseline_config_loads` | Baseline config works |
| `test_vol_config_loads` | Volatility sizing config works |
| `test_kelly_config_loads` | Kelly sizing config works |
| `test_ml_config_loads` | ML config has correct features/model/mode |

### TestConfigValidation (4 tests)

| Test | What |
|------|------|
| `test_invalid_ml_model_raises` | Invalid model name → ValueError |
| `test_invalid_ml_mode_raises` | Invalid mode → ValueError |
| `test_regime_n_regimes_min` | n_regimes < 2 → ValueError |
| `test_validation_n_splits_min` | n_splits < 2 → ValueError |

### TestEndToEnd (1 test)

Full Phase 3 pipeline in one test:
1. Generate 600 bars of synthetic data
2. Build feature pipeline (8 generators)
3. Run CPCV validation (5 splits, 10 paths)
4. Train final model on all data
5. Verify regime detection (bull/bear labels)
6. Apply DSR analysis (using CPCV path count as n_trials)

## Design decisions

1. **All new config fields have defaults**: No Phase 2 config changes needed.
   `FeatureConfig` defaults to empty lists (pipeline builder falls back to default
   generators). `MLConfig` defaults to xgboost/classification. `RegimeConfig` and
   `TrackingConfig` default to disabled.

2. **Training reads from `data_handler.data` not `get_latest_bars`**: The backtest
   API limits data access to the current bar index. Training needs all history
   before the backtest starts, so it accesses the handler's internal data store
   directly. This is intentional coupling — training is a pre-backtest step.

3. **`_FEATURE_BUILDERS` dict for config mapping**: Avoids a long if/elif chain.
   Adding a new feature type is one line: `"name": lambda cfg: FeatureClass(...)`.

4. **InMemoryHandler in tests**: Integration tests don't use CSV files or yfinance.
   `InMemoryHandler` wraps a DataFrame and implements the full `DataHandler` ABC.
   This keeps tests fast, deterministic, and self-contained.

5. **Overfitting is documented, not hidden**: The ML backtest trains on the same
   data it evaluates. We report the inflated metrics transparently and point to
   CPCV + DSR as the correct evaluation methods.

## Files

| File | Lines | Action |
|------|-------|--------|
| `src/config.py` | 173 | Modified — 5 new config models + BacktestConfig fields |
| `scripts/run_backtest.py` | 376 | Modified — ML pipeline wiring, feature builders |
| `configs/ml_backtest_config.yaml` | 84 | Created — full ML backtest config |
| `tests/test_phase3_integration.py` | 481 | Created — 23 integration tests |
| `Makefile` | 112 | Modified — added run-ml target |

## Running total

**997 tests passing** (974 after Week 7 + 23 new).

`make check` passes: black, ruff, mypy, all tests.

## Phase 3 Summary

| Week | Deliverable | Tests Added | Total |
|------|-------------|-------------|-------|
| 1-2 | Feature engineering pipeline (10 generators) | 123 | 727 |
| 3-4 | ML models (XGBoost, LightGBM) + MLStrategy | 66 | 793 |
| 5 | CPCV validation framework | 64 | 857 |
| 6 | Deflated Sharpe Ratio + strategy comparison | 39 | 896 |
| 7 | Regime detection + experiment tracking | 78 | 974 |
| 8 | Integration testing + config + wiring | 23 | 997 |

**Total Phase 3: 393 new tests across 8 weeks.**

Components built:
- `src/features/` — 10 feature generators + composable pipeline
- `src/ml/` — MLModel ABC, XGBoost, LightGBM, MLStrategy
- `src/validation/` — CPCV, DSR, PSR, E[max SR]
- `src/regime/` — HMM regime detection
- `src/tracking/` — MLflow experiment tracking
- Extended config system (5 new sections, fully backwards compatible)
- `make run-ml`, `make report`, `make run-compare` + individual targets

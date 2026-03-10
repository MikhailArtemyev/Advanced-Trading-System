# Phase 3, Week 7 — Regime Detection & Experiment Tracking

## What this is

Two independent components:

1. **HMM Regime Detection** — classifies market periods as bull, bear, or sideways
   using a Hidden Markov Model on return and volatility observations.

2. **MLflow Experiment Tracker** — logs backtest parameters, metrics, and artifacts
   for reproducibility and comparison across runs.

**Before (Week 6):**
```
No way to know what market regime the strategy is operating in
No persistent record of backtest runs and their results
Strategy treats all market conditions identically
```

**After (Week 7):**
```
RegimeDetector identifies bull/bear/sideways from price data
Transition matrix shows regime persistence and switching probabilities
predict_current() and predict_proba() for live regime assessment
ExperimentTracker logs every backtest to MLflow with full config + metrics
Context manager API for clean run lifecycle
```

## Part 1: HMM Regime Detection

### How it works

A Gaussian Hidden Markov Model treats market regimes as hidden states that
generate observable features. We observe two features per bar:

```
Feature 1: daily return  = close.pct_change()
Feature 2: volatility    = returns.rolling(vol_window).std()
```

The HMM learns:
- **Emission parameters**: mean and variance of (return, volatility) for each regime
- **Transition matrix**: P(regime_t | regime_{t-1}) — how likely each regime is to
  persist or switch
- **Initial probabilities**: starting regime distribution

After fitting, it assigns each bar to the most likely regime using the Viterbi algorithm.

### Auto-labeling

Regimes are numbered 0, 1, 2, ... by the HMM with no inherent meaning. We
auto-label them by sorting on mean return:

| n_regimes | Labeling |
|-----------|----------|
| 2 | lowest mean → "bear", highest → "bull" |
| 3 | lowest → "bear", highest → "bull", middle → "sideways" |
| 4+ | lowest → "bear", highest → "bull", all middle → "sideways" |

### What was built (`src/regime/hmm.py`)

**`RegimeResult`** — dataclass returned by `fit_predict()`:
```python
@dataclass
class RegimeResult:
    regimes: pd.Series           # date → regime label (0, 1, 2, ...)
    regime_names: dict[int, str] # regime number → "bull"/"bear"/"sideways"
    transition_matrix: np.ndarray # shape (n_regimes, n_regimes), rows sum to 1
    regime_means: dict[int, float]  # mean return per regime
    regime_vols: dict[int, float]   # volatility per regime
    current_regime: int             # most recent regime
```

**`RegimeDetector`** — the main class:
```python
detector = RegimeDetector(
    n_regimes=3,       # number of hidden states
    vol_window=20,     # rolling volatility window
    n_iter=100,        # max EM iterations
    random_state=42,   # for reproducibility
)
```

Three public methods:

| Method | Purpose |
|--------|---------|
| `fit_predict(ohlcv)` | Fit HMM on full history, return RegimeResult |
| `predict_current(ohlcv)` | Predict regime for latest bar (must fit first) |
| `predict_proba(ohlcv)` | Get probability distribution over regimes for latest bar |

Plus `is_fitted` property.

### Design decisions

1. **`covariance_type="diag"` not `"full"`**: Full covariance matrices are
   numerically fragile with only 2 features — Cholesky decomposition fails when
   covariances become near-singular. Diagonal covariance is more stable and
   sufficient since return and volatility are relatively independent features.

2. **Separate `predict_current()` and `predict_proba()`**: `predict_current()`
   returns the single most likely regime (for strategy logic). `predict_proba()`
   returns the full distribution (for position sizing — e.g., scale down when
   regime is uncertain).

3. **Unfitted model raises RuntimeError**: Consistent with the pattern used by
   `MLModel` in `src/ml/`. Calling `predict_current()` or `predict_proba()`
   before `fit_predict()` raises immediately rather than producing garbage.

4. **Observations lose `vol_window` rows**: Building the volatility feature
   requires `vol_window` bars of history. After `dropna()`, the regime series
   is `n_bars - vol_window` long (1 for pct_change + vol_window-1 for rolling).
   This is documented in the returned `RegimeResult.regimes` index.

### Future use (not wired in yet)

Regime detection enables:
- Regime-conditional strategy selection (different strategies for bull vs bear)
- Dynamic position sizing (reduce exposure in bear/high-vol regimes)
- Risk management (tighter drawdown limits when bear regime probability is high)
- Feature engineering (regime as an input feature to ML models)

These integrations are planned for Week 8 or Phase 4.

## Part 2: MLflow Experiment Tracker

### What was built (`src/tracking/mlflow_tracker.py`)

**`ExperimentTracker`** — wraps MLflow with a simplified interface:

```python
tracker = ExperimentTracker(
    experiment_name="trading_system",  # MLflow experiment name
    tracking_uri="mlruns",             # local directory
)
```

**Run lifecycle** — two options:

```python
# Explicit
run_id = tracker.start_run("baseline_backtest")
tracker.log_params({...})
tracker.log_metrics({...})
tracker.end_run()

# Context manager (preferred — guarantees cleanup)
with tracker.run("baseline_backtest") as run_id:
    tracker.log_params({...})
    tracker.log_metrics({...})
```

Starting a run while another is active raises `RuntimeError`.
Context manager calls `end_run()` even if an exception occurs.

**Logging methods:**

| Method | What it does |
|--------|-------------|
| `log_params(dict)` | Logs config params. Nested dicts are flattened with dots: `{"sizing": {"method": "vol"}}` → `"sizing.method": "vol"`. Values > 500 chars are truncated. |
| `log_metrics(dict, step?)` | Logs numeric metrics. Non-numeric and bool values are silently skipped. Optional step for time-series. |
| `log_artifact(path)` | Logs a file. Raises `FileNotFoundError` if missing. |
| `log_backtest_result(config, metrics, trade_metrics?)` | Convenience: logs config as params, metrics as metrics, trade metrics with `trade_` prefix. |
| `log_model_result(name, train_score, val_score?, feature_importance?)` | Convenience: logs model name, scores, top 20 feature importances as metrics. |

**Helper:** `_flatten_dict()` — module-level function that recursively flattens
nested dicts with dot-separated keys.

### Design decisions

1. **Wraps MLflow, doesn't replace it**: `ExperimentTracker` is a thin wrapper.
   Users can still access `mlflow` directly if they need advanced features
   (model registry, remote tracking servers, etc.).

2. **Bool explicitly excluded from metrics**: Python `bool` is a subclass of
   `int`, so `isinstance(True, int)` returns `True`. We check
   `not isinstance(value, bool)` to prevent logging `True`/`False` as 1.0/0.0.

3. **Param truncation at 500 chars**: MLflow has a 500-character limit for
   parameter values. We truncate with `"..."` rather than raising an error.

4. **Feature importance capped at top 20**: With 30+ features, logging all
   importance values clutters the MLflow UI. Top 20 is sufficient for analysis.

5. **Local tracking URI by default**: Uses `mlruns/` directory. Production
   deployments would pass a remote URI (e.g., `http://mlflow-server:5000`).

## Tests

### Regime Detection — 44 tests in `tests/test_regime.py`

| Class | Tests | What |
|-------|-------|------|
| TestRegimeResult | 2 | Creation, all fields populated |
| TestRegimeDetectorInit | 6 | Defaults, custom params, n_regimes < 2, vol_window < 2, is_fitted |
| TestFitPredict | 14 | Returns result, series length, values in range, 2-regime labeling, 3-regime labeling, transition matrix shape, rows sum to 1, means/vols populated, current regime valid, is_fitted after, missing close, insufficient data, bull > bear mean, deterministic |
| TestPredictCurrent | 5 | Returns int, unfitted raises, consistent with fit_predict, subset of data, insufficient data |
| TestPredictProba | 5 | Returns array, sums to 1, non-negative, unfitted raises, argmax matches predict |
| TestBuildObservations | 3 | Columns, length, window effect |
| TestLabelRegimes | 3 | 2 regimes, 3 regimes, 4 regimes |
| TestEdgeCases | 3 | Custom vol_window, many regimes (n=5), refit on new data |
| TestIntegration | 2 | Bull/bear detection on synthetic data, full workflow |

Uses synthetic data generators with clear regime structure:
- `_make_trending_data()`: bull first half (μ=+0.2%, σ=0.8%) + bear second half (μ=-0.3%, σ=1.5%)
- `_make_three_regime_data()`: bull + sideways + bear segments

Integration test uses extreme separation (μ=+0.5%/σ=0.5% vs μ=-0.8%/σ=2.5%)
to ensure HMM reliably detects both regimes.

### Experiment Tracker — 34 tests in `tests/test_experiment_tracker.py`

| Class | Tests | What |
|-------|-------|------|
| TestFlattenDict | 6 | Flat, nested, deeply nested, mixed, empty, list values |
| TestTrackerInit | 2 | Initialization, custom experiment name |
| TestStartEndRun | 4 | Returns ID, clears ID, active run raises, optional name |
| TestContextManager | 2 | Normal flow, cleanup on error |
| TestLogParams | 3 | Simple, nested, long value truncated |
| TestLogMetrics | 5 | Numeric, skips non-numeric, skips bool, with step, integers |
| TestLogArtifact | 3 | Existing file, nonexistent raises, Path object |
| TestLogBacktestResult | 3 | Config + metrics, with trade metrics, filters non-numeric |
| TestLogModelResult | 4 | Basic, with val_score, with feature importance, special chars |
| TestTrackerIntegration | 2 | Full workflow with verification, multiple sequential runs |

Uses `tmp_path` fixture for isolated MLflow directories. Integration test
verifies logged values via `MlflowClient.get_run()`.

## Files

| File | Lines | Action |
|------|-------|--------|
| `src/regime/__init__.py` | 13 | Created — exports RegimeDetector, RegimeResult |
| `src/regime/hmm.py` | 241 | Created — HMM regime detection |
| `src/tracking/__init__.py` | 11 | Created — exports ExperimentTracker |
| `src/tracking/mlflow_tracker.py` | 223 | Created — MLflow wrapper |
| `tests/test_regime.py` | 492 | Created — 44 tests |
| `tests/test_experiment_tracker.py` | 321 | Created — 34 tests |
| `requirements.txt` | 35 | Updated — added hmmlearn>=0.3.0, mlflow>=2.0.0 |

## Running total

**974 tests passing** (896 after Week 6 + 78 new).

`make check` passes: black, ruff, mypy, all tests.

## Usage examples

```python
# Regime detection
from src.regime import RegimeDetector

detector = RegimeDetector(n_regimes=3, vol_window=20)
result = detector.fit_predict(ohlcv_df)

print(f"Current regime: {result.regime_names[result.current_regime]}")
print(f"Transition matrix:\n{result.transition_matrix}")
for r, name in result.regime_names.items():
    print(f"  {name}: mean={result.regime_means[r]:.4f}, vol={result.regime_vols[r]:.4f}")

# Live prediction
current = detector.predict_current(recent_ohlcv)
proba = detector.predict_proba(recent_ohlcv)
print(f"Bear probability: {proba[bear_id]:.1%}")
```

```python
# Experiment tracking
from src.tracking import ExperimentTracker

tracker = ExperimentTracker("my_experiment")

with tracker.run("sma_vol_backtest") as run_id:
    tracker.log_backtest_result(
        config={"strategy": "sma", "sizing": {"method": "volatility"}},
        metrics={"sharpe_ratio": 0.68, "max_drawdown_pct": -12.5},
        trade_metrics={"win_rate_pct": 42.0, "profit_factor": 1.2},
    )
    tracker.log_artifact("output/equity_comparison.png")

# View results: mlflow ui --backend-store-uri mlruns
```

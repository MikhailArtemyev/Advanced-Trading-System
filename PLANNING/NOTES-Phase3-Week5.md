# Phase 3, Week 5 — CPCV Validation Framework

## What this is

Combinatorial Purged Cross-Validation (CPCV) — a financial ML validation method
that prevents information leakage when evaluating models on time-series data.

Standard k-fold CV doesn't work for financial time-series because forward-return
targets overlap across train/test boundaries. If you predict 5-day returns, then
sample N and sample N+3 share 3 days of the same return window. Training on one
and testing on the other leaks information → inflated accuracy.

**Before (Weeks 3-4):**
```
ML models exist but no proper validation method
Could only do train/test split or walk-forward (single path)
No way to know if model accuracy is real or leaked
```

**After (Week 5):**
```
CPCV generates C(N,k) combinatorial train/test paths
Purging removes train samples whose targets overlap with test boundaries
Embargo adds a buffer zone after purge to handle serial correlation
Multi-path evaluation gives robust accuracy estimates with variance
```

## The three mechanisms

### 1. Combinatorial Paths

Instead of a single train/test split, CPCV divides data into N groups and uses
every combination of k groups as test sets. With default N=6, k=2: C(6,2) = 15 paths.

Each path uses k groups for testing and the remaining N-k groups for training.
This gives 15 independent accuracy estimates instead of one — you get a mean
and standard deviation, making it much harder for a model to look good by luck.

### 2. Purging

For each path, samples near train/test boundaries are removed from the train set.
The `purge_window` parameter controls how many samples are removed on each side.

```
Test block:         [  t=50  ...  t=100  ]
Purge zone:    [t=45 ... t=50]     [t=100 ... t=105]
                  ^^removed^^       ^^removed^^
```

Purge is symmetric — removes from both sides of every contiguous test block.
This handles the case where test groups are non-adjacent (e.g., groups 1 and 4
are both test → two separate purge zones).

### 3. Embargo

After purging, an additional buffer is added *after* each test block to address
serial correlation that persists beyond the purge window. The size is
`embargo_pct * len(test_indices)` — proportional to test size.

```
Test block:         [  t=50  ...  t=100  ]
Purge:         [t=45..t=50]         [t=100..t=105]
Embargo:                             [t=105..t=108]
                                      ^^extra buffer^^
```

Embargo applies per contiguous test block, not just at the global test_max.
This is important when test groups are non-adjacent.

## What was built

### CPCVSplit (`src/validation/cpcv.py`)

```python
@dataclass
class CPCVSplit:
    train_indices: np.ndarray   # integer indices for training
    test_indices: np.ndarray    # integer indices for testing
    path_id: int                # sequential path identifier
```

### CPCVResult (`src/validation/cpcv.py`)

```python
@dataclass
class CPCVResult:
    scores: list[float]              # one score per path
    mean_score: float                # average across paths
    std_score: float                 # standard deviation
    n_paths: int                     # total paths evaluated
    train_results: list[TrainResult] # per-path training results
    predictions: pd.Series | None    # concatenated OOS predictions
```

When a sample appears in multiple paths' test sets, the first occurrence is kept
in the predictions series.

### CPCVValidator (`src/validation/cpcv.py`)

```python
validator = CPCVValidator(
    n_splits=6,         # divide data into 6 contiguous groups
    n_test_splits=2,    # use 2 groups as test per path → C(6,2)=15 paths
    purge_window=5,     # remove 5 samples on each side of test boundaries
    embargo_pct=0.01,   # 1% of test size as post-test buffer
)
```

**Two public methods:**

| Method | Purpose |
|--------|---------|
| `get_splits(n_samples)` | Returns list of CPCVSplit with purged/embargoed indices |
| `run(features, target, model, scoring_fn?)` | Full CPCV: train/predict/score on all paths |

**`run()` loop per path:**
1. Split features/target by train/test indices
2. `model.train(train_features, train_target)` — model is retrained every path
3. `model.predict(test_features)` → list of ModelPrediction
4. `scoring_fn(test_target, predictions)` → score
5. Collect OOS predictions (first occurrence wins for duplicates)

**Private methods:**

| Method | Purpose |
|--------|---------|
| `_make_groups(n_samples)` | Divides into N contiguous groups, last absorbs remainder |
| `_purge(train_idx, test_idx, n_samples)` | Removes train samples near test boundaries |
| `_embargo(train_idx, test_idx, n_samples)` | Removes train samples in post-test buffer |

**Helper:** `_find_contiguous_blocks(indices)` — identifies non-adjacent test blocks
so purge/embargo apply correctly around each one.

**Default scoring:** Direction accuracy — fraction of predictions where
`sign(signal) == sign(actual_target)`. Can be replaced with any
`Callable[[pd.Series, list[ModelPrediction]], float]`.

**Input validation:** Checks length match, no empty inputs, no NaN in features or target.

## Key design decisions

1. **Model retrained each path**: The model is trained from scratch on each path's
   train set. This tests generalization, not memorization. It also means CPCV is
   expensive — 15 paths × full training cycle.

2. **Purge is symmetric**: Removes from *both* sides of test boundaries, not just
   before. Forward-return targets look ahead, but features (like rolling means)
   can also leak backward.

3. **Embargo per contiguous block**: If test groups 1 and 4 are selected, embargo
   applies after both blocks separately — not just after the global maximum test index.

4. **Last group absorbs remainder**: `_make_groups(103)` with n_splits=6 gives
   groups of size [17, 17, 17, 17, 17, 18]. Simple and deterministic.

5. **Depends on MLModel ABC**: CPCV imports from `src.ml.base_model`, coupling
   validation to the model interface. This is intentional — CPCV is specifically
   for evaluating MLModel implementations.

## Tests

**64 tests** in `tests/test_cpcv.py`:

| Class | Tests | What |
|-------|-------|------|
| TestCPCVSplit | 2 | Creation, train/test no overlap |
| TestCPCVResult | 2 | Creation, default values |
| TestCPCVValidatorInit | 12 | Valid init, custom params, n_paths property, all validation errors |
| TestFindContiguousBlocks | 6 | Single block, two blocks, single element, all separated, empty, unsorted |
| TestGetSplits | 10 | Correct count, no overlap, all samples in test, indices in range, path IDs, small/invalid n_samples, group boundaries, determinism, remainder |
| TestPurging | 6 | Removes boundary samples, zero window, doesn't remove test, symmetric, non-adjacent blocks, large window |
| TestEmbargo | 5 | Removes post-test, zero pct, multiple blocks, at end of data, combined with purge |
| TestDefaultScoringFn | 4 | Perfect, all wrong, half correct, empty |
| TestCPCVRun | 8 | Returns result, scores populated, mean/std correct, train results, predictions collected, custom scoring, model retrained each path, purge+embargo |
| TestInputValidation | 4 | Length mismatch, empty, NaN features, NaN target |
| TestEdgeCases | 4 | Large purge, 100% embargo, many splits few samples, single test split |
| TestCPCVIntegration | 1 | End-to-end with real XGBoost on synthetic data |

Uses `StubMLModel` (hand-written, returns configurable predictions) and
`PerfectMLModel` (returns target as signal). No real ML libraries needed
for unit tests — only the integration test imports XGBoost.

## Files

| File | Lines | Action |
|------|-------|--------|
| `src/validation/__init__.py` | 25 | Created — exports CPCV + DSR classes |
| `src/validation/cpcv.py` | 402 | Created — full CPCV implementation |
| `tests/test_cpcv.py` | 806 | Created — 64 tests |

## Usage example

```python
from src.validation import CPCVValidator
from src.ml import XGBoostSignalModel

validator = CPCVValidator(
    n_splits=6, n_test_splits=2,
    purge_window=5, embargo_pct=0.01,
)

model = XGBoostSignalModel(mode="classification")
result = validator.run(features, target, model)

print(f"Accuracy: {result.mean_score:.1%} +/- {result.std_score:.1%}")
print(f"Paths evaluated: {result.n_paths}")

# OOS predictions for further analysis
if result.predictions is not None:
    print(f"OOS predictions: {len(result.predictions)} samples")
```

## Reference

de Prado, M. L. (2018). *Advances in Financial Machine Learning.*
Chapter 12: Backtesting through Cross-Validation.

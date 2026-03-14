# ML Strategy Removal

**Date:** 2026-03-14
**Branch:** phase-5

---

## Why It Was Removed

The ML strategy (XGBoost/LightGBM classification with walk-forward training) was removed because it had fundamental design problems that could not be fixed incrementally:

1. **Look-ahead bias in walk-forward training.** The bar counter used to slice the feature cache did not account for leading NaN rows dropped during feature computation. This caused the model to train on ~50 bars of future data, producing unrealistic returns (+3609% on volatile stocks).

2. **Poor real performance after fixing look-ahead.** Once the bias was corrected, the ML strategy consistently underperformed all SMA-based strategies across stable, volatile, and mixed stock groups.

3. **Overfitting on small training sets.** The pipeline generated ~27 features but the initial walk-forward window had only ~140 training samples — far below the 10-20x samples-per-feature ratio needed to generalize.

4. **Excessive trade frequency.** The ML strategy re-evaluated every bar, producing 20x more trades than SMA strategies. Classification probabilities oscillating around the threshold caused rapid entry/exit churn.

5. **Patch-on-patch degradation.** Each fix (signal confirmation, feature reduction, regularization tuning) introduced new problems or killed performance in different ways. The codebase became difficult to reason about.

## What Was Kept

- `src/ml/base_model.py` — the `MLModel` ABC, `ModelPrediction`, and `TrainResult` dataclasses. These are used by the CPCV validator (`src/validation/cpcv.py`) and will serve as the interface for any future ML implementation.

## What Was Removed

- `src/ml/xgboost_model.py`, `src/ml/lightgbm_model.py`, `src/ml/ml_strategy.py`
- `configs/ml_backtest_config.yaml`
- `tests/test_ml_model.py`, `tests/test_ml_strategy.py`
- All ML-specific integration tests and XGBoost integration test in CPCV
- `build_ml_model()`, `build_feature_pipeline()`, and ml_strategy branch in `scripts/run_backtest.py`
- `MLConfig` from `src/config.py`
- `xgboost` and `lightgbm` from `requirements.txt`
- `make run-ml` target from Makefile

## Result

All checks pass (format, lint, type-check, 1394 tests). The system is clean and ready for a properly designed ML strategy in a future phase.

# Full Codebase Refactoring Plan

**Updated:** 2026-03-18
**Scope:** All `src/`, `tests/`, `scripts/`, `configs/`
**Branch:** `refactoring`
**Constraint:** Run `make check` after every step. All passing tests must continue to pass.

---

## Phase 1 — Bugs & Correctness

### 1.1 Config parameter name mismatch (CRITICAL)
- **File:** `configs/paper_trading_config.yaml` lines 17-18
- **Problem:** Uses `fast_period` / `slow_period` but `SMACrossoverStrategy.__init__()` expects `short_window` / `long_window`. Paper trading silently uses default SMA windows, producing incorrect signals.
- **Fix:** Rename to `short_window: 10`, `long_window: 30`.

### 1.2 Bare `except Exception` in YFinanceDataHandler
- **File:** `src/data/yfinance_handler.py` ~line 80
- **Problem:** Swallows all download errors, returns `False` with no logging. Silent data failures lead to empty backtests with zero trades.
- **Fix:** Log the exception with `logger.exception()`, re-raise or return with error context.

### 1.3 `PaperBroker._manual_prices` uses `hasattr()` pattern
- **File:** `src/broker/paper_broker.py` ~line 75
- **Problem:** Lazy initialization via `hasattr()` instead of proper `__init__`. Fragile — will break if attribute naming changes.
- **Fix:** Initialize `_manual_prices: dict[str, float] = {}` in `__init__`.

### 1.4 `PaperBroker._data_handler` assigned via private attribute
- **File:** `scripts/run_paper_trading.py` ~line 172
- **Problem:** `broker._data_handler = data_handler` — couples script to PaperBroker internals.
- **Fix:** Add `data_handler` as a constructor parameter or a public setter method.

---

## Phase 2 — Remove Dead Modules

These modules are well-written but **never called from any production code path** (backtest engine, paper trading, scripts). They were built for an ML pipeline that was removed. Keeping them creates false expectations (config sections suggest they work) and maintenance overhead.

### 2.1 Remove `src/features/` (~500 LOC)
- **Files:** `base_feature.py`, `pipeline.py`, `technical.py`, `statistical.py`, `__init__.py`
- **Evidence:** Not imported in `scripts/run_backtest.py`, `scripts/run_paper_trading.py`, `src/backtest/engine.py`, or any strategy.
- **Also remove:** `tests/test_features_technical.py` (449 lines), `tests/test_features_statistical.py` (460 lines), `tests/test_features_pipeline.py` (331 lines) — ~1,240 lines of tests for dead code.
- **Also remove:** `FeatureConfig` from `src/config.py` and YAML templates.

### 2.2 Remove `src/tracking/` (~200 LOC)
- **Files:** `mlflow_tracker.py`, `__init__.py`
- **Evidence:** Not imported in any script or engine. MLflow was for ML experiment tracking.
- **Also remove:** `tests/test_experiment_tracker.py` (321 lines).
- **Also remove:** `TrackingConfig` from `src/config.py` and YAML templates.

### 2.3 Remove `src/ml/` (~110 LOC)
- **Files:** `base_model.py`, `__init__.py`
- **Evidence:** Only an ABC with zero implementations. Only used by CPCV validator (also being removed).
- **Also remove:** References from config and `__init__.py` exports.

### 2.4 Remove `src/validation/cpcv.py` (~350 LOC)
- **Problem:** CPCV validator requires `MLModel` implementations that don't exist. The validator cannot validate anything.
- **Also remove:** `tests/test_cpcv.py` (752 lines).
- **Keep:** `src/validation/deflated_sharpe.py` — used in `scripts/run_full_report.py`. Remove `ValidationConfig` from config (deflated Sharpe doesn't need config).

### 2.5 Remove `src/regime/` (~240 LOC)
- **Files:** `hmm.py`, `__init__.py`
- **Evidence:** Not called from backtest engine, strategies, or paper trading. Config section (`RegimeConfig`) is parsed but ignored.
- **Also remove:** `tests/test_regime.py` (492 lines).
- **Also remove:** `RegimeConfig` from `src/config.py`.

### 2.6 Remove dead integration test
- **File:** `tests/test_phase3_integration.py` — tests features + regime + CPCV together. All modules being removed.

### 2.7 Clean up config model
- **File:** `src/config.py`
- Remove `FeatureConfig`, `ValidationConfig`, `RegimeConfig`, `TrackingConfig` and their fields in `BacktestConfig`.
- Remove corresponding sections from all YAML configs in `configs/`.
- **Impact:** Config becomes honest — every section maps to real behavior.

### Summary of Phase 2 removals

| Module | Source LOC | Test LOC | Total removed |
|--------|-----------|----------|---------------|
| `src/features/` | ~500 | ~1,240 | ~1,740 |
| `src/tracking/` | ~200 | ~321 | ~521 |
| `src/ml/` | ~110 | 0 (tested via CPCV) | ~110 |
| `src/validation/cpcv.py` | ~350 | ~752 | ~1,102 |
| `src/regime/` | ~240 | ~492 | ~732 |
| Phase 3 integration test | 0 | ~300 | ~300 |
| **Total** | **~1,400** | **~3,105** | **~4,505** |

---

## Phase 3 — Remove Redundancy in Remaining Code

### 3.1 `CorrelationTracker.get_matrix()` is a pure alias
- **File:** `src/portfolio/correlation.py` ~line 126
- `get_matrix()` just calls `calculate_matrix()`. Remove it, update callers.

### 3.2 Duplicated `_equal_weight()` in optimizers
- **Files:** `src/optimization/mean_variance.py` ~line 125, `src/optimization/risk_parity.py` ~line 103
- Identical fallback method. Move to `src/optimization/base_optimizer.py` or a shared utility.

### 3.3 Duplicated constants `_MIN_OBSERVATIONS` and `_TRADING_DAYS`
- **Files:** `src/optimization/mean_variance.py` line 13-14, `src/optimization/risk_parity.py` line 13-14
- Same values. Create `src/optimization/constants.py`.

### 3.4 Duplicated SMA crossover logic
- **Files:** `src/strategy/sma_crossover.py` ~line 91-150, `src/strategy/multi_asset_sma.py` ~line 165-216
- `_check_crossover()` is nearly identical. Extract to a shared utility in `src/strategy/utils.py`.

### 3.5 Dead `LIMIT orders not supported` warning
- **File:** `src/execution/execution_handler.py` ~line 76
- Verify if limit orders are actually supported. If yes, remove the warning. If not, raise properly.

### 3.6 Duplicate/near-duplicate config files
- `configs/backtest_phase2_vol.yaml` ≈ `configs/volatility_sizing.yaml` — consolidate into one and delete the other.

---

## Phase 4 — Error Handling & Robustness

### 4.1 Generic `except Exception` in run_full_report.py
- **File:** `scripts/run_full_report.py` ~line 635
- Replace `print(f"FAILED: {e}")` with `logger.exception()` for stack traces.

### 4.2 No config file existence check
- **Files:** All scripts calling `load_config(args.config)`
- Add `Path(args.config).exists()` check with user-friendly error message.

### 4.3 Hardcoded base prices in synthetic tick generator
- **File:** `scripts/run_paper_trading.py` ~line 236
- `base_prices = {"AAPL": 175.0, "MSFT": 380.0, "GOOGL": 140.0}` — fails for other symbols.
- Derive from config symbols or accept as parameter.

### 4.4 Mixed `print()` vs `logger` in scripts
- **Files:** `scripts/run_backtest.py`, `scripts/run_full_report.py`, `src/backtest/engine.py`
- `BacktestEngine.run()` uses `print()` for progress (lines 185, 217, 222-225). Should use `logger.info()`.
- Standardize: `print()` only for final user-facing summaries, `logger` for everything else.

---

## Phase 5 — Test Refactoring

### 5.1 Create shared `conftest.py` with common fixtures
- **File:** `tests/conftest.py`
- Move duplicated helpers:
  - `FakeDataHandler` (defined identically in `test_mean_reversion.py`, `test_momentum_strategy.py`, `test_pairs_trading.py`)
  - `_make_ohlcv()` (defined in 3 feature test files)
  - `_make_prices()` / `_make_order_event()` / `_mock_strategy()` (scattered across 10+ files)
  - Common constants: `DEFAULT_CAPITAL`, `DEFAULT_PRICE`, `DEFAULT_COMMISSION_PCT`, `SYMBOLS`

### 5.2 Remove trivial / always-true tests
- `test_events.py` lines 18-56: Enum existence tests (`assert EventType.MARKET.value == "MARKET"`) — zero value.
- `test_paper_broker.py` lines 71-85: Pure enum validation.
- `test_portfolio.py` line 240: `assert len(all_signals) >= 0` — always true, should be `> 0`.
- `test_strategy.py` line 330: `assert isinstance(exit_signals, list)` — test content, not type.
- `test_strategy.py` lines 332-377: `test_no_duplicate_long_signals` — can never fail due to implementation.
- `test_config.py` line 191-197: `test_load_actual_config_file` — passes silently when file missing.

### 5.3 Add missing critical-path integration tests
- **BacktestEngine full flow:** Market → Signal → Order → Risk check → Fill → Portfolio update, with real (not mocked) components.
- **Order rejection flow:** Order hits risk limit → rejected → portfolio unchanged.
- **Multi-symbol rebalancing:** Multiple symbols with conflicting signals in one bar.
- **Equity tracking accuracy:** Verify equity after a sequence of buys/sells matches expected math.
- **Strategy signal correctness:** Verify actual signal strengths against known price series.

### 5.4 Reduce excessive mocking in engine tests
- **File:** `tests/test_backtest_engine.py` (565 lines)
- Currently uses 4 mock classes. Tests verify engine "runs without error" but not that results are correct.
- Add integration tests that wire real Portfolio + RiskManager + ExecutionHandler with a small CSV dataset and verify trade history, P&L, and equity curve.

### 5.5 Consolidate config tests
- Three files with overlapping coverage: `test_config.py`, `test_config_validation.py`, `test_config_env_validation.py`.
- Merge into one file or clearly separate by responsibility.

### 5.6 Fix flaky async timing tests
- **File:** `test_paper_engine.py` — timing assertions like `uptime >= 0.05` after `sleep(0.1)`.
- Mock `datetime.now()` for deterministic tests, or use wider margins.

---

## Phase 6 — Minor Polish

### 6.1 Hardcoded output directory in run_full_report.py
- `OUTPUT_DIR = Path("output")` — add `--output` CLI arg.

### 6.2 Makefile `run` target always downloads data
- `run: download-data` — add a check for existing data files.

### 6.3 Hardcoded `CONFIGS` list in run_full_report.py
- Lines 44-53 — no auto-discovery. If a new config is added to `configs/`, it's missed.
- Consider auto-discovering `configs/backtest_*.yaml` or accepting CLI args.

### 6.4 Inconsistent callback naming
- `add_listener()`, `add_fill_callback()`, `add_alert_callback()`, `on_bar` setter — different patterns for the same concept.
- Standardize to `add_*_callback()` or a unified `add_callback(event_type, fn)`.

---

## Execution Order

| Step | Phase | What | Risk | Tests impact |
|------|-------|------|------|-------------|
| 1 | Phase 1 | Fix config param mismatch | Low | 0 |
| 2 | Phase 1 | Fix error handling bugs | Low | ~3 updated |
| 3 | Phase 2 | Remove dead modules + config sections | Medium | ~3,100 tests removed |
| 4 | Phase 2 | Clean up config model | Low | ~10 config tests updated |
| 5 | Phase 3 | Remove code duplication | Low | ~10 tests updated |
| 6 | Phase 3 | Consolidate configs | Low | 0 |
| 7 | Phase 4 | Error handling improvements | Low | ~3 new tests |
| 8 | Phase 4 | Standardize logging | Low | 0 |
| 9 | Phase 5 | Create conftest.py, consolidate helpers | Medium | Many tests updated |
| 10 | Phase 5 | Remove trivial tests, add real ones | Medium | Net positive coverage |
| 11 | Phase 5 | Add integration tests | Low | New tests only |
| 12 | Phase 6 | Polish | Low | 0 |

**After completion:** Update `CLAUDE.md` to reflect the simplified architecture (no features, tracking, regime, ml, or cpcv modules).

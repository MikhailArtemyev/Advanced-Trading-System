# Full Codebase Refactoring Plan

**Prepared:** 2026-03-13
**Scope:** All `src/`, `tests/`, `scripts/`, `configs/`
**Branch:** `phase-4`

---

## Priority 1 — Bugs & Correctness (fix first)

### 1.1 Config parameter name mismatch (CRITICAL)
- **File:** `configs/paper_trading_config.yaml` line 18-19
- **Problem:** Uses `fast_period` / `slow_period` but `SMACrossoverStrategy` expects `short_window` / `long_window`. Paper trading silently uses wrong SMA windows.
- **Fix:** Rename to `short_window: 10`, `long_window: 30`

### 1.2 Unused `rejected_orders` counter in BacktestEngine
- **File:** `src/backtest/engine.py` ~line 166
- **Problem:** Counter is initialized but never incremented or returned.
- **Fix:** Either wire it up (increment in risk manager rejection path) or remove it.

### 1.3 `BarAggregator.reset()` called but not defined
- **File:** `src/live/data_handler.py` ~line 109
- **Problem:** `LiveDataHandler` calls `self._bar_aggregator.reset()` but `BarAggregator` has no `reset()` method.
- **Fix:** Add `reset()` to `BarAggregator` or remove the call.

---

## Priority 2 — Dead Code & Redundancy

### 2.1 `CorrelationTracker.get_matrix()` is a pure alias
- **File:** `src/portfolio/correlation.py` ~line 126
- **Problem:** `get_matrix()` just calls `calculate_matrix()` — redundant wrapper.
- **Fix:** Remove `get_matrix()`, update any callers to use `calculate_matrix()`.

### 2.2 Duplicated `_equal_weight()` in optimizers
- **Files:** `src/optimization/mean_variance.py` ~line 125, `src/optimization/risk_parity.py` ~line 103
- **Problem:** Identical fallback method in both optimizer classes.
- **Fix:** Move to a shared base class or module-level utility.

### 2.3 Duplicated constants `_MIN_OBSERVATIONS` and `_TRADING_DAYS`
- **Files:** `src/optimization/mean_variance.py` line 13-14, `src/optimization/risk_parity.py` line 13-14
- **Problem:** Same magic numbers defined in both files.
- **Fix:** Create `src/optimization/constants.py` with shared constants.

### 2.4 Duplicated SMA crossover logic
- **Files:** `src/strategy/sma_crossover.py` ~line 91-150, `src/strategy/multi_asset_sma.py` ~line 165-216
- **Problem:** `_check_crossover()` implementations are nearly identical.
- **Fix:** Extract shared `_check_crossover()` to `src/strategy/base_strategy.py` or a utility module.

### 2.5 Duplicated ML model signal/confidence mapping
- **Files:** `src/ml/xgboost_model.py` ~line 146, `src/ml/lightgbm_model.py` ~line 141
- **Problem:** Same prediction → signal/confidence conversion logic.
- **Fix:** Move to `src/ml/base_model.py` as a shared method.

### 2.6 Dead `LIMIT orders not supported` warning
- **File:** `src/execution/execution_handler.py` ~line 76
- **Problem:** Hardcoded warning string; limit orders were implemented in later phases but this message may still fire.
- **Fix:** Verify if limit orders are supported; if yes, remove the warning. If not, raise a proper exception.

---

## Priority 3 — Hardcoded Values → Constants/Config

### 3.1 Hardcoded `regression_signal_scale=100.0` in ML models
- **Files:** `src/ml/xgboost_model.py` line 50, `src/ml/lightgbm_model.py` line 51
- **Fix:** Accept as constructor parameter with default.

### 3.2 Hardcoded feature windows
- **Files:** `src/features/technical.py` lines 29-32, `src/features/statistical.py` line 28
- **Problem:** Default windows like `[5, 10, 20, 50]` and horizons `[1, 5, 10, 20]` are class-level hardcoded.
- **Fix:** These are acceptable as defaults, but should be overridable via constructor params (verify they already are — if so, no change needed).

### 3.3 Hardcoded base prices in synthetic tick generator
- **File:** `scripts/run_paper_trading.py` line 168
- **Problem:** `base_prices = {"AAPL": 175.0, "MSFT": 380.0, "GOOGL": 140.0}`
- **Fix:** Accept as parameter or derive from config.

### 3.4 Hardcoded Bollinger Band `num_std=2.0`
- **File:** `src/features/technical.py` ~line 158
- **Fix:** Verify it's a constructor parameter. If not, make it one.

### 3.5 Hardcoded Hurst exponent minimum window
- **File:** `src/features/statistical.py` ~line 134 (`window < 20`)
- **Fix:** Extract to a class constant `_MIN_HURST_WINDOW = 20`.

### 3.6 Hardcoded deflated Sharpe `1e-10` edge case threshold
- **File:** `src/validation/deflated_sharpe.py` ~line 80
- **Fix:** Extract to module constant `_EPSILON = 1e-10`.

---

## Priority 4 — Error Handling Cleanup

### 4.1 Bare `except Exception` in YFinanceDataHandler
- **File:** `src/data/yfinance_handler.py` ~line 80
- **Problem:** Swallows all download errors, returns `False` with no logging.
- **Fix:** Log the exception, re-raise or return with error context.

### 4.2 Silent feature builder failures in run_backtest.py
- **File:** `scripts/run_backtest.py` ~line 155
- **Problem:** Unknown feature types only print a warning, don't raise.
- **Fix:** Use `logger.warning()` instead of `print()`. Optionally raise if no valid features are found.

### 4.3 Generic `except Exception` in run_full_report.py
- **File:** `scripts/run_full_report.py` ~line 635
- **Problem:** Generic catch with `print(f"FAILED: {e}")` — no stack trace.
- **Fix:** Use `logger.exception()` for stack trace.

### 4.4 No config file existence check before `load_config()`
- **Files:** All scripts that call `load_config(args.config)`
- **Fix:** Add `Path(args.config).exists()` check with clear error message.

### 4.5 `PaperBroker._manual_prices` uses `hasattr()` pattern
- **File:** `src/broker/paper_broker.py` ~line 75
- **Problem:** Lazy initialization via `hasattr()` instead of proper `__init__`.
- **Fix:** Initialize `_manual_prices = {}` in `__init__`.

---

## Priority 5 — Inconsistent Patterns

### 5.1 Mixed `print()` vs `logger` usage in scripts
- **Files:** `scripts/run_backtest.py`, `scripts/run_full_report.py`
- **Problem:** Some output uses `print()`, some uses `logger.info()`.
- **Fix:** Use `print()` only for user-facing output (banners, summaries). Use `logger` for everything else.

### 5.2 Inconsistent position access patterns
- **Problem:** Some places use `positions[symbol]`, some use `.get(symbol)`, some use `try/except`.
- **Fix:** Standardize: use `.get()` at boundaries, direct access internally where symbol is guaranteed.

### 5.3 Inconsistent callback naming
- **Problem:** `add_listener()`, `add_fill_callback()`, `add_alert_callback()`, `on_bar` setter — all do the same thing.
- **Fix:** Standardize to `add_callback(event_type, fn)` pattern or at minimum use consistent naming like `add_*_callback()`.

### 5.4 All imports should be absolute (not mixed)
- **Problem:** Some files use relative imports (`from ..events.event`), others use absolute (`from src.events.event`).
- **Fix:** Standardize to relative imports within `src/` packages (already mostly the case), absolute in scripts.

---

## Priority 6 — Test Refactoring

### 6.1 Create shared `conftest.py` with common fixtures
- **File:** `tests/conftest.py` (create or extend)
- **Contents:**
  ```python
  # Constants
  DEFAULT_CAPITAL = 100_000.0
  DEFAULT_PRICE = 150.0
  DEFAULT_COMMISSION_PCT = 0.001
  DEFAULT_QUANTITY = 100
  SYMBOLS = ["AAPL", "MSFT", "GOOGL"]

  # Fixtures
  @pytest.fixture
  def empty_portfolio() -> Portfolio: ...

  @pytest.fixture
  def sample_ohlcv() -> pd.DataFrame: ...

  @pytest.fixture
  def mock_data_handler() -> MagicMock: ...

  @pytest.fixture
  def market_buy_order() -> OrderEvent: ...
  ```

### 6.2 Consolidate duplicated `_make_*` helpers
- **Problem:** `_make_ohlcv()`, `_mock_strategy()`, `_make_order_event()` etc. reimplemented in 10+ test files.
- **Fix:** Move to `conftest.py` as fixtures or helper functions.

### 6.3 Remove dead tests
- `test_strategy.py` lines 332-377: `test_no_duplicate_long_signals` — can never fail.
- `test_config.py` line 191-197: `test_load_actual_config_file` — passes silently when file missing.
- `test_events.py` lines 18-56: Enum existence tests — provide zero value.
- `test_paper_broker.py` lines 71-85: Pure enum validation tests.

### 6.4 Fix always-true assertions
- `test_portfolio.py` line 240: `assert len(all_signals) >= 0` → should be `> 0`.
- `test_strategy.py` line 330: `assert isinstance(exit_signals, list)` → test actual content.

### 6.5 Standardize mock patterns
- **Problem:** Some tests use custom mock classes, some use `MagicMock()`, some use factory functions.
- **Fix:** Standardize to factory functions in `conftest.py` returning configured mocks.

### 6.6 Use `pytest.mark.parametrize` for repetitive test patterns
- Many test classes repeat the same test with different inputs (prices, quantities, symbols).
- Consolidate with `@pytest.mark.parametrize`.

### 6.7 Fix flaky async timing tests
- **File:** `test_paper_engine.py` — timing assertions like `uptime >= 0.05` after `sleep(0.1)`.
- **Fix:** Use wider margins or mock `datetime.now()` for deterministic tests.

---

## Priority 7 — Minor Polish

### 7.1 Abbreviated variable names in math-heavy code
- `mu`, `cov`, `sr`, `t`, `se` in optimization/validation modules.
- **Assessment:** Acceptable in math contexts. Add inline comments where unclear. No rename needed.

### 7.2 Hardcoded output directory in run_full_report.py
- `OUTPUT_DIR = Path("output")` — make configurable via CLI arg.

### 7.3 Makefile `run` target always runs `download-data`
- Make conditional or add a check for existing data.

### 7.4 Duplicate/near-duplicate config files
- `backtest_phase2_vol.yaml` vs `volatility_sizing.yaml` — document purpose or consolidate.

---

## Execution Order

| Step | Priority | Estimated Tests Affected | Risk |
|------|----------|------------------------|------|
| 1. Fix config parameter mismatch (1.1) | P1 | 0 (config only) | Low |
| 2. Fix dead code bugs (1.2, 1.3) | P1 | ~5 new | Low |
| 3. Remove dead code (2.1-2.6) | P2 | ~10 updated | Medium |
| 4. Extract constants (3.1-3.6) | P3 | ~5 updated | Low |
| 5. Error handling (4.1-4.5) | P4 | ~3 new | Low |
| 6. Pattern consistency (5.1-5.4) | P5 | 0 | Low |
| 7. Test refactoring (6.1-6.7) | P6 | Many updated | Medium |
| 8. Polish (7.1-7.4) | P7 | 0 | Low |

**Constraint:** All 1471 existing tests must continue to pass after each step. Run `make check` after every change.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
make check              # Run ALL checks: black --check, ruff, mypy, pytest (use before committing)
make test               # Run all tests (pytest tests/ -v)
make test-fast          # Run tests, stop on first failure (pytest tests/ -x -q)
pytest tests/test_cpcv.py -v                    # Run a single test file
pytest tests/test_cpcv.py::TestPurging -v       # Run a single test class
pytest tests/test_cpcv.py::TestPurging::test_purge_symmetric -v  # Single test
make format             # Auto-format with black
make lint-fix           # Auto-fix lint issues with ruff
make type-check         # mypy src/ --ignore-missing-imports
make report             # Run all 8 configs and generate comparison report + charts
```

## Architecture

Event-driven backtesting engine. The core loop processes events in sequence:

```
MarketEvent → Strategy.calculate_signals() → SignalEvent
→ Portfolio.generate_order() → OrderEvent
→ RiskManager.check_order() → approve/reject
→ ExecutionHandler.execute_order() → FillEvent
→ Portfolio.update_fill()
```

**`src/backtest/engine.py`** — `BacktestEngine` drives this loop. It pulls bars from `DataHandler`, pushes `MarketEvent`s, and dispatches each event type to the appropriate component.

### Key packages

- **`src/events/`** — Event types (Market, Signal, Order, Fill) and the event queue
- **`src/data/`** — `DataHandler` ABC with CSV and YFinance implementations. `get_latest_bars(symbol, n)` is the primary interface strategies use
- **`src/strategy/`** — `Strategy` ABC. Implementations: `SMACrossoverStrategy`, `MultiAssetSMAStrategy`, `MLStrategy`. Strategies emit `SignalEvent`s with `SignalType.LONG/SHORT/EXIT` and a `strength` float
- **`src/portfolio/`** — Manages positions, cash, equity. Calls `PositionSizer` to determine order quantity. Tracks correlation matrix across assets
- **`src/risk/`** — `RiskManager` (drawdown, daily loss, concentration limits) and `PositionSizer` ABC (fixed fraction, volatility-based, Kelly criterion)
- **`src/optimization/`** — `PortfolioOptimizer` ABC with mean-variance and risk-parity implementations. Adjusts target weights across assets
- **`src/features/`** — `FeaturePipeline` orchestrates `BaseFeature` implementations (technical indicators, statistical features). `pipeline.run(ohlcv)` returns features + target; `pipeline.compute_features_only(ohlcv)` for inference
- **`src/ml/`** — `MLModel` ABC with `XGBoostSignalModel` and `LightGBMSignalModel`. Both support classification and regression modes. `MLStrategy` bridges ML predictions to the event loop
- **`src/validation/`** — `CPCVValidator` (combinatorial purged cross-validation) and `deflated_sharpe_ratio()` (multiple testing adjustment)
- **`src/config.py`** — Pydantic models for YAML config. `load_config(path)` returns typed config object

### Configuration

All backtest behavior is driven by YAML files in `configs/`. The config has sections: `data`, `execution`, `strategy`, `sizing`, `risk`, `optimization`. See `configs/backtest_config.yaml` for the default.

### Scripts

- **`scripts/run_backtest.py`** — Main entry point. Has builder functions (`build_data_handler`, `build_strategy`, etc.) that wire config to components
- **`scripts/run_full_report.py`** — Runs all configs, generates `output/strategy_report.txt` + PNG charts

## Code Quality

- **Black** for formatting (line-length 88)
- **Ruff** for linting (pyflakes, pycodestyle, bugbear, isort, comprehensions, pyupgrade)
- **MyPy** with strict settings (`disallow_untyped_defs`, `disallow_incomplete_defs`). Tests are exempt from `disallow_untyped_defs`
- Python 3.11+ required (uses `X | Y` union syntax, `dict[str, Any]` generics)
- All `zip()` calls must use `strict=True` (ruff B905)

## Project Progress

Phased development with detailed plans in `PLANNING/Phase-{1,2,3}-Plan.md` and weekly notes in `PLANNING/NOTES-Phase{N}-Week{M}.md`. Currently on Phase 3 (ML & Advanced Analytics), weeks 1-6 complete, weeks 7-8 remaining.

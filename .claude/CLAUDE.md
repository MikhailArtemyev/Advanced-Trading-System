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
make report             # Run all configs and generate comparison report + charts
make paper              # Run paper trading session (synthetic data)
make run-config CONFIG=path  # Run with a custom config
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
- **`src/strategy/`** — `Strategy` ABC. Implementations: `SMACrossoverStrategy`, `MultiAssetSMAStrategy`. Strategies emit `SignalEvent`s with `SignalType.LONG/SHORT/EXIT` and a `strength` float
- **`src/portfolio/`** — Manages positions, cash, equity. Calls `PositionSizer` to determine order quantity. Tracks correlation matrix across assets
- **`src/risk/`** — `RiskManager` (drawdown, daily loss, concentration limits) and `PositionSizer` ABC (fixed fraction, volatility-based, Kelly criterion)
- **`src/optimization/`** — `PortfolioOptimizer` ABC with mean-variance and risk-parity implementations. Adjusts target weights across assets
- **`src/features/`** — `FeaturePipeline` orchestrates `BaseFeature` implementations (technical indicators, statistical features). `pipeline.run(ohlcv)` returns features + target; `pipeline.compute_features_only(ohlcv)` for inference
- **`src/ml/`** — `MLModel` ABC, `ModelPrediction`, and `TrainResult` dataclasses. Concrete implementations were removed (see `PLANNING/NOTES-ML-Removal.md`); the ABC remains as interface for future ML work and CPCV validator
- **`src/validation/`** — `CPCVValidator` (combinatorial purged cross-validation) and `deflated_sharpe_ratio()` (multiple testing adjustment)
- **`src/regime/`** — `RegimeDetector` using Gaussian HMM (hmmlearn). `fit_predict(ohlcv)` returns `RegimeResult` with regime labels, transition matrix, means/vols. Uses `covariance_type="diag"` (not "full" — numerically fragile with 2 features)
- **`src/tracking/`** — `ExperimentTracker` wrapping MLflow. Context manager API, nested dict flattening, bool filtering
- **`src/config.py`** — Pydantic models for YAML config. `load_config(path)` returns typed config object. Sections: data, execution, strategy, sizing, risk, optimization, features, validation, regime, tracking, live (with sub-sections: data, broker, database, alerts)
- **`src/live/`** — `LiveDataFeed` ABC with `WebSocketDataFeed` and `AlpacaDataFeed` (real-time WebSocket streaming, auto-reconnect). `AlpacaHistoricalClient` for REST bar backfill. `BarAggregator` converts ticks → OHLCV bars. `LiveDataHandler` wraps BarAggregator behind `DataHandler` ABC so strategies work unchanged
- **`src/broker/`** — `BrokerAdapter` ABC with `PaperBroker` (simulated execution) and `AlpacaBroker` (Alpaca REST API, paper + live modes). `OrderManager` state machine: PENDING → SUBMITTED → FILLED/CANCELLED/REJECTED. Bridges OrderEvent ↔ BrokerOrder, BrokerFill → FillEvent
- **`src/engine/`** — `PaperTradingEngine` async event loop (two tasks: bar polling + event processing). Reuses Strategy, Portfolio, RiskManager from backtest. State persistence and crash recovery via `SQLStorage`
- **`src/storage/`** — `StorageBackend` ABC with `SQLStorage` (SQLAlchemy-backed, supports SQLite and PostgreSQL) and `NullStorage` (no-op). Persists trades, equity snapshots, orders, engine state, and sessions
- **`src/alerts/`** — `AlertChannel` ABC with `SlackAlert`, `EmailAlert`, `WebhookAlert`. `AlertManager` routes alerts to channels with level filtering (`INFO`/`WARNING`/`CRITICAL`) and cooldown-based deduplication. Integrates with `HealthMonitor` callbacks
- **`src/monitoring/`** — `HealthMonitor` tracks bar freshness, event latency, order fill rates, feed status. `HealthStatus`: HEALTHY/DEGRADED/UNHEALTHY. Alert callbacks fire on non-healthy state

### Configuration

All backtest behavior is driven by YAML files in `configs/`. The config has sections: `data`, `execution`, `strategy`, `sizing`, `risk`, `optimization`, `features`, `validation`, `regime`, `tracking`, `live`. The `live` section has sub-sections: `data` (feed config), `broker` (paper/alpaca), `database` (SQLite/PostgreSQL persistence), `alerts` (Slack/email/webhook notifications). All sections have defaults so older configs load unchanged.

### Scripts

- **`scripts/run_backtest.py`** — Main entry point. Has builder functions (`build_data_handler`, `build_strategy`, etc.) that wire config to components. Optionally persists results to database when `config.live.database.enabled` is true
- **`scripts/run_full_report.py`** — Runs all configs, generates `output/strategy_report.txt` + PNG charts
- **`scripts/run_paper_trading.py`** — Paper trading session. Wires LiveDataHandler + PaperBroker/AlpacaBroker + PaperTradingEngine with SQLStorage persistence, health monitoring, alert notifications, periodic state saves, and reconciliation on shutdown. Run with `make paper`

## Code Quality

- **Black** for formatting (line-length 88)
- **Ruff** for linting (pyflakes, pycodestyle, bugbear, isort, comprehensions, pyupgrade)
- **MyPy** with strict settings (`disallow_untyped_defs`, `disallow_incomplete_defs`). Tests are exempt from `disallow_untyped_defs`
- Python 3.11+ required (uses `X | Y` union syntax, `dict[str, Any]` generics)
- All `zip()` calls must use `strict=True` (ruff B905)

## Project Progress

Phased development with detailed plans in `PLANNING/Phase-{1,2,3,4,5}-Plan.md` and weekly notes in `PLANNING/NOTES-Phase{N}-Week{S}.md`.

Phase 4 (Paper Trading) is COMPLETE. Phase 5 (Live Broker Integration & Production Readiness) is IN PROGRESS — Weeks 1-4 complete. 1446 tests, all checks green. Key additions: Alpaca broker + data feed, SQLStorage persistence, alerting system.

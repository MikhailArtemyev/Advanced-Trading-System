# Phase 5 — Week 3: SQLite Persistence Layer

**Status:** COMPLETE
**Tests added:** 54
**Cumulative tests:** 1404 (after ML removal reduced count from 1456)

---

## What Was Built

### `src/storage/` — New Package

Replaces the old `StateManager` (JSON file snapshots) with a unified SQLAlchemy-backed persistence layer. Supports SQLite and PostgreSQL via configurable `db_url`.

### `src/storage/models.py` — SQLAlchemy ORM Models

Five tables using SQLAlchemy 2.0 DeclarativeBase + Mapped[] types:

| Table | Key Columns |
|-------|-------------|
| `sessions` | id (UUID PK), session_type, started_at, ended_at, config_json, initial_capital, final_equity |
| `trades` | id, session_id (FK), timestamp, symbol, side, quantity, price, commission, pnl |
| `equity_snapshots` | id, session_id (FK), timestamp, equity, cash, positions_value, num_positions |
| `orders` | id, session_id (FK), order_id, timestamp, symbol, side, order_type, quantity, status, filled_quantity, filled_avg_price |
| `engine_state` | id, session_id (FK), saved_at, cash, bars_processed, events_processed, orders_submitted, orders_rejected, fills_processed, positions_json |

### `src/storage/backend.py` — StorageBackend ABC

Defines the interface:

| Method | Description |
|--------|-------------|
| `create_session()` | Start a new trading session, returns session_id |
| `end_session()` | Mark session complete with final equity |
| `record_trade()` | Persist a single trade |
| `record_equity_snapshot()` | Persist equity/cash/positions snapshot |
| `record_order()` | Persist order state |
| `save_engine_state()` | Persist full engine state for crash recovery |
| `load_engine_state()` | Load latest engine state for a session |
| `get_trades()` | Query trades with optional symbol/date filters |
| `get_equity_curve()` | Returns equity history as pd.DataFrame |
| `get_orders()` | Query orders with optional status filter |
| `list_sessions()` | List all sessions with summary info |
| `close()` | Close database connection |

### `src/storage/sql_storage.py` — SQLStorage

Full implementation of `StorageBackend` using SQLAlchemy sessions. Constructor:

```python
SQLStorage(db_url="sqlite:///trading.db")
```

- Auto-creates tables on init via `Base.metadata.create_all()`
- Applies `check_same_thread=False` only for SQLite URLs
- Supports PostgreSQL via `psycopg2-binary` driver

### `src/storage/null_storage.py` — NullStorage

No-op implementation. All writes are silent, reads return empty results. Used for tests and backtests without database.

### `src/config.py` — Updated

- Removed `PersistenceConfig`
- Added `DatabaseConfig`: `enabled` (bool), `db_url` (str), `save_interval_seconds` (int)
- Updated `LiveConfig`: `database: DatabaseConfig` replaces `persistence: PersistenceConfig`

### `src/portfolio/portfolio.py` — Updated

- Added optional `storage: StorageBackend | None` and `session_id: str` to constructor
- `on_fill()`: after in-memory append, calls `storage.record_trade()` if set
- `update_timeindex()`: after in-memory append, calls `storage.record_equity_snapshot()` if set
- Existing API unchanged — hot-path reads still from in-memory lists

### `src/engine/paper_engine.py` — Updated

- Added `storage` and `session_id` params to constructor
- Added `restore_from_storage()`: loads state from DB and delegates to `restore_state()`
- `get_state()` unchanged — still returns dict for `save_engine_state()`

### `src/backtest/engine.py` — Updated

- Added optional `storage: StorageBackend | None` to constructor
- Creates session at start of `run()`, wires storage+session_id into portfolio
- Bulk-writes trades and equity from in-memory lists at end of `run()`
- Ends session with final equity

### Deleted Files

- `src/engine/state_manager.py` — JSON snapshot persistence (replaced by SQLStorage)
- `tests/test_state_manager.py` — 24 tests (replaced by test_storage.py)

### Scripts Updated

- **`scripts/run_paper_trading.py`** — Imports `SQLStorage` instead of `StateManager`. Creates session on startup, passes storage to Portfolio and PaperTradingEngine. Periodic save via `storage.save_engine_state()`. Restore via `engine.restore_from_storage()`. Shutdown calls `storage.end_session()` + `storage.close()`
- **`scripts/run_backtest.py`** — Optionally creates `SQLStorage` when `config.live.database.enabled` is true, passes to `BacktestEngine`

---

## Tests

### `tests/test_storage.py` — 32 Tests (new)

| Test Class | Count | Coverage |
|------------|-------|----------|
| `TestSessionLifecycle` | 4 | create, end, list, list_empty |
| `TestTrades` | 5 | record_and_get, get_by_symbol, get_by_date_range, get_empty, bulk |
| `TestEquitySnapshots` | 3 | record_and_get_curve, get_empty_curve, bulk |
| `TestOrders` | 2 | record_and_get, get_by_status |
| `TestEngineState` | 4 | save_and_load, load_returns_none, load_returns_latest, round_trip_positions |
| `TestCrossSessions` | 1 | trades_isolated_by_session |
| `TestNullStorage` | 10 | all methods callable, reads return empty |
| `TestPortfolioStorageIntegration` | 2 | portfolio records trades and equity to DB |
| `TestBacktestStorageIntegration` | 1 | backtest persists session to DB |

### Updated Tests

- `tests/test_phase4_integration.py` — `TestStatePersistenceRoundTrip` rewritten for SQLStorage, config compat tests updated, module export tests updated
- `tests/test_stability.py` — Uses SQLStorage for state snapshot tests
- `tests/test_config.py` — `PersistenceConfig` tests replaced with `DatabaseConfig` tests

---

## Design Decisions

1. **Observer-inline pattern** — Portfolio writes to DB inline after in-memory appends. In-memory lists stay for hot-path performance; DB is the durable store
2. **NullStorage for zero-config tests** — all existing tests pass unchanged without a database
3. **Backtest batching** — trades and equity bulk-written at end of `run()`, avoiding per-bar DB overhead
4. **SQLAlchemy as abstraction** — supports SQLite (development) and PostgreSQL (production) with same code
5. **`check_same_thread` conditional** — only applied for SQLite URLs, not PostgreSQL
6. **`psycopg2-binary` added** — PostgreSQL driver in requirements.txt for production use

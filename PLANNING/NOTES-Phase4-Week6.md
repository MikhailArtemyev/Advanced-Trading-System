# Phase 4, Week 6 — State Persistence & Reconciliation

## What this is

JSON-based state snapshots for crash recovery and an auditable position reconciliation
check comparing the Portfolio's view of positions against the PaperBroker's.

**Before:**
```
Engine state is lost when the process stops
No way to verify Portfolio and Broker positions agree
No crash recovery — must restart from scratch
```

**After:**
```
StateManager saves/loads JSON snapshots with timestamps and version metadata
Automatic pruning keeps only the last N snapshots (configurable)
PaperTradingEngine.reconcile_positions() compares Portfolio vs Broker
PaperTradingEngine.get_state() serializes full engine state for snapshots
```

## What was built

### 1. State Manager (`src/engine/state_manager.py`)

Manages timestamped JSON snapshots in a configurable directory:

| Method | Description |
|--------|-------------|
| `save_snapshot(state)` | Write state dict as JSON with `_snapshot_timestamp` and `_snapshot_version` metadata. Auto-prunes oldest beyond limit. |
| `load_latest_snapshot()` | Load most recent snapshot, or `None` if empty |
| `load_snapshot(path)` | Load a specific snapshot file |
| `list_snapshots()` | Return metadata (path, filename, size_bytes, modified) for all snapshots |
| `clear_snapshots()` | Delete all snapshots, return count deleted |
| `_prune_snapshots()` | Remove oldest snapshots beyond `max_snapshots` |

Snapshot files are named `snapshot_YYYYMMDD_HHMMSS_ffffff.json` and contain:

```json
{
  "_snapshot_timestamp": "2025-06-01T14:30:22.123456",
  "_snapshot_version": "1.0",
  "cash": 95000.0,
  "positions": {"AAPL": {"quantity": 50, "avg_cost": 100.0}},
  "statistics": {"bars_processed": 120, "fills_processed": 3},
  "trades": [...],
  "orders": {...}
}
```

### 2. Engine State Serialization (`src/engine/paper_engine.py`)

`get_state()` returns a serializable dict:

- `statistics` — bars_processed, events_processed, orders_submitted/rejected, fills_processed, equity, positions
- `cash` — current cash balance
- `initial_capital` — starting capital
- `trades` — list of trade dicts (timestamp, symbol, side, quantity, price, commission, pnl)
- `orders` — dict of order dicts (order_id, symbol, side, type, quantity, status, filled fields)

### 3. Position Reconciliation (`src/engine/paper_engine.py`)

`reconcile_positions()` compares Portfolio vs PaperBroker positions:

```python
report = await engine.reconcile_positions()
# {
#   "timestamp": "2025-06-01T14:30:22",
#   "matched": True,
#   "symbols_checked": 3,
#   "discrepancies": []    # or list of {symbol, engine_quantity, broker_quantity, difference}
# }
```

In paper mode, positions should always match. Discrepancies indicate a bug in the
event pipeline (e.g., a fill processed by Portfolio but missed by PaperBroker).

## Tests

- **`tests/test_state_manager.py`** — 23 tests: save/load/prune snapshots, metadata, directory creation, round-trip
- **`tests/test_reconciliation.py`** — 14 tests: matching positions, quantity mismatch, symbol only in engine/broker, multiple discrepancies, get_state()

## Files changed

| File | Action | What |
|------|--------|------|
| `src/engine/state_manager.py` | Created | StateManager with JSON snapshots |
| `src/engine/paper_engine.py` | Modified | Added get_state() and reconcile_positions() |
| `src/engine/__init__.py` | Modified | Added StateManager export |
| `tests/test_state_manager.py` | Created | 23 tests |
| `tests/test_reconciliation.py` | Created | 14 tests |

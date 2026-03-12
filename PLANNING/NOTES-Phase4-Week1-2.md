# Phase 4, Weeks 1-2 — Live Data Feed & LiveDataHandler

## What this is

The foundation for paper trading: a real-time data ingestion pipeline that converts
raw market ticks into OHLCV bars and presents them through the same `DataHandler`
interface used by the backtesting engine. This means strategies run unchanged in
both backtest and live modes.

**Before:**
```
Strategies only work with historical CSV/yfinance data
No way to process real-time price updates
No live-mode DataHandler
```

**After:**
```
LiveDataFeed ABC + WebSocketDataFeed for real-time tick streaming
BarAggregator converts ticks → OHLCV bars on configurable intervals
LiveDataHandler wraps BarAggregator behind DataHandler ABC
Config extended with LiveConfig (data feed, broker, persistence sections)
Strategies use get_latest_bars() identically in backtest and live mode
```

## What was built

### 1. Tick & Data Feed (`src/live/data_feed.py`)

- **`Tick`** — frozen dataclass: symbol, timestamp, price, volume, bid, ask
- **`FeedStatus`** — enum: DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING, ERROR
- **`LiveDataFeed`** — ABC defining the live feed interface:
  - `async connect()` / `async disconnect()`
  - `async subscribe(symbols)` / `async unsubscribe(symbols)`
  - `add_listener(callback)` — register tick listeners
- **`WebSocketDataFeed`** — implementation with exponential backoff reconnection

### 2. Bar Aggregator (`src/live/bar_aggregator.py`)

- **`Bar`** — dataclass: symbol, timestamp, OHLCV, tick_count
- **`BarAggregator`** — converts ticks to time-based bars
  - Configurable interval (1s, 1m, 5m, etc.)
  - `on_bar` callback when a bar completes
  - Multi-symbol support with independent bar tracking
  - Gap detection when expected bars are missing
  - `flush()` to force-complete partial bars at session end
  - `get_bars_as_dataframe()` for DataHandler compatibility

### 3. Live Data Handler (`src/live/data_handler.py`)

Bridges BarAggregator to the DataHandler ABC:

| Method | Behavior in Live Mode |
|--------|----------------------|
| `get_latest_bars(symbol, n)` | Returns last n completed bars from aggregator |
| `update_bars()` | Returns True once per new bar (flag-based, resets on read) |
| `get_current_timestamp()` | Returns timestamp of most recent bar |
| `continue_backtest` | Always True — runs until explicitly stopped |
| `bar_count` | Total bars received since start |

### 4. Config Extension (`src/config.py`)

Four new Pydantic models added to `BacktestConfig`:

| Config | Key Fields | Defaults |
|--------|-----------|----------|
| `LiveDataConfig` | feed_type, url, bar_interval_seconds, max_history, reconnect_attempts | websocket, "", 60, 5000, 10 |
| `BrokerConfig` | broker_type, api_key, fill_delay_ms, slippage_model | paper, "", 100, fixed |
| `PersistenceConfig` | enabled, state_dir, save_interval_seconds, max_snapshots | True, "state/", 300, 10 |
| `LiveConfig` | data, broker, persistence | defaults for all sub-configs |

All have `Field(default_factory=...)` so Phase 1-3 configs load unchanged.

## How the pieces connect

```
WebSocketDataFeed           (async tick stream)
      │
      ▼ on_tick()
BarAggregator               (accumulates ticks into bars)
      │
      ▼ on_bar callback
LiveDataHandler             (DataHandler ABC implementation)
      │
      ▼ get_latest_bars()
Strategy                    (unchanged from backtest mode)
```

## Tests

- **`tests/test_data_feed.py`** — ~35 tests: Tick, FeedStatus, listeners, WebSocket connect/disconnect, subscribe, reconnection
- **`tests/test_bar_aggregator.py`** — ~70 tests: bar creation, completion, multi-symbol, gap detection, flush, reset, DataFrame output
- **`tests/test_live_data_handler.py`** — ~45 tests: initialization, get_latest_bars, update_bars, bar_count, multi-symbol, reset
- **`tests/test_config.py`** — ~25 new tests for LiveConfig, BrokerConfig, PersistenceConfig

## Files changed

| File | Action | What |
|------|--------|------|
| `src/live/__init__.py` | Created | Package exports |
| `src/live/data_feed.py` | Created | Tick, FeedStatus, LiveDataFeed ABC, WebSocketDataFeed |
| `src/live/bar_aggregator.py` | Created | Bar, BarAggregator |
| `src/live/data_handler.py` | Created | LiveDataHandler (DataHandler ABC) |
| `src/config.py` | Modified | Added LiveDataConfig, BrokerConfig, PersistenceConfig, LiveConfig |
| `tests/test_data_feed.py` | Created | ~35 tests |
| `tests/test_bar_aggregator.py` | Created | ~70 tests |
| `tests/test_live_data_handler.py` | Created | ~45 tests |
| `tests/test_config.py` | Modified | +25 live config tests |

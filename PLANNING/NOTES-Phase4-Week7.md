# Phase 4, Week 7 — Monitoring & Health Checks

## What this is

Real-time health monitoring for the paper trading engine. Tracks bar freshness,
event processing latency, order fill rates, and data feed status. Fires alert
callbacks when the engine enters a degraded or unhealthy state.

**Before:**
```
No way to know if the engine is healthy during a session
No latency or throughput tracking
No alerting when data feed drops or bars stop arriving
```

**After:**
```
HealthMonitor tracks: bar age, event latency, fill rate, feed status
Three health levels: HEALTHY → DEGRADED → UNHEALTHY
Alert callbacks fire automatically on non-healthy state
HealthReport dataclass with all metrics
Paper trading script with health monitoring, state snapshots, and reconciliation
```

## What was built

### 1. Health Monitor (`src/monitoring/health.py`)

| Class | Description |
|-------|-------------|
| `HealthStatus` | Enum: HEALTHY, DEGRADED, UNHEALTHY |
| `HealthReport` | Dataclass: status, timestamp, uptime, bar_age, latency, bars/min, fill_rate, warnings, errors |
| `HealthMonitor` | Stateful monitor with configurable thresholds |

**Status determination logic:**

```
UNHEALTHY  ←  data feed disconnected
DEGRADED   ←  bar age > threshold  OR  event latency > threshold
HEALTHY    ←  everything else
```

**HealthMonitor methods:**

| Method | Description |
|--------|-------------|
| `start()` | Mark monitoring start time |
| `record_bar(timestamp)` | Record bar processing (updates bar age tracking) |
| `record_event_latency(ms)` | Record event processing time (rolling window) |
| `record_order_result(filled)` | Track fill vs rejection rate |
| `set_data_feed_status(connected)` | Update feed connection status |
| `add_alert_callback(cb)` | Register callback for non-healthy reports |
| `get_health_report()` | Generate current HealthReport, fire alerts if needed |
| `reset()` | Clear all monitoring state |

### 2. Paper Trading Script (`scripts/run_paper_trading.py`)

Full async paper trading session with all Phase 4 components:

```
┌─────────────┐
│ Config YAML │
└──────┬──────┘
       ▼
  build_components()
       │
       ├── LiveDataHandler + BarAggregator
       ├── SMACrossoverStrategy
       ├── Portfolio + PositionSizer
       ├── PaperBroker + OrderManager
       ├── RiskManager
       ├── StateManager
       └── HealthMonitor
       │
       ▼
  asyncio.gather(
    engine.start(),       # bar polling + event processing
    feed_ticks(),         # synthetic tick generation
    periodic_save(),      # state snapshots every N seconds
    monitor_health(),     # periodic health reports
  )
       │
       ▼
  On shutdown:
    - Final state snapshot
    - Position reconciliation
    - Session summary
```

Run with: `make paper`

### 3. Paper Trading Config (`configs/paper_trading_config.yaml`)

```yaml
live:
  data:
    feed_type: "websocket"
    bar_interval_seconds: 60
    max_history: 5000
  broker:
    broker_type: "paper"
    fill_delay_ms: 100
  persistence:
    enabled: true
    state_dir: "state/"
    save_interval_seconds: 300
    max_snapshots: 10
```

## Tests

- **`tests/test_monitoring.py`** — 32 tests: HealthStatus enum, healthy/degraded/unhealthy states, bar tracking, latency averaging, fill rates, bars/min calculation, alert callbacks, reset

## Files changed

| File | Action | What |
|------|--------|------|
| `src/monitoring/__init__.py` | Created | Package exports |
| `src/monitoring/health.py` | Created | HealthMonitor, HealthReport, HealthStatus |
| `scripts/run_paper_trading.py` | Created | Full paper trading script |
| `configs/paper_trading_config.yaml` | Created | Paper trading configuration |
| `Makefile` | Modified | Added `make paper` target |
| `tests/test_monitoring.py` | Created | 32 tests |

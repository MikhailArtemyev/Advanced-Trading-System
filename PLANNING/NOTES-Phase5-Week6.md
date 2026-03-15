# Phase 5 — Week 6: Monitoring Dashboard

**Status:** COMPLETE
**Tests added:** 19
**Cumulative tests:** 1512

---

## What Was Built

### `src/dashboard/terminal_ui.py` — TradingDashboard

Rich-based terminal UI showing real-time paper trading session status.

**Constructor:** `engine: PaperTradingEngine`, `health_monitor: HealthMonitor`, `refresh_rate: float = 1.0`

**Layout (6 panels):**

```
┌────────────────────────────────────────────┐
│  Header: Strategy | Symbols | Uptime       │
├──────────────────────┬─────────────────────┤
│  Equity              │  Recent Trades      │
│  - Equity, Cash      │  - Last 10 trades   │
│  - P&L, Return %     │  - Time/Side/Qty/Px │
├──────────────────────┼─────────────────────┤
│  Positions           │  Health             │
│  - Sym/Qty/Cost/Val  │  - Status/Feed/Lat  │
│  - Unrealized P&L    │  - Bars/min/Fill %  │
├──────────────────────┴─────────────────────┤
│  Footer: Bars | Events | Orders | Fills    │
└────────────────────────────────────────────┘
```

**Lifecycle:**
- `async start()` — runs Live display until `stop()` is called
- Waits for engine to start before entering main loop
- Keeps running after engine stops (shows frozen final state)
- Redirects root logger through `RichHandler` to prevent log/display corruption
- Restores original log handlers on exit
- `stop()` — signals dashboard to exit

### Integration

- `scripts/run_paper_trading.py` — Added `--dashboard` CLI flag
- Dashboard runs as concurrent async task in `asyncio.gather()`
- When dashboard is active, the `monitor_health` log task is skipped (dashboard shows health in panel)
- Shutdown handler calls `dash.stop()`
- Tick pacing: 1 bar/second with dashboard (vs 50ms without) for visible demo

### Makefile

- Added `make paper-dashboard` target

### Dependencies

- Added `rich>=13.0.0` to `requirements.txt`

---

## Tests

### `tests/test_dashboard.py` — 19 Tests

| Test Class | Count | Coverage |
|------------|-------|---------|
| `TestInit` | 2 | Creation, custom refresh rate |
| `TestLayout` | 4 | Returns Layout, calls statistics, with/without positions |
| `TestPanels` | 8 | Header, equity (+/-), trades, health (degraded/unhealthy), stats, zero-qty positions |
| `TestLifecycle` | 4 | stop(), keeps running after engine stops, stop_call, log handler restoration |
| `TestExports` | 1 | Importable from package |

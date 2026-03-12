# Phase 4, Week 5 — Paper Trading Engine

## What this is

The async event-driven engine that ties all paper trading components together.
`PaperTradingEngine` mirrors `BacktestEngine`'s event loop but runs asynchronously
against live data, processing events as they arrive rather than replaying history.

**Before:**
```
BacktestEngine processes historical data synchronously (one bar at a time)
No way to run strategies against live data
Components exist (LiveDataHandler, PaperBroker, OrderManager) but aren't wired up
```

**After:**
```
PaperTradingEngine runs two async loops: bar polling + event processing
Reuses existing Strategy, Portfolio, RiskManager unchanged
Orders route through OrderManager → PaperBroker instead of ExecutionHandler
Engine runs continuously until explicitly stopped
Runtime statistics tracking (bars, events, orders, fills)
Demo script generates synthetic ticks and runs a full paper trading session
```

## What was built

### 1. Paper Trading Engine (`src/engine/paper_engine.py`)

Two concurrent async tasks:

```
┌─────────────────┐     ┌──────────────────────┐
│ _bar_poll_loop   │     │ _event_process_loop   │
│                  │     │                       │
│ polls            │     │ processes events:     │
│ data_handler     │ ──► │  MarketEvent          │
│ .update_bars()   │     │  SignalEvent          │
│                  │     │  OrderEvent           │
│ emits            │     │  FillEvent            │
│ MarketEvent      │     │                       │
└─────────────────┘     └──────────────────────┘
```

Event dispatch matches BacktestEngine exactly:

| Event | Handler | Action |
|-------|---------|--------|
| MarketEvent | `_handle_market()` | Strategy.on_market_data() + Portfolio.update_timeindex() |
| SignalEvent | `_handle_signal()` | Portfolio.on_signal() → generates OrderEvent |
| OrderEvent | `_handle_order()` | RiskManager.check_order() → OrderManager.submit_order_event() |
| FillEvent | `_handle_fill()` | Portfolio.on_fill() + sync position to strategy |

Key differences from BacktestEngine:
- Async (`async def start()`, `await asyncio.gather(...)`)
- Uses `isinstance()` checks for event type narrowing (mypy compatibility)
- Orders go through OrderManager/PaperBroker instead of ExecutionHandler
- Risk manager rejection doesn't stop the engine — just skips that order
- Runs continuously until `stop()` is called

### 2. Demo Script (`scripts/run_paper_demo.py`)

Generates 14,400 synthetic ticks with sine wave price movement, feeds them through
the full pipeline: BarAggregator → LiveDataHandler → PaperTradingEngine with
SMA(5,15) crossover strategy.

## Tests

- **`tests/test_paper_engine.py`** — ~27 tests: initialization, start/stop, bar polling, event handling, risk manager integration, fill event generation, full cycle, LiveDataHandler integration

## Files changed

| File | Action | What |
|------|--------|------|
| `src/engine/__init__.py` | Created | Package exports |
| `src/engine/paper_engine.py` | Created | PaperTradingEngine |
| `scripts/run_paper_demo.py` | Created | Demo with synthetic ticks |
| `Makefile` | Modified | Added `make demo` target |
| `tests/test_paper_engine.py` | Created | ~27 tests |

# Phase 2, Week 6 — Walk-Forward Testing & Multi-Asset SMA Strategy

## What this is

Two pieces that connect the portfolio optimizers from Week 5 into the backtest
engine, and then add a framework to detect whether a strategy is genuinely
profitable or just overfitting to historical data.

**Before (Weeks 1-5):**
```
Optimizers exist but nothing uses them
You backtest over one date range and hope the results generalize
No way to tell if good backtest numbers are real or just curve-fitting
```

**After (Week 6):**
```
MultiAssetSMAStrategy trades multiple stocks with SMA crossovers
  and periodically asks an optimizer "how should I rebalance?"
WalkForwardRunner splits data into rolling train/test windows
  and measures how much in-sample performance persists out-of-sample
```

## What was built

### 1. Multi-Asset SMA Strategy (`src/strategy/multi_asset_sma.py`)

Trades multiple symbols using the same SMA crossover logic as `SMACrossoverStrategy`,
but adds two things: multiple assets at once, and optional weight rebalancing via
a `PortfolioOptimizer`.

How it works:
```
1. Start with equal weights (1/n for each symbol)
2. Every `rebalance_frequency` bars, call optimizer.optimize()
   to get new target weights based on recent returns
3. On each bar, check every symbol for SMA crossover:
   - Short SMA crosses above long SMA → LONG signal
   - Short SMA crosses below long SMA → EXIT signal
4. LONG signal strength = target weight for that symbol
   (so the position sizer can allocate proportionally)
5. EXIT signal strength = 1.0 (always close the full position)
```

| Parameter | Default | What |
|-----------|---------|------|
| `short_window` | 20 | Short moving average period |
| `long_window` | 50 | Long moving average period |
| `rebalance_frequency` | 20 | Bars between optimizer calls |

If no optimizer is provided, weights stay equal forever — it still works fine,
just without the smart rebalancing.

The crossover logic is duplicated from `SMACrossoverStrategy` rather than
inheriting from it. This keeps the class hierarchy flat and avoids diamond
inheritance issues when both need `Strategy` as their base.

### 2. Walk-Forward Runner (`src/backtest/walk_forward.py`)

A framework for detecting overfitting. The core idea: if your strategy looks
great when tested on the data it was trained on, but terrible on fresh data,
it's overfitting.

How it works:
```
Given: 2 years of data

Split into rolling windows:
  Window 1: Train Jan-Jun  →  Test Jul-Aug
  Window 2: Train Mar-Aug  →  Test Sep-Oct
  Window 3: Train May-Oct  →  Test Nov-Dec
  ...

For each window:
  1. Run backtest on training period → get in-sample metrics
  2. Run backtest on testing period  → get out-of-sample metrics

Aggregate:
  Walk-Forward Efficiency = avg OOS return / avg IS return × 100
  Parameter Stability = std(OOS Sharpes) / mean(OOS Sharpes)
```

**Walk-Forward Efficiency (WFE)** tells you how much performance survives
out-of-sample:
- WFE > 50% → strategy likely has a real edge
- WFE 30-50% → uncertain, needs more investigation
- WFE < 30% → probably overfitting

**Parameter Stability** (coefficient of variation of OOS Sharpe ratios) tells
you how consistent the strategy is across windows:
- Lower = more consistent and reliable
- Higher = results vary wildly depending on time period

Three dataclasses support the framework:

**WalkForwardWindow** — one train/test pair:
```python
{
    "window_id": 0,
    "train_start": datetime(2024, 1, 1),
    "train_end": datetime(2024, 6, 30),
    "test_start": datetime(2024, 7, 1),
    "test_end": datetime(2024, 8, 31),
    "train_metrics": {"annualized_return_pct": 15.2, ...},
    "test_metrics": {"annualized_return_pct": 8.1, ...},
}
```

**WalkForwardResult** — aggregated output:
```python
{
    "windows": [...],
    "aggregate_metrics": {
        "num_windows": 5,
        "avg_oos_annualized_return_pct": 7.5,
        "avg_oos_sharpe_ratio": 0.82,
        "avg_oos_max_drawdown_pct": -12.3,
        "avg_oos_volatility_pct": 18.1,
        "pct_profitable_windows": 60.0,
    },
    "walk_forward_efficiency": 52.3,
    "parameter_stability": 0.45,
}
```

The runner uses an **engine factory** pattern — you provide a function that takes
`(start_date, end_date)` strings and returns a `BacktestEngine`. This keeps the
runner decoupled from engine construction details (which data handler, strategy,
portfolio settings, etc.).

Design choices:
- Uses **calendar days** (not trading days) for window sizes — simpler and
  avoids needing a trading calendar
- **Clamps** the last test window to `end_date` if it would slightly overshoot,
  rather than dropping it entirely
- **No parameter re-optimization** between windows — each window gets the same
  strategy config (the runner tests stability, not adaptation)

## How the pieces connect

```
Week 5 optimizers ──→ MultiAssetSMAStrategy._maybe_rebalance()
                        │
                        ▼
              optimizer.optimize(symbols, returns_df, current_weights)
                        │
                        ▼
              target_weights updated → used as signal strength
                        │
                        ▼
              Position sizer uses strength to scale position sizes

WalkForwardRunner
    │
    ▼
  generate_windows() → list of train/test date ranges
    │
    ▼
  For each window: engine_factory(start, end) → BacktestEngine
    │
    ▼
  engine.run() → {"metrics": {...}} → stored per window
    │
    ▼
  _aggregate_results() → WFE + parameter stability + aggregate OOS metrics
```

```
src/strategy/
├── __init__.py            # now exports MultiAssetSMAStrategy
├── base_strategy.py       # Strategy ABC (unchanged)
├── sma_crossover.py       # SMACrossoverStrategy (unchanged)
└── multi_asset_sma.py     # NEW: MultiAssetSMAStrategy

src/backtest/
├── __init__.py            # now exports walk-forward classes
├── engine.py              # BacktestEngine (unchanged)
└── walk_forward.py        # NEW: WalkForwardRunner + dataclasses
```

## Tests

**29 tests** in `tests/test_multi_asset_sma.py`:

*Init (10 tests):*
- Default parameters, custom parameters, validation errors (short >= long,
  short < 1, rebalance < 1), equal initial weights, single symbol, with/without
  optimizer

*Signals (7 tests):*
- No signal on insufficient data, no signal on first bar, generates signals
  over full run, signals for multiple symbols, LONG strength == target weight,
  EXIT strength == 1.0, no duplicate LONG when already positioned

*Rebalance (5 tests):*
- Weights change with optimizer, weights stay equal without optimizer, counter
  resets after rebalance, timing of first rebalance, handles insufficient data

*Returns builder (2 tests):*
- Correct DataFrame shape and columns, empty DataFrame when no data

*Getters (3 tests):*
- get_target_weights returns copy, get_sma_values before/after calculation

*Integration (2 tests):*
- Full BacktestEngine run with mock optimizer
- Full BacktestEngine run with real MeanVarianceOptimizer

**27 tests** in `tests/test_walk_forward.py`:

*Dataclasses (4 tests):*
- WalkForwardWindow creation and defaults, WalkForwardResult creation and defaults

*Init (5 tests):*
- Valid initialization, invalid train/test/step days, start after end

*Window generation (6 tests):*
- Correct number of windows and dates, no windows if range too short, test_end
  clamped to end_date, single window, overlapping windows, sequential IDs

*Run + aggregation (7 tests):*
- Returns WalkForwardResult, metrics populated on each window, WFE calculation
  verified manually, parameter stability verified, aggregate keys present,
  engine without performance tracker, empty windows

*Edge cases (5 tests):*
- Zero IS return → WFE = 0, zero mean Sharpe → stability = 0, single window →
  stability = 0, negative WFE (strategy loses money OOS), pct_profitable_windows

**Total: 468 tests passing, `make check` fully green.**

## Files changed

| File | Action | What |
|------|--------|------|
| `src/strategy/multi_asset_sma.py` | Created | MultiAssetSMAStrategy with optimizer integration |
| `src/backtest/walk_forward.py` | Created | WalkForwardWindow, WalkForwardResult, WalkForwardRunner |
| `tests/test_multi_asset_sma.py` | Created | 29 tests |
| `tests/test_walk_forward.py` | Created | 27 tests |
| `src/strategy/__init__.py` | Modified | Added MultiAssetSMAStrategy export |
| `src/backtest/__init__.py` | Modified | Added walk-forward class exports |

## Usage examples

```python
from src.strategy.multi_asset_sma import MultiAssetSMAStrategy
from src.optimization import MeanVarianceOptimizer

# Without optimizer — equal weights, pure SMA crossover
strategy = MultiAssetSMAStrategy(
    symbols=["AAPL", "MSFT", "GOOGL"],
    parameters={"short_window": 10, "long_window": 30},
)

# With optimizer — rebalances every 20 bars
optimizer = MeanVarianceOptimizer(target="max_sharpe", max_weight=0.50)
strategy = MultiAssetSMAStrategy(
    symbols=["AAPL", "MSFT", "GOOGL"],
    parameters={"short_window": 20, "long_window": 50, "rebalance_frequency": 20},
    optimizer=optimizer,
)

# Check current weights at any time
print(strategy.get_target_weights())
# {"AAPL": 0.35, "MSFT": 0.40, "GOOGL": 0.25}
```

```python
from src.backtest.walk_forward import WalkForwardRunner

# Set up walk-forward analysis
runner = WalkForwardRunner(
    train_days=180,    # 6 months training
    test_days=60,      # 2 months testing
    step_days=60,      # advance 2 months between windows
    start_date="2022-01-01",
    end_date="2024-01-01",
)

# See what windows will be generated
windows = runner.generate_windows()
for w in windows:
    print(f"Train: {w.train_start:%Y-%m-%d} to {w.train_end:%Y-%m-%d}")
    print(f"Test:  {w.test_start:%Y-%m-%d} to {w.test_end:%Y-%m-%d}")

# Run with an engine factory
def make_engine(start_date: str, end_date: str):
    # ... set up data handler, strategy, portfolio, etc.
    return BacktestEngine(data_handler, strategy, portfolio, execution_handler)

result = runner.run(make_engine)

# Interpret results
print(f"WFE: {result.walk_forward_efficiency:.1f}%")
if result.walk_forward_efficiency > 50:
    print("Strategy likely has a genuine edge")
elif result.walk_forward_efficiency < 30:
    print("Strategy is probably overfitting")

print(f"Parameter stability: {result.parameter_stability:.2f}")
print(f"Profitable windows: {result.aggregate_metrics['pct_profitable_windows']:.0f}%")
```

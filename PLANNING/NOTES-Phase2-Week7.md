# Phase 2, Week 7 — Integration & Enhanced Performance Metrics

## What this is

Wiring together all the Phase 2 components so they work end-to-end from a
single config file, and adding richer performance reporting so you can see
what your portfolio is actually doing (how many positions, how much exposure,
how much churn).

**Before (Weeks 1-6):**
```
Optimizers exist but the run script ignores them
Strategy is hardcoded to SMACrossoverStrategy
Performance report shows returns and Sharpe but nothing about
  position counts, exposure levels, or portfolio turnover
Run script prints "optimization not yet implemented" warning
```

**After (Week 7):**
```
Run script auto-selects strategy based on config:
  1 symbol or no optimizer → SMACrossoverStrategy
  Multiple symbols + optimizer → MultiAssetSMAStrategy
All Phase 2 components wired via factory functions
Performance report now includes extended risk metrics
Everything driven by configs/backtest_config.yaml
```

## What was built

### 1. Updated Run Script (`scripts/run_backtest.py`)

Two new factory functions that complete the config-driven pipeline:

**`build_optimizer(config)`** — reads `config.optimization.method` and creates
the right optimizer:

| Config value | Result |
|-------------|--------|
| `"none"` | `None` — no optimization |
| `"mean_variance"` | `MeanVarianceOptimizer(**params)` |
| `"risk_parity"` | `RiskParityOptimizer(**params)` |

Any parameters under `optimization.parameters` in the YAML get passed directly
to the optimizer constructor (e.g. `target: "min_variance"`, `max_weight: 0.5`).

**`build_strategy(config, optimizer)`** — auto-selects the strategy:
```
if len(symbols) > 1 AND optimizer is not None:
    → MultiAssetSMAStrategy (with optimizer for periodic rebalancing)
else:
    → SMACrossoverStrategy (classic single-asset or equal-weight multi-asset)
```

The `rebalance_frequency` flows from `config.optimization.rebalance_frequency`
into the strategy parameters, but if you explicitly set it in
`strategy.parameters`, that takes precedence (via `dict.setdefault`).

The run script now also prints strategy name, optimization method, and
rebalance frequency at the end of each run.

### 2. Enhanced Performance Metrics (`src/performance/metrics.py`)

**`calculate_extended_metrics(trades, portfolio_equity_history)`** — builds on
`calculate_metrics()` and adds two sub-dicts:

`risk_metrics`:
```python
{
    "avg_position_count": 2.3,       # mean positions held across all bars
    "max_position_count": 5,         # peak concurrent positions
    "gross_exposure_avg": 0.45,      # mean of |positions_value|/equity per bar
    "turnover": 3.78,               # annualized portfolio turnover
}
```

`trade_metrics` (same as `calculate_trade_metrics` output, included when trades
are non-empty):
```python
{
    "total_trades": 12,
    "win_rate_pct": 58.3,
    "profit_factor": 1.8,
    # ... etc
}
```

Position counts and exposure come from the portfolio's `equity_history`, which
already records `num_positions` and `positions_value` per bar — so no changes
to Portfolio were needed.

**`_calculate_turnover(trades, portfolio_equity_history)`** — annualized
portfolio turnover:
```
turnover = (total_traded_value / avg_equity) * (252 / trading_days)
```

Where `total_traded_value` is the sum of `|price × quantity|` across all trades.
High turnover means the strategy trades frequently relative to portfolio size.

**Updated `print_report(trades, portfolio_equity_history)`** — new optional
parameter `portfolio_equity_history`. When provided, it triggers extended
metrics and prints the additional risk section. Fully backward compatible —
existing callers that only pass `trades` get the same output as before.

The run script now passes `portfolio.equity_history` to `print_report`, so
every backtest run automatically shows the extended metrics.

## How the pieces connect

```
backtest_config.yaml
    │
    ▼
load_config() → BacktestConfig
    │
    ├── build_data_handler()     → DataHandler (csv or yfinance)
    ├── build_position_sizer()   → PositionSizer (fixed, vol, kelly)
    ├── build_risk_manager()     → RiskManager | None
    ├── build_optimizer()        → PortfolioOptimizer | None      ← NEW
    └── build_strategy()         → Strategy (auto-selected)       ← NEW
            │
            ▼
    BacktestEngine.run()
            │
            ▼
    print_report(trades, portfolio.equity_history)                ← UPDATED
            │
            ▼
    Extended report with risk metrics + trade metrics
```

The factory pattern means every component is config-driven. To switch from
equal-weight SMA to optimizer-rebalanced multi-asset, you just change:
```yaml
optimization:
  method: "mean_variance"   # was "none"
  rebalance_frequency: 20
  parameters:
    target: "max_sharpe"
    max_weight: 0.40
```

## Tests

**28 tests** in `tests/test_week7.py`:

*TestBuildOptimizer (5 tests):*
- `none` returns None
- `mean_variance` with defaults and custom params
- `risk_parity` with defaults and custom params

*TestBuildStrategy (6 tests):*
- Single symbol + no optimizer → SMACrossoverStrategy
- Multi symbol + no optimizer → SMACrossoverStrategy
- Single symbol + optimizer → SMACrossoverStrategy (need >1 symbol)
- Multi symbol + optimizer → MultiAssetSMAStrategy with optimizer attached
- rebalance_frequency flows from optimization config
- Strategy-level rebalance_frequency overrides optimization config

*TestExtendedMetrics (8 tests):*
- Includes all base metric keys
- risk_metrics sub-dict present with all expected keys
- avg_position_count and max_position_count computed correctly
- trade_metrics included/absent based on trades list
- No portfolio history → position metrics omitted, turnover still works
- Insufficient data → error dict

*TestCalculateTurnover (6 tests):*
- No trades → 0.0
- Basic calculation verified against manual math
- Uses portfolio_history for avg_equity when provided
- Zero equity → 0.0, no equity data → 0.0
- Multiple trades summed correctly

*TestPrintReportExtended (3 tests):*
- With portfolio history → shows extended metrics
- Without history → no extended metrics, trade metrics still shown
- Backward-compatible call still works

**Total: 496 tests passing, `make check` fully green.**

## Files changed

| File | Action | What |
|------|--------|------|
| `scripts/run_backtest.py` | Modified | Added `build_optimizer`, `build_strategy` factories; wired into `main()`; removed optimization warning; added strategy/optimizer info output |
| `src/performance/metrics.py` | Modified | Added `calculate_extended_metrics`, `_calculate_turnover`; updated `print_report` for extended metrics |
| `configs/backtest_config.yaml` | Modified | Removed "(Week 5)" placeholder comment |
| `tests/test_week7.py` | Created | 28 tests |

## Usage examples

```yaml
# configs/backtest_config.yaml — enable optimizer
optimization:
  method: "mean_variance"        # or "risk_parity"
  rebalance_frequency: 20
  parameters:
    target: "max_sharpe"
    max_weight: 0.40
```

```bash
# Run with default config (no optimization, SMACrossoverStrategy)
python scripts/run_backtest.py --config configs/backtest_config.yaml

# Output now includes:
# Risk Metrics:
#   Volatility:        15.23%
#   Max Drawdown:      -8.45%
#   Avg Positions:     1.2
#   Max Positions:     3
#   Avg Exposure:      9.5%
#   Turnover (ann.):   2.34
#
# Strategy:           SMACrossoverStrategy
# Position Sizing:    fixed_fraction
# Risk Manager:       enabled
# Optimization:       none
```

```python
# Using extended metrics programmatically
from src.performance.metrics import PerformanceTracker

tracker = PerformanceTracker(initial_capital=100000.0)
# ... run backtest ...

# Base metrics (unchanged)
metrics = tracker.calculate_metrics()

# Extended metrics (new)
extended = tracker.calculate_extended_metrics(
    trades=trade_list,
    portfolio_equity_history=portfolio.equity_history,
)
print(extended["risk_metrics"]["turnover"])        # annualized turnover
print(extended["risk_metrics"]["avg_position_count"])  # mean positions
print(extended["trade_metrics"]["win_rate_pct"])   # win rate
```

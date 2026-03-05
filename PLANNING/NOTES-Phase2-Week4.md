# Phase 2, Week 4 — Extended Configuration & Data Source

## What this is

Week 4 connects the pieces built in Weeks 1-3. Instead of hardcoding which position
sizer or risk limits to use, you now configure them in the YAML file. Also adds a
data handler that downloads from Yahoo Finance automatically.

**Before (Weeks 1-3):**
```
Config only had: data, execution, strategy sections
run_backtest.py hardcoded: HistoricalCSVDataHandler, no sizer, no risk manager
You had to run download_data.py manually before backtesting
```

**After (Week 4):**
```
Config has: data, execution, strategy, sizing, risk, optimization sections
run_backtest.py reads config and builds the right components via factory functions
Set data_source: "yfinance" and data downloads automatically with caching
```

## What was built

### 1. Extended config schema (`src/config.py`)

Three new Pydantic models added to the config system:

**SizingConfig** — which position sizer to use and its parameters:
```yaml
sizing:
  method: "fixed_fraction"    # or "volatility" or "kelly"
  parameters:
    fraction: 0.10            # passed directly to the sizer constructor
```

The `parameters` dict gets unpacked as `**kwargs` into the sizer's `__init__`. So
for volatility-based sizing you'd write:
```yaml
sizing:
  method: "volatility"
  parameters:
    risk_fraction: 0.03
    atr_period: 14
    atr_multiplier: 2.0
```

**RiskConfig** — risk manager limits, configurable per-backtest:
```yaml
risk:
  enabled: true               # set to false to skip all risk checks
  max_position_pct: 0.10      # max single position as % of equity
  max_portfolio_exposure_pct: 1.0
  max_daily_loss_pct: 0.03
  max_drawdown_pct: 0.15
  max_open_positions: 20
  max_orders_per_day: 100
```

The defaults are identical to the `RiskLimits` dataclass from Week 2. Pydantic
validates all percentage fields: must be > 0 and <= 1.0. Integer fields must be >= 1.

**OptimizationConfig** — placeholder for Week 5:
```yaml
optimization:
  method: "none"              # none, mean_variance, risk_parity
  rebalance_frequency: 20
  parameters: {}
```

If you set `method` to anything other than `"none"`, the script prints a warning
saying it's not implemented yet.

**Backwards compatible**: all three sections have `Field(default_factory=...)` in
`BacktestConfig`. Existing YAML files that don't include these sections still load
fine — Pydantic fills in the defaults.

### 2. YFinance data handler (`src/data/yfinance_handler.py`)

A new `YFinanceDataHandler` that downloads OHLCV data from Yahoo Finance and
caches it as CSV files. Uses **composition** — wraps `HistoricalCSVDataHandler`
rather than inheriting from it.

How it works:
```
YFinanceDataHandler.__init__()
  |
  for each symbol:
  |   does cached CSV exist and cover the date range?
  |     yes → skip download
  |     no  → download via yfinance, save as CSV
  |
  create internal HistoricalCSVDataHandler(cache_dir, ...)
  |
  all method calls (get_latest_bars, update_bars, etc.)
    → delegated to the internal CSV handler
```

To use it, just change one line in the config:
```yaml
data:
  data_source: "yfinance"     # instead of "csv"
  data_path: "data/cache"     # where to cache downloaded CSVs
```

The cached CSV format is identical to what `download_data.py` produces: lowercase
columns `date,open,high,low,close,volume`. So you can switch between `csv` and
`yfinance` data sources without changing anything else.

Cache validation checks that the existing CSV's date range fully covers the
requested range. If you change `start_date` or `end_date` in the config to extend
beyond what's cached, it re-downloads.

### 3. Updated run script (`scripts/run_backtest.py`)

Three factory functions that read the config and build the right components:

| Function | What it does |
|----------|-------------|
| `build_data_handler(config)` | Returns `HistoricalCSVDataHandler` or `YFinanceDataHandler` based on `data_source` |
| `build_position_sizer(config)` | Returns `FixedFractionSizer`, `VolatilityBasedSizer`, or `KellyCriterionSizer` based on `sizing.method` |
| `build_risk_manager(config)` | Returns `RiskManager` with custom limits, or `None` if `risk.enabled` is false |

The `main()` function now:
- Uses the factories instead of hardcoded constructors
- Passes `position_sizer` to Portfolio
- Passes `risk_manager` to BacktestEngine
- Prints a risk summary after the run (rejected orders, halt status)
- Prints a portfolio summary (long/short exposure, positions)
- Shows which sizing method and risk manager state were used

### 4. Updated config file (`configs/backtest_config.yaml`)

The example config now has all six sections:
```yaml
data:         # symbols, dates, data source
execution:    # capital, commission, slippage
strategy:     # strategy name and params
sizing:       # position sizer method and params     ← new
risk:         # risk limits and enabled flag          ← new
optimization: # placeholder for Week 5               ← new
```

## How the pieces connect

```
backtest_config.yaml
  |
  load_config() → BacktestConfig (validated by Pydantic)
  |
  build_data_handler()    → HistoricalCSVDataHandler or YFinanceDataHandler
  build_position_sizer()  → FixedFractionSizer / VolatilityBasedSizer / KellyCriterionSizer
  build_risk_manager()    → RiskManager (or None if disabled)
  |
  BacktestEngine(data_handler, strategy, portfolio, execution_handler,
                 performance_tracker, risk_manager)
  |
  engine.run()
  |
  print: performance report, risk summary, portfolio summary, execution stats
```

## Tests

**20 new tests** in `tests/test_config.py`:
- SizingConfig: defaults, valid/invalid methods, with parameters
- RiskConfig: defaults match RiskLimits, disabled flag, invalid percentages (0, >1, negative), boundary values
- OptimizationConfig: defaults, valid/invalid methods, invalid rebalance frequency
- BacktestConfigExtended: backwards compat without new sections, with all sections, YAML loading both old and new format

**10 new tests** in `tests/test_yfinance_handler.py` (all mock yfinance — no network):
- Downloads when no cache exists
- Skips download when cache is valid
- Re-downloads when cache doesn't cover requested range
- Raises ValueError on empty download
- Delegation works: get_latest_bars, update_bars, continue_backtest, get_current_timestamp, reset
- CSV format: correct columns, all lowercase

**8 new tests** in `tests/test_run_backtest_factories.py`:
- build_position_sizer: default fixed fraction, with custom params, volatility, kelly
- build_risk_manager: default enabled, disabled returns None, custom limits
- build_data_handler: unknown source raises ValueError

**Total: 388 tests passing, `make check` fully green.**

## Files changed

| File | Action | What |
|------|--------|------|
| `src/config.py` | Modified | Added SizingConfig, RiskConfig, OptimizationConfig; updated BacktestConfig |
| `src/data/yfinance_handler.py` | Created | YFinanceDataHandler with caching and CSV delegation |
| `src/data/__init__.py` | Modified | Added exports (DataHandler, HistoricalCSVDataHandler, YFinanceDataHandler) |
| `scripts/run_backtest.py` | Modified | Factory functions, risk/portfolio summary, sizing info |
| `configs/backtest_config.yaml` | Modified | Added sizing, risk, optimization sections |
| `tests/test_config.py` | Modified | 20 new tests |
| `tests/test_yfinance_handler.py` | Created | 10 tests |
| `tests/test_run_backtest_factories.py` | Created | 8 tests |

## Usage examples

```yaml
# Use volatility-based sizing with custom ATR settings
sizing:
  method: "volatility"
  parameters:
    risk_fraction: 0.02
    atr_period: 20
    atr_multiplier: 3.0

# Use Kelly criterion
sizing:
  method: "kelly"
  parameters:
    kelly_fraction: 0.25
    max_position_fraction: 0.15
```

```yaml
# Tight risk limits for conservative backtesting
risk:
  enabled: true
  max_position_pct: 0.05
  max_drawdown_pct: 0.10
  max_open_positions: 5
  max_orders_per_day: 20
```

```yaml
# Disable risk manager entirely
risk:
  enabled: false
```

```yaml
# Auto-download data from Yahoo Finance
data:
  symbols: [AAPL, MSFT, GOOGL, AMZN, TSLA]
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  data_source: "yfinance"
  data_path: "data/cache"
```


### Total Return: 12.2%
# Trading System MVP

Event-driven backtesting engine for systematic trading strategies.

## How It Works

This is a backtester — it replays historical stock data and simulates what would happen if you followed a trading strategy. You give it price data and a strategy, it tells you how much money you would have made (or lost).

### The Event Loop

Everything runs on a simple loop that processes one price bar (one day) at a time:

```
For each day of historical data:
  1. MarketEvent  — "here's today's price data"
  2. SignalEvent  — strategy looks at prices, decides: buy, sell, or do nothing
  3. OrderEvent   — portfolio turns that decision into a concrete order (how many shares)
     └─ RiskManager checks the order (concentration, exposure, drawdown, daily loss)
  4. FillEvent    — execution handler simulates filling the order (with slippage + commission)
```

Events go into a queue and get processed one by one before moving to the next day.

### Key Components

- **DataHandler** — loads CSV or yfinance price data, feeds it bar-by-bar. Only gives the strategy data up to "today" (no peeking at the future).
- **Strategy** — decides when to buy/sell. SMA crossover or multi-asset SMA with portfolio optimization.
- **Portfolio** — tracks positions (long and short), cash, and P&L. Converts signals into sized orders using a PositionSizer.
- **PositionSizer** — determines how many shares to trade. Three methods: fixed fraction, volatility-based (ATR), and Kelly criterion.
- **RiskManager** — pre-trade risk checks: max drawdown circuit breaker, daily loss limit, order rate limit, position concentration, portfolio exposure.
- **ExecutionHandler** — simulates filling orders with realistic slippage and commission costs.
- **PortfolioOptimizer** — computes target allocation weights across multiple assets. Mean-variance (max Sharpe or min variance) and risk parity methods.
- **CorrelationTracker** — tracks rolling pairwise correlations between assets for portfolio monitoring.
- **WalkForwardRunner** — walk-forward analysis with rolling train/test windows and walk-forward efficiency (WFE) calculation.
- **PerformanceTracker** — calculates metrics (Sharpe, Sortino, Calmar, max drawdown, win rate, turnover, etc.) and generates reports.
- **BacktestEngine** — wires everything together and runs the loop.

## Quick Start

```bash
make run
```

This downloads data (if needed) and runs the backtest using `configs/backtest_config.yaml`. Edit that file to change symbols, dates, or strategy parameters.

Or step by step:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download sample data:
   ```bash
   make download-data
   ```

3. Run backtest:
   ```bash
   make run
   ```

## Project Structure

```
trading_system/
├── src/
│   ├── events/             # Event types and queue
│   ├── data/               # Data handlers (CSV, yfinance)
│   ├── strategy/           # Trading strategies (SMA crossover, multi-asset SMA)
│   ├── portfolio/          # Portfolio management, correlation tracking
│   ├── risk/               # Position sizing and risk management
│   ├── optimization/       # Portfolio optimization (mean-variance, risk parity)
│   ├── execution/          # Order execution simulation
│   ├── performance/        # Performance metrics and visualization
│   └── backtest/           # Backtest engine and walk-forward analysis
├── tests/                  # 600+ unit and integration tests
├── configs/                # Example configuration files
├── scripts/                # Utility scripts (run_backtest, download_data)
├── data/                   # Market data (not in git)
├── notebooks/              # Jupyter notebooks
└── output/                 # Backtest results
```

## Configuration

All configuration is done via YAML files. See `configs/` for examples.

### Sections

**data** — Data source and date range:
```yaml
data:
  symbols: [AAPL, MSFT, GOOGL]
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  data_source: "csv"       # csv or yfinance
  data_path: "data/sample"
```

**execution** — Capital and transaction costs:
```yaml
execution:
  initial_capital: 100000.0
  commission_pct: 0.001     # 0.1%
  slippage_pct: 0.0005      # 0.05%
```

**strategy** — Strategy selection and parameters:
```yaml
strategy:
  name: "sma_crossover"
  parameters:
    short_window: 20
    long_window: 50
```

**sizing** — Position sizing method:
```yaml
sizing:
  method: "fixed_fraction"   # fixed_fraction, volatility, or kelly
  parameters:
    fraction: 0.10           # 10% of equity per position
```

**risk** — Risk management limits:
```yaml
risk:
  enabled: true
  max_position_pct: 0.10     # Max 10% in one position
  max_portfolio_exposure_pct: 1.0
  max_daily_loss_pct: 0.03   # Halt after 3% daily loss
  max_drawdown_pct: 0.15     # Circuit breaker at 15% drawdown
  max_open_positions: 20
  max_orders_per_day: 100
```

**optimization** — Portfolio optimization (multi-asset):
```yaml
optimization:
  method: "mean_variance"    # none, mean_variance, or risk_parity
  rebalance_frequency: 20    # Reoptimize every 20 bars
  parameters:
    target: "max_sharpe"     # max_sharpe or min_variance
    max_weight: 0.40
```

### Example Configs

| Config | Description |
|--------|-------------|
| `backtest_config.yaml` | Default single-asset backtest |
| `kelly_sizing.yaml` | Kelly criterion position sizing |
| `volatility_sizing.yaml` | ATR-based volatility sizing |
| `mean_variance_optimization.yaml` | Mean-variance portfolio optimization |
| `risk_parity_optimization.yaml` | Risk parity allocation |
| `conservative_risk.yaml` | Tight risk limits example |

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Full Check (format + lint + types + tests)
```bash
make check
```

### Code Formatting
```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking
```bash
mypy src/
```

## Adding New Strategies

1. Create new file in `src/strategy/`
2. Inherit from `Strategy` base class
3. Implement `calculate_signals()` method

## Phase 2 Status

COMPLETE — 600+ tests, all checks green.

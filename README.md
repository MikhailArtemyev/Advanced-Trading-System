# Advanced Trading System

An event-driven backtesting and paper-trading engine built around one question: **how do you know a
backtest result is real?**

Most backtesters will happily tell you that a strategy made money. This one is built to tell you when it
didn't — when the result came from leakage, from overlapping labels, or from trying enough strategies
that one of them was always going to look good.

## How it answers that

**Combinatorial purged cross-validation, with embargo.** Financial labels overlap in time, so an ordinary
train/test split leaks the future into the past. CPCV generates C(N, k) train/test paths, purges training
samples whose label windows overlap the test set, and embargoes a buffer after each test fold.

**Deflated Sharpe Ratio.** A Sharpe of 2 means little if it is the best of two hundred attempts. DSR
adjusts the observed Sharpe for the number of trials, the skew and kurtosis of the returns and the length
of the sample, then asks whether the result survives.

**Walk-forward efficiency.** Compares out-of-sample against in-sample performance across rolling windows.
A strategy that only works in-sample fails here visibly.

Everything else in the repo — the feature pipeline, the ML models, the regime detection, the execution
layer — exists to feed those three checks something worth checking.

## At a glance

|  |  |
|---|---|
| Source | 68 files, ~10,800 lines of Python |
| Tests | **1,249**, across 50 files |
| CI | black · ruff · mypy · pytest on a Python matrix · coverage, on every push |
| Requires | Python 3.11+ |
| Modes | Historical backtest · paper trading against live data with a simulated broker |

<!-- TODO: add an equity-curve figure here. One plot from a walk-forward run, saved to docs/equity.png,
     then:  ![Walk-forward equity curve](docs/equity.png)
     This is the single highest-value addition to this file. -->

## How It Works

This is a backtester and paper trading engine — it can replay historical stock data to simulate strategy performance, or run strategies in real-time against live data with a simulated broker. You give it price data and a strategy, it tells you how much money you would have made (or lost).

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
- **Strategy** — decides when to buy/sell. SMA crossover, multi-asset SMA with portfolio optimization, or ML-driven signal generation.
- **Portfolio** — tracks positions (long and short), cash, and P&L. Converts signals into sized orders using a PositionSizer.
- **PositionSizer** — determines how many shares to trade. Three methods: fixed fraction, volatility-based (ATR), and Kelly criterion.
- **RiskManager** — pre-trade risk checks: max drawdown circuit breaker, daily loss limit, order rate limit, position concentration, portfolio exposure.
- **ExecutionHandler** — simulates filling orders with realistic slippage and commission costs.
- **PortfolioOptimizer** — computes target allocation weights across multiple assets. Mean-variance (max Sharpe or min variance) and risk parity methods.
- **FeaturePipeline** — composable feature engineering: 10 generators (SMA, RSI, MACD, Bollinger, ATR, returns, z-score, higher moments, Hurst, volatility). Produces clean, NaN-free feature matrices for ML models.
- **ML Models** — XGBoost and LightGBM signal models behind a common `MLModel` ABC. Classification (direction) or regression (return) modes. `MLStrategy` bridges predictions into the event loop.
- **CPCV Validator** — combinatorial purged cross-validation for financial ML. Generates C(N,k) train/test paths with purging and embargo to prevent information leakage.
- **Deflated Sharpe Ratio** — statistical test adjusting Sharpe ratios for multiple testing bias. Answers: "is this Sharpe real, or just luck from testing many strategies?"
- **RegimeDetector** — Hidden Markov Model that classifies market periods as bull, bear, or sideways from return and volatility observations.
- **ExperimentTracker** — MLflow wrapper for logging backtest parameters, metrics, and artifacts across runs.
- **CorrelationTracker** — tracks rolling pairwise correlations between assets for portfolio monitoring.
- **WalkForwardRunner** — walk-forward analysis with rolling train/test windows and walk-forward efficiency (WFE) calculation.
- **PerformanceTracker** — calculates metrics (Sharpe, Sortino, Calmar, max drawdown, win rate, turnover, etc.) and generates reports.
- **BacktestEngine** — wires everything together and runs the historical backtest loop.
- **LiveDataFeed** — WebSocket-based tick streaming with automatic reconnection and exponential backoff.
- **BarAggregator** — converts raw ticks into OHLCV bars on configurable intervals (1s, 1m, 5m, etc.) with gap detection.
- **LiveDataHandler** — bridges BarAggregator to the DataHandler ABC so strategies work unchanged in live mode.
- **BrokerAdapter** — abstract broker interface (paper, Alpaca, IBKR). PaperBroker simulates execution with slippage and commission.
- **OrderManager** — order lifecycle state machine (PENDING → SUBMITTED → FILLED/CANCELLED/REJECTED) with audit trail.
- **PaperTradingEngine** — async event loop for paper trading. Reuses Strategy, Portfolio, RiskManager from backtesting.
- **StateManager** — JSON snapshot persistence for crash recovery. Auto-prunes old snapshots.
- **HealthMonitor** — tracks bar freshness, event latency, fill rates. HEALTHY/DEGRADED/UNHEALTHY status with alert callbacks.

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
│   ├── backtest/           # Backtest engine and walk-forward analysis
│   ├── features/           # Feature engineering pipeline (10 generators)
│   ├── ml/                 # ML models (XGBoost, LightGBM) and MLStrategy
│   ├── validation/         # CPCV cross-validation and Deflated Sharpe Ratio
│   ├── regime/             # HMM regime detection (bull/bear/sideways)
│   ├── tracking/           # MLflow experiment tracking
│   ├── live/               # Live data feed, bar aggregation, LiveDataHandler
│   ├── broker/             # BrokerAdapter ABC, PaperBroker, OrderManager
│   ├── engine/             # PaperTradingEngine, StateManager
│   └── monitoring/         # HealthMonitor, HealthReport, HealthStatus
├── tests/                  # 1349 unit, integration, and stability tests
├── configs/                # Configuration files (10 configs)
├── scripts/                # Utility scripts (backtest, paper trading, reports)
├── data/                   # Market data (not in git)
├── notebooks/              # Jupyter notebooks
└── output/                 # Backtest results and reports
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
  name: "sma_crossover"     # sma_crossover or ml_strategy
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

**features** — Feature engineering (for ML strategy):
```yaml
features:
  technical:
    - type: "sma"
      windows: [5, 10, 20, 50]
    - type: "rsi"
      period: 14
    - type: "macd"
    - type: "bollinger"
    - type: "atr"
  statistical:
    - type: "returns"
      horizons: [1, 5, 10, 20]
    - type: "zscore"
    - type: "volatility"
      windows: [5, 20, 60]
  target:
    horizon: 5
    type: "direction"        # direction or return
```

**ml** — ML model configuration:
```yaml
ml:
  model: "xgboost"           # xgboost or lightgbm
  mode: "classification"     # classification or regression
  parameters:
    n_estimators: 200
    max_depth: 4
    learning_rate: 0.05
```

**regime** — Market regime detection (optional):
```yaml
regime:
  enabled: false
  n_regimes: 3
  vol_window: 20
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

**live** — Paper/live trading settings (optional, has defaults):
```yaml
live:
  data:
    feed_type: "websocket"       # websocket or polling
    bar_interval_seconds: 60     # bar duration in seconds
    max_history: 5000            # max bars to retain per symbol
    reconnect_attempts: 10
  broker:
    broker_type: "paper"         # paper, alpaca, or ibkr
    fill_delay_ms: 100           # simulated fill delay
  persistence:
    enabled: true
    state_dir: "state/"          # snapshot directory
    save_interval_seconds: 300   # auto-save every 5 minutes
    max_snapshots: 10            # keep last 10 snapshots
```

### Example Configs

| Config | Description |
|--------|-------------|
| `backtest_config.yaml` | Default single-asset SMA crossover |
| `backtest_baseline.yaml` | Baseline (fixed 10%, no risk) |
| `backtest_phase2_vol.yaml` | Volatility sizing + risk management |
| `backtest_phase2_meanvar.yaml` | Mean-variance optimized portfolio |
| `backtest_phase2_riskparity.yaml` | Risk parity allocation |
| `kelly_sizing.yaml` | Kelly criterion position sizing |
| `conservative_risk.yaml` | Tight risk limits example |
| `ml_backtest_config.yaml` | ML strategy (XGBoost classification) |
| `paper_trading_config.yaml` | Paper trading with SMA crossover |

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

## Running Strategies

```bash
make run              # Default SMA crossover
make run-ml           # ML strategy (XGBoost)
make run-baseline     # Baseline (no risk management)
make run-vol          # Volatility sizing
make run-meanvar      # Mean-variance optimization
make run-riskparity   # Risk parity optimization
make report           # Run ALL configs, generate comparison report + charts
make paper            # Run paper trading session (synthetic data)
make run-config CONFIG=path  # Run with a custom config
```

## Adding New Strategies

1. Create new file in `src/strategy/`
2. Inherit from `Strategy` base class
3. Implement `calculate_signals()` method

## Adding New ML Models

1. Create new file in `src/ml/`
2. Inherit from `MLModel` ABC
3. Implement `train()`, `predict()`, `save()`, `load()`, `get_feature_importance()`
4. Add to `build_ml_model()` in `scripts/run_backtest.py`
5. Add model name to `MLConfig.validate_model()` in `src/config.py`

## Adding New Brokers

1. Create new file in `src/broker/`
2. Inherit from `BrokerAdapter` ABC
3. Implement `connect()`, `disconnect()`, `submit_order()`, `cancel_order()`, `get_order_status()`, `get_positions()`, `get_account_info()`
4. Add broker type to `BrokerConfig.validate_broker_type()` in `src/config.py`

## AI assistance

Parts of this code and documentation were written with AI assistance.

## Project Status

Phase 4 complete — 1,249 tests, all checks green.

| Phase | Focus | Tests |
|-------|-------|-------|
| 1 | Core backtesting engine, events, portfolio | ~300 |
| 2 | Risk management, position sizing, optimization, walk-forward | ~600 |
| 3 | Feature engineering, ML models, CPCV, DSR, regime detection | ~997 |
| 4 | Paper trading, live data, broker adapter, monitoring | ~1349 |

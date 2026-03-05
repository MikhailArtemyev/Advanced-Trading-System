# How the Trading System Works

## What It Does

This is a backtester — it replays historical stock data and simulates what would happen if you followed a trading strategy. You give it price data and a strategy, it tells you how much money you would have made (or lost).

## The Event Loop

Everything runs on a simple loop that processes one price bar (one day) at a time:

```
For each day of historical data:
  1. MarketEvent  — "here's today's price data"
  2. SignalEvent  — strategy looks at prices, decides: buy, sell, or do nothing
  3. OrderEvent   — portfolio turns that decision into a concrete order (how many shares)
  4. FillEvent    — execution handler simulates filling the order (with slippage + commission)
```

Events go into a queue and get processed one by one before moving to the next day.

## Key Components

- **DataHandler** — loads CSV price data, feeds it bar-by-bar. Only gives the strategy data up to "today" (no peeking at the future).
- **Strategy** — decides when to buy/sell. Currently uses SMA crossover (buy when short moving average crosses above long, sell when it crosses below).
- **Portfolio** — tracks positions, cash, and P&L. Converts signals into sized orders.
- **ExecutionHandler** — simulates filling orders. Adds realistic slippage and commission costs.
- **PerformanceTracker** — calculates metrics (Sharpe ratio, max drawdown, win rate, etc.) and generates charts.
- **BacktestEngine** — wires everything together and runs the loop.

## How to Run

```bash
cd trading_system
make run
```

This downloads data (if needed) and runs the backtest using `configs/backtest_config.yaml`. Edit that file to change symbols, dates, or strategy parameters.

## Current Strategy: SMA Crossover

- Calculates a 20-day and 50-day simple moving average
- **Buy** when the 20-day crosses above the 50-day (short-term momentum is rising)
- **Sell** when the 20-day crosses below the 50-day (momentum is fading)

Simple, but it works as a proof of concept.

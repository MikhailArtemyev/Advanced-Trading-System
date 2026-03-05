# Phase 2, Week 3 — Enhanced Portfolio (Multi-Asset Support)

## What this is

Four additions that make the portfolio handle real-world multi-asset trading:

1. **Rolling Correlation Tracker** — tracks how pairs of assets move together
2. **Short selling** — the portfolio can now bet against stocks (sell first, buy later)
3. **Portfolio summary** — a snapshot of long/short exposure at any point in time
4. **Enhanced equity curve** — two new fields in the equity history

All four are backwards-compatible. Existing code works exactly the same unless you
opt in to the new features.

## What was built

### 1. Correlation Tracker (`src/portfolio/correlation.py`)

Tracks pairwise Pearson correlations using a rolling window of price returns.

```
Portfolio.update_timeindex()  -->  feeds latest prices to tracker
                              -->  tracker keeps a rolling window per symbol
                              -->  you can query any pair or get the full NxN matrix
```

Key design decisions:
- **No look-ahead bias** — only uses prices fed via `update()`, never peeks at future data
- **Returns-based** — correlations are on percentage returns, not raw prices (correct way)
- **Configurable** — `window` (default 60 bars) and `min_periods` (default 30) are tunable
- **Safe defaults** — returns `None` for insufficient data, `1.0` for same-symbol, `0.0` for constant prices

| Method | What it does |
|--------|-------------|
| `update(symbol, price)` | Feed a new price. History trimmed to `window` length |
| `get_correlation(a, b)` | Pairwise Pearson correlation between two symbols |
| `calculate_matrix()` | Full NxN correlation DataFrame |
| `get_matrix()` | Alias for `calculate_matrix()` |
| `symbols` (property) | List of currently tracked symbols |

The tracker plugs into Portfolio through a single constructor parameter:

```python
tracker = CorrelationTracker(window=60, min_periods=30)
portfolio = Portfolio(
    initial_capital=100000.0,
    symbols=["AAPL", "MSFT", "GOOGL"],
    correlation_tracker=tracker,
)
# Correlations update automatically on each portfolio.update_timeindex() call
```

### 2. Short selling (changes to `src/portfolio/portfolio.py`)

The portfolio now handles four fill paths instead of two:

| Fill Side | Position State | Action |
|-----------|---------------|--------|
| BUY | flat or long | Add to long (existing, unchanged) |
| BUY | short | Cover short — P&L = (avg_cost − fill_price) × qty − commission |
| SELL | long | Close/reduce long (existing, unchanged) |
| SELL | flat or short | Open/add to short — cash credited, weighted avg_cost updated |

Signal handling:

| Signal | Position | Result |
|--------|----------|--------|
| LONG | flat | BUY order (existing, unchanged) |
| SHORT | flat | SELL order, sized by position sizer |
| SHORT | has position | None (already in a trade) |
| EXIT | long | SELL full position (existing, unchanged) |
| EXIT | short | BUY to cover full position |
| EXIT | flat | None |

Unrealized P&L works correctly for shorts because the math is symmetric:
- Long 100 at $150, price $160: `unrealized = (160 − 150) × 100 = +1000`
- Short 100 at $150, price $140: `unrealized = (140 − 150) × (−100) = +1000`

The only code change needed in `update_timeindex()` was `quantity > 0` → `quantity != 0`.

### 3. Portfolio summary (`get_portfolio_summary()`)

New method returning a dict with aggregate stats:

```python
{
    "long_exposure": 16000.0,    # sum of qty × price for longs
    "short_exposure": 14500.0,   # sum of |qty| × price for shorts
    "net_exposure": 1500.0,      # long − short
    "gross_exposure": 30500.0,   # long + short
    "long_count": 1,
    "short_count": 1,
    "num_positions": 2,
    "cash": 100000.0,
    "equity": 101500.0,
}
```

### 4. Enhanced equity curve

Each `equity_history` entry now includes two extra fields:

| Field | What |
|-------|------|
| `positions_value` | equity − cash (total value of all open positions) |
| `num_positions` | count of positions where quantity ≠ 0 |

Old fields (`timestamp`, `equity`, `cash`) are unchanged. Existing code that reads
the equity curve keeps working — the extra keys are just ignored.

## How the pieces connect

```
BacktestEngine.run()
  |
  for each bar:
  |   emit MarketEvent
  |   process events:
  |     MARKET  -> Strategy.on_market_data() -> may emit Signal
  |     SIGNAL  -> Portfolio.on_signal()     -> generates BUY or SELL order
  |                                             (LONG → BUY, SHORT → SELL, EXIT → close)
  |     ORDER   -> RiskManager.check_order() -> approved? -> ExecutionHandler
  |     FILL    -> Portfolio.on_fill()       -> 4 paths: open/close long or short
  |   Portfolio.update_timeindex()
  |     -> update unrealized P&L (longs AND shorts)
  |     -> feed prices to CorrelationTracker (if attached)
  |     -> record equity snapshot (with positions_value + num_positions)
```

## Tests

**21 unit tests** in `tests/test_correlation.py`:
- Init with default/custom params, empty state
- Single/multiple price updates, window trimming
- Perfect positive/negative correlation, low correlation, insufficient data
- Same symbol returns 1.0, constant prices returns 0.0, unknown symbol returns None
- Matrix shape, symmetry, diagonal = 1.0, NaN for insufficient data
- No look-ahead bias verification

**21 unit tests** in `tests/test_portfolio_shorts.py`:
- Sell-to-open short, avg_cost, cash credited
- Buy-to-cover (partial and full), P&L calculation
- Add-to-short weighted avg_cost update
- Trade recording for shorts and covers
- SHORT signal generates SELL order, blocked when already in position
- EXIT signal generates BUY-to-cover, covers full position
- Equity with shorts (profit on price drop, loss on price rise)
- Unrealized P&L and market_value for shorts

**12 unit tests** in `tests/test_portfolio_summary.py`:
- Empty, long-only, short-only, mixed portfolios
- Net/gross exposure calculations
- Enhanced equity curve fields (positions_value, num_positions)
- DataFrame columns backwards compatibility

**4 integration tests** in `tests/test_portfolio_correlation_integration.py`:
- Portfolio works with and without tracker
- Matrix populates after enough bars
- Tracker only receives symbols that have prices

**Total: 350 tests passing, `make check` fully green.**

## Files changed

| File | Action | What |
|------|--------|------|
| `src/portfolio/correlation.py` | Created | CorrelationTracker class |
| `src/portfolio/__init__.py` | Modified | Added exports (CorrelationTracker, Portfolio, Position, Trade) |
| `src/portfolio/portfolio.py` | Modified | Short selling, correlation integration, summary, enhanced equity |
| `tests/test_correlation.py` | Created | 21 tests |
| `tests/test_portfolio_shorts.py` | Created | 21 tests |
| `tests/test_portfolio_summary.py` | Created | 12 tests |
| `tests/test_portfolio_correlation_integration.py` | Created | 4 tests |
| `tests/test_portfolio.py` | Modified | 1 test updated (SHORT signal now generates order) |

## Usage examples

```python
# Short selling — just use SignalType.SHORT in your strategy
signal = SignalEvent(
    timestamp=now,
    symbol="AAPL",
    signal_type=SignalType.SHORT,
    strength=1.0,
)
# Portfolio generates a SELL order, sized by position sizer

# Exit a short — same EXIT signal as for longs
exit_signal = SignalEvent(
    timestamp=now,
    symbol="AAPL",
    signal_type=SignalType.EXIT,
    strength=1.0,
)
# Portfolio detects the short and generates a BUY-to-cover order
```

```python
# Correlation tracking
from src.portfolio.correlation import CorrelationTracker

tracker = CorrelationTracker(window=60, min_periods=10)
portfolio = Portfolio(
    initial_capital=100000.0,
    symbols=["AAPL", "MSFT", "GOOGL"],
    correlation_tracker=tracker,
)

# After running some bars...
corr = tracker.get_correlation("AAPL", "MSFT")  # e.g. 0.72
matrix = tracker.calculate_matrix()               # full NxN DataFrame
```

```python
# Portfolio summary
summary = portfolio.get_portfolio_summary()
print(f"Net exposure: ${summary['net_exposure']:,.0f}")
print(f"Gross exposure: ${summary['gross_exposure']:,.0f}")
print(f"Long: {summary['long_count']} positions, Short: {summary['short_count']} positions")
```

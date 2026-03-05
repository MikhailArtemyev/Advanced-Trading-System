# Phase 2, Week 2 — Risk Manager

## What this is

A pre-trade risk manager that sits between the Portfolio and the ExecutionHandler.
Before any order gets filled, it goes through a series of safety checks. Think of
it like a compliance officer — it can approve, reduce, or reject an order.

**Before (Week 1):**
```
Portfolio creates order  -->  ExecutionHandler fills it immediately
```

**After (Week 2):**
```
Portfolio creates order  -->  RiskManager checks it  -->  approved?  -->  fill
                                                     -->  rejected?  -->  skip
                                                     -->  too big?   -->  reduce, then fill
```

## What was built

### `src/risk/risk_manager.py` — the core module

Three pieces:

1. **RiskLimits** — a frozen config object with 6 knobs you can tune:
   - `max_position_pct` (10%) — one stock can't be more than 10% of your portfolio
   - `max_portfolio_exposure_pct` (100%) — total exposure can't exceed your equity
   - `max_daily_loss_pct` (3%) — stop trading if you lose 3% in a single day
   - `max_drawdown_pct` (15%) — halt everything if portfolio drops 15% from peak
   - `max_open_positions` (20) — no more than 20 stocks at once
   - `max_orders_per_day` (100) — rate limit to prevent runaway strategies

2. **RiskCheckResult** — what you get back after a check:
   - Was it approved? If not, why?
   - Was the quantity reduced? If so, there's a warning explaining it.

3. **RiskManager** — runs 7 checks in priority order:

| # | Check | What it does | Hard reject? |
|---|-------|-------------|-------------|
| 1 | Circuit breaker | Halt all trading if drawdown exceeds limit | Yes |
| 2 | Daily loss limit | Stop trading for the day if losses are too high | Yes |
| 3 | Order rate limit | Reject if too many orders today | Yes |
| 4 | Open position count | Reject BUY if at max positions | Yes |
| 5 | Position concentration | Reduce BUY quantity if one stock is too large | Can reduce |
| 6 | Total exposure | Reduce BUY quantity if portfolio is too exposed | Can reduce |
| 7 | Correlation | Placeholder for future — does nothing yet | — |

Checks 1–4 are hard stops (reject the order entirely). Checks 5–6 are soft — they
shrink the order to fit within limits. Only approved orders count toward the daily
order counter (so rejected orders don't eat into your rate limit).

The risk manager also tracks:
- **Daily P&L** — resets when a new calendar day is detected
- **Peak equity** — highest portfolio value seen (for drawdown math)
- **Halt state** — once max drawdown is breached, all trading stops until manually reset

### Engine integration (`src/backtest/engine.py`)

The engine was modified to wire the risk manager into the event loop:

- **ORDER events**: Before sending to ExecutionHandler, the engine asks the risk
  manager to check the order. If rejected, it increments `rejected_orders` and skips.
  If the quantity was adjusted, it updates the order before passing it on.

- **FILL events (SELL side)**: After a sell fill, the engine looks at the latest
  trade's P&L and feeds it to the risk manager's daily P&L tracker.

- **End of each bar**: The engine updates the risk manager's peak equity so drawdown
  tracking stays current.

- **Results**: The final results dict now includes `rejected_orders` count.

Everything is backwards-compatible. If you don't pass a `risk_manager` to the engine,
it works exactly like before — no checks, no rejections.

## How SELL orders are handled

SELL orders skip checks 4, 5, and 6 (position count, concentration, exposure).
You always want to be able to exit a position — blocking a SELL would trap you in
a losing trade. Checks 1–3 (halt, daily loss, rate limit) still apply to SELLs
because those are about overall system health, not position sizing.

## How the pieces connect

```
BacktestEngine.run()
  |
  for each bar:
  |   emit MarketEvent
  |   process events:
  |     MARKET  -> Strategy.on_market_data() -> may emit Signal
  |     SIGNAL  -> Portfolio.on_signal()     -> may emit Order
  |     ORDER   -> RiskManager.check_order() -> approved? -> ExecutionHandler
  |                                          -> rejected? -> skip, count it
  |     FILL    -> Portfolio.on_fill()       -> update P&L tracker
  |   update portfolio valuation
  |   update peak equity in risk manager
```

## Tests

**46 unit tests** in `tests/test_risk_manager.py` covering:
- RiskLimits defaults and frozen immutability
- Each of the 7 checks individually
- Day rollover (daily P&L and order counter reset)
- Peak equity tracking
- Full reset
- Multiple checks interacting (priority ordering, combined adjustments)
- Edge cases (zero equity, zero price, sell bypasses)

**6 integration tests** added to `tests/test_backtest_engine.py`:
- Engine accepts risk_manager parameter
- Engine works without risk_manager (backwards compat)
- Results always contain `rejected_orders`
- Rate limit actually rejects orders during a backtest
- Reset clears rejected orders and risk manager state
- No-signal backtest with risk manager does nothing

**Total: 292 tests passing, `make check` fully green.**

## Files changed

| File | Action | What |
|------|--------|------|
| `src/risk/risk_manager.py` | Created | RiskLimits, RiskCheckResult, RiskManager |
| `src/risk/__init__.py` | Modified | Added 3 new exports |
| `src/backtest/engine.py` | Modified | Wired risk manager into event loop |
| `tests/test_risk_manager.py` | Created | 46 unit tests |
| `tests/test_backtest_engine.py` | Modified | 6 integration tests added |

## Usage example

```python
from src.risk.risk_manager import RiskLimits, RiskManager

# Use defaults
engine = BacktestEngine(..., risk_manager=RiskManager())

# Or customize limits
limits = RiskLimits(
    max_position_pct=0.05,       # 5% per position
    max_drawdown_pct=0.10,       # 10% max drawdown
    max_orders_per_day=50,       # 50 orders per day
)
engine = BacktestEngine(..., risk_manager=RiskManager(limits=limits))

# Results now include rejection count
results = engine.run()
print(f"Rejected orders: {results['rejected_orders']}")
```

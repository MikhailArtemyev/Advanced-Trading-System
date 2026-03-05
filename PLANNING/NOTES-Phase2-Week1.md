# Phase 2, Week 1: What We Did

## The Problem

In Phase 1, the Portfolio had a hardcoded rule for deciding how many shares to buy:

```
quantity = int(equity * 0.10 / price)
```

This means "always spend 10% of your money on each trade." That's fine for a prototype, but real trading systems need smarter ways to decide position sizes — sometimes you want to bet bigger, sometimes smaller, depending on the situation.

## What We Built

We created a **pluggable position sizing system**. Instead of one hardcoded formula, the Portfolio can now use any sizing algorithm. We built three:

### 1. FixedFractionSizer (the original)

Same as before — spend a fixed percentage of equity per trade. This is the default, so nothing breaks.

### 2. VolatilityBasedSizer (the smart one)

Uses ATR (Average True Range) to measure how much a stock's price jumps around day-to-day. The logic is simple:

- **Stock is volatile (big daily swings)?** Buy fewer shares — it's risky.
- **Stock is calm (small daily swings)?** Buy more shares — it's safer.

This way you risk roughly the same dollar amount per trade, regardless of how wild the stock is.

### 3. KellyCriterionSizer (the math-heavy one)

Uses the Kelly formula from gambling theory. It looks at your past trades and asks:

- What's your win rate? (How often do you make money?)
- When you win, how much do you win vs. when you lose?

Then it calculates the mathematically optimal bet size. We use "half Kelly" by default because full Kelly is too aggressive and can lead to huge drawdowns.

## How It Plugs In

The Portfolio constructor now takes an optional `position_sizer` parameter:

```python
# Default — same as Phase 1
portfolio = Portfolio(initial_capital=100000, symbols=["AAPL"])

# Custom — use volatility-based sizing
from src.risk.position_sizer import VolatilityBasedSizer
sizer = VolatilityBasedSizer(risk_fraction=0.02)
portfolio = Portfolio(initial_capital=100000, symbols=["AAPL"], position_sizer=sizer)
```

If you don't pass a sizer, it uses FixedFractionSizer with the same 10% rule as before. All existing code works exactly the same — zero breaking changes.

## What Changed Where

| File | What happened |
|------|--------------|
| `src/risk/__init__.py` | New package for risk management |
| `src/risk/position_sizer.py` | The three sizers + base class + result dataclass |
| `src/portfolio/portfolio.py` | Now delegates sizing to the sizer instead of hardcoding it |
| `tests/test_position_sizer.py` | 48 tests for all three sizers |
| `tests/test_portfolio.py` | 5 new integration tests for portfolio + sizer |

## Key Design Decisions

- **Sizers recommend, Portfolio constrains.** The sizer says "buy 200 shares," but the Portfolio still checks if you have enough cash. This keeps the cash logic in one place.
- **data_handler is optional.** Only VolatilityBasedSizer needs historical price data. The others work without it. If the volatility sizer can't get data, it falls back to a simple fixed fraction.
- **Trade history uses `list[Any]`.** KellyCriterionSizer needs past trades but importing the Trade class would create a circular dependency. Using `Any` keeps it clean.

## Verification

All 240 tests pass. `make check` is fully green (formatting, linting, type checking, tests).

# Phase 5 — Week 5: Strategy Templates

**Status:** COMPLETE
**Tests added:** 47 (13 momentum + 17 mean reversion + 17 pairs trading)
**Cumulative tests:** 1493

---

## What Was Built

### `src/strategy/momentum.py` — MomentumStrategy

Cross-sectional momentum: ranks symbols by trailing return, goes long top N, short bottom N.

**Parameters:**

| Param | Default | Description |
|-------|---------|-------------|
| `lookback_period` | 20 | Return lookback window |
| `top_n` | 3 | Number of long positions |
| `bottom_n` | 0 | Number of short positions (0 = long-only) |
| `rebalance_frequency` | 5 | Bars between rebalances |
| `min_bars` | 30 | Minimum bars before trading |

**Signal logic:**
- Computes trailing return for each symbol over `lookback_period`
- Ranks symbols by return
- Emits LONG for top N, SHORT for bottom N
- EXIT for symbols with positions not in top/bottom
- Skips symbols with insufficient data or existing matching positions

### `src/strategy/mean_reversion.py` — MeanReversionStrategy

Z-score based mean reversion with holding period limits.

**Parameters:**

| Param | Default | Description |
|-------|---------|-------------|
| `lookback_period` | 20 | Z-score window |
| `entry_threshold` | 2.0 | Z-score to enter |
| `exit_threshold` | 0.5 | Z-score to exit |
| `max_holding_period` | 10 | Force exit after N bars (0 = disabled) |

**Signal logic:**
- Z-score = (price - mean) / std over lookback window
- LONG when z <= -entry_threshold (oversold)
- SHORT when z >= +entry_threshold (overbought)
- EXIT when |z| <= exit_threshold (reverted) or max holding exceeded
- Handles zero std gracefully (no signal)

### `src/strategy/pairs_trading.py` — PairsTradingStrategy

Statistical pairs trading with OLS hedge ratio.

**Parameters:**

| Param | Default | Description |
|-------|---------|-------------|
| `lookback_period` | 60 | Spread calculation window |
| `entry_threshold` | 2.0 | Spread z-score to enter |
| `exit_threshold` | 0.5 | Spread z-score to exit |
| `min_bars` | 60 | Minimum bars before trading |

**Signal logic:**
- Requires exactly 2 symbols (A, B)
- Hedge ratio via OLS: regress A on B
- Spread = A - hedge_ratio * B
- Z-score of spread → entry/exit
- Spread high: short A, long B. Spread low: long A, short B
- EXIT both legs when spread reverts

### Updated Files

- `src/strategy/__init__.py` — Added exports for all three strategies
- `scripts/run_backtest.py` — `build_strategy()` handles `"momentum"`, `"mean_reversion"`, `"pairs_trading"`

---

## Tests

| File | Count | Coverage |
|------|-------|---------|
| `test_momentum_strategy.py` | 13 | Init validation, top/bottom ranking, exit signals, no duplicates, long-only mode |
| `test_mean_reversion.py` | 17 | Init, z-score entry/exit both directions, max holding period, zero std edge case |
| `test_pairs_trading.py` | 17 | Init, hedge ratio (perfect/zero-var/negative), spread signals, exit on reversion |

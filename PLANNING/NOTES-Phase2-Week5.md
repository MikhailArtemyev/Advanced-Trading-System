# Phase 2, Week 5 — Portfolio Optimization

## What this is

Two algorithms that answer the question: "Given 5 stocks, how much of my money
should I put into each one?" Instead of splitting equally (20% each), these
optimizers look at historical returns and correlations to find smarter allocations.

**Before (Weeks 1-4):**
```
Portfolio uses position sizer to decide how many shares per trade
But if you're trading 5 stocks, each one gets sized independently
No coordination — you might end up overweight in correlated stocks
```

**After (Week 5):**
```
Optimizer looks at all assets together and produces target weights
Mean-Variance: maximize return per unit of risk (classic Markowitz)
Risk Parity: make each stock contribute equally to total portfolio risk
Both fall back to equal weights if there isn't enough data
```

## What was built

### 1. Base classes (`src/optimization/base_optimizer.py`)

**AllocationResult** — what every optimizer returns:
```python
{
    "weights": {"AAPL": 0.35, "MSFT": 0.40, "GOOGL": 0.25},  # sums to 1.0
    "method": "mean_variance_max_sharpe",
    "expected_return": 0.12,       # annualized
    "expected_volatility": 0.18,   # annualized
    "sharpe_ratio": 0.56,
    "notes": "",
}
```

**PortfolioOptimizer** — abstract base class with one method:
```python
def optimize(symbols, returns_df, current_weights) -> AllocationResult
```

All concrete optimizers implement this interface, so they're interchangeable.

### 2. Mean-Variance Optimizer (`src/optimization/mean_variance.py`)

Classical Markowitz optimization. You give it historical returns, it finds the
portfolio weights that either:
- **max_sharpe** (default) — best return per unit of risk
- **min_variance** — lowest possible volatility

How it works:
```
1. Take daily returns for each asset
2. Compute annualized mean returns (mu) and covariance matrix (cov)
3. Optionally apply Ledoit-Wolf shrinkage to the covariance matrix
4. Run scipy.optimize.minimize (SLSQP) to find optimal weights
5. Subject to: weights sum to 1.0, each weight between min_weight and max_weight
```

Configurable parameters:

| Parameter | Default | What |
|-----------|---------|------|
| `risk_free_rate` | 0.02 | For Sharpe ratio calculation |
| `target` | "max_sharpe" | "max_sharpe" or "min_variance" |
| `max_weight` | 0.30 | No single asset above 30% |
| `min_weight` | 0.0 | Allow zero allocation |
| `shrinkage` | True | Apply Ledoit-Wolf covariance shrinkage |

**Ledoit-Wolf shrinkage** — the sample covariance matrix is noisy when you don't
have a ton of data. Shrinkage blends it toward a simpler target (scaled identity
matrix) to make the optimizer more stable. Without it, mean-variance tends to
produce extreme weights that don't hold up out-of-sample.

Safety fallbacks:
- Fewer than 30 observations → equal weight
- No return data for requested symbols → equal weight
- Single asset → 100% weight
- Optimizer fails to converge → equal weight

### 3. Risk Parity Optimizer (`src/optimization/risk_parity.py`)

Instead of optimizing for return, risk parity makes each asset contribute the
same amount of risk to the portfolio. The intuition:

- A stock that swings wildly (high volatility) gets a smaller weight
- A calm stock gets a bigger weight
- Result: no single stock dominates your portfolio's risk

This is more robust than mean-variance in practice because:
- It doesn't need return forecasts (only covariance)
- It's naturally diversified
- Less sensitive to estimation error

How it works:
```
1. Compute annualized covariance matrix
2. For candidate weights w, calculate each asset's risk contribution:
   risk_contribution_i = w_i * (cov @ w)_i / portfolio_vol
3. Minimize sum of squared deviations from equal contribution (vol/n each)
4. Subject to: weights sum to 1.0, each weight between min_weight and max_weight
```

Key behavior:
- Assets with identical volatility → roughly equal weights
- High-vol asset paired with low-vol asset → high-vol gets less weight

| Parameter | Default | What |
|-----------|---------|------|
| `max_weight` | 0.40 | Upper bound per asset |
| `min_weight` | 0.01 | Lower bound per asset (no zeros) |

Same safety fallbacks as mean-variance (equal weight on insufficient data,
single asset, optimizer failure).

## How the pieces connect

```
These optimizers don't plug into the backtest engine yet — that's Week 6.
For now they're standalone: you feed them returns, they return weights.

Week 6 will add MultiAssetSMAStrategy which calls optimizer.optimize()
periodically to rebalance target weights across assets.

src/optimization/
├── __init__.py           # exports all three classes
├── base_optimizer.py     # AllocationResult + PortfolioOptimizer ABC
├── mean_variance.py      # MeanVarianceOptimizer + Ledoit-Wolf shrinkage
└── risk_parity.py        # RiskParityOptimizer
```

## Tests

**24 tests** in `tests/test_optimizer.py`:

*AllocationResult (3 tests):*
- Creation with explicit weights, defaults, weights sum to 1.0

*MeanVarianceOptimizer init (2 tests):*
- Default parameters, custom parameters

*MeanVarianceOptimizer optimize (7 tests):*
- Equal weight fallback on insufficient data
- Max-Sharpe produces non-zero return and positive volatility
- Min-variance produces positive volatility
- Weights sum to 1.0
- Weight bounds respected (min/max)
- Single asset gets 100% weight
- No return data for requested symbols → equal weight fallback

*Ledoit-Wolf shrinkage (2 tests):*
- Shrunk covariance differs from sample covariance
- Shrinkage still produces valid optimization result

*RiskParityOptimizer init (2 tests):*
- Default parameters, custom parameters

*RiskParityOptimizer optimize (6 tests):*
- Equal weight fallback on insufficient data
- Equal-volatility assets get roughly equal weights
- High-vol asset gets lower weight than low-vol asset
- Weights sum to 1.0
- Weight bounds respected
- Single asset gets 100% weight

*Optimizer comparison (2 tests):*
- Both optimizers produce valid weights on same data
- Mean-variance and risk parity produce different allocations

**Total: 412 tests passing, `make check` fully green.**

## Files changed

| File | Action | What |
|------|--------|------|
| `src/optimization/__init__.py` | Created | Package exports (AllocationResult, MeanVarianceOptimizer, RiskParityOptimizer) |
| `src/optimization/base_optimizer.py` | Created | AllocationResult dataclass + PortfolioOptimizer ABC |
| `src/optimization/mean_variance.py` | Created | MeanVarianceOptimizer + Ledoit-Wolf shrinkage function |
| `src/optimization/risk_parity.py` | Created | RiskParityOptimizer |
| `tests/test_optimizer.py` | Created | 24 tests |

## Usage examples

```python
from src.optimization import MeanVarianceOptimizer, RiskParityOptimizer

# Mean-Variance: maximize Sharpe ratio
mv = MeanVarianceOptimizer(target="max_sharpe", max_weight=0.40)
result = mv.optimize(
    symbols=["AAPL", "MSFT", "GOOGL"],
    returns_df=returns,       # DataFrame with daily returns
    current_weights={},
)
print(result.weights)         # e.g. {"AAPL": 0.35, "MSFT": 0.40, "GOOGL": 0.25}
print(result.sharpe_ratio)    # e.g. 0.56

# Mean-Variance: minimize volatility
mv_min = MeanVarianceOptimizer(target="min_variance")
result = mv_min.optimize(["AAPL", "MSFT", "GOOGL"], returns, {})

# Risk Parity: equal risk contribution
rp = RiskParityOptimizer(max_weight=0.50)
result = rp.optimize(["AAPL", "MSFT", "GOOGL"], returns, {})
# High-vol stocks get less weight, low-vol stocks get more
```

```python
# Disable shrinkage (not recommended, but useful for comparison)
mv_raw = MeanVarianceOptimizer(shrinkage=False)
result_raw = mv_raw.optimize(symbols, returns, {})

# Tight weight constraints
mv_tight = MeanVarianceOptimizer(max_weight=0.25, min_weight=0.10)
# Forces diversification: every asset between 10% and 25%
```

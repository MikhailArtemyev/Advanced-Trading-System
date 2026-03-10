# Phase 3, Week 6 — Deflated Sharpe Ratio & Strategy Comparison

## What this is

Two deliverables this week:

1. **Deflated Sharpe Ratio (DSR)** — statistical test for whether a strategy's
   Sharpe ratio is real or just the result of testing many strategies and picking
   the best one.

2. **Full strategy comparison report** — `make report` runs all 8 configurations,
   ranks them, applies DSR analysis, and outputs a text report + charts.

**Before (Week 5):**
```
CPCV validates individual models
No way to know if a strategy's Sharpe survives multiple testing
No automated comparison of all available configurations
```

**After (Week 6):**
```
DSR adjusts Sharpe for selection bias from N trials
PSR accounts for non-normal returns (skewness, kurtosis)
`make report` runs all configs and generates ranked comparison + DSR analysis
Output: strategy_report.txt + 3 PNG charts
```

## The multiple testing problem

If you test 8 strategies and pick the one with the highest Sharpe, you're not
measuring skill — you're measuring luck. Even with zero-skill random strategies,
the *expected maximum Sharpe* from 8 independent trials is ~1.46. So a Sharpe of
0.68 (the best in our system) is *worse than random* after adjusting for selection.

The DSR framework (Bailey & López de Prado, 2014) formalizes this:

```
Observed Sharpe: 0.68   (looks decent)
E[max SR] from 8 trials: 1.46   (expected under null hypothesis)
Deflated Sharpe: 0.68 - 1.46 = -0.78   (negative = no real edge)
p-value: 0.0000   (not significant)
```

This is the key finding from `make report`: **none of the SMA crossover
configurations have statistically significant Sharpe ratios**. The strategy
doesn't have genuine predictive skill — different sizing and risk configs just
shift returns around without adding alpha.

## What was built

### 1. Deflated Sharpe Ratio (`src/validation/deflated_sharpe.py`)

Three functions and one dataclass:

**`DSRResult`** — result container:
```python
@dataclass
class DSRResult:
    observed_sharpe: float       # raw Sharpe ratio
    expected_max_sharpe: float   # E[max(SR)] under null
    deflated_sharpe: float       # observed - expected_max
    p_value: float               # P(true SR > benchmark)
    is_significant: bool         # p_value > (1 - significance_level)
    n_trials: int                # number of strategies tested
```

**`expected_max_sharpe(n_trials, sharpe_std=1.0)`**

Expected value of the maximum Sharpe ratio when testing N independent strategies
with no true skill. Uses the Euler-Mascheroni approximation:

```
E[max(SR)] = sigma * [(1 - gamma) * Phi_inv(1 - 1/N)
                      + gamma * Phi_inv(1 - 1/(N*e))]
```

where gamma = 0.5772... (Euler-Mascheroni constant) and Phi_inv is the standard
normal quantile function.

Key properties:
- n_trials=1 → returns 0.0 (no selection bias)
- Increases with n_trials (more trials = higher expected max)
- Scales linearly with sharpe_std
- n_trials=10 → ~1.58 (just from luck!)

**`probabilistic_sharpe(observed_sr, benchmark_sr, n_observations, skewness=0.0, kurtosis=3.0)`**

Probability that the true Sharpe ratio exceeds a benchmark, accounting for
estimation error in the observed Sharpe. The standard error of the Sharpe ratio
incorporates non-normality:

```
SE(SR) = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1))
```

Key properties:
- SR = benchmark → returns 0.5 (50/50 chance)
- Negative skewness decreases PSR (left tail risk)
- High kurtosis decreases PSR (fat tails = more uncertainty)
- More observations → narrower SE → more decisive result

**`deflated_sharpe_ratio(observed_sr, n_trials, n_observations, ...)`**

Combines both: uses `expected_max_sharpe` as the benchmark for `probabilistic_sharpe`.
Returns `DSRResult` with significance flag at 5% level (configurable).

### 2. Strategy Comparison Report (`scripts/run_full_report.py`)

**`make report`** runs this script, which:

1. Runs all 8 configurations through the backtest engine
2. Computes metrics + return statistics (skewness, kurtosis) per config
3. Builds a text report with 5 sections:
   - **DATA** — symbols, date range, capital
   - **SIDE-BY-SIDE COMPARISON** — grouped table (5 configs per group)
   - **RANKINGS** — sorted by Sharpe, total return, max drawdown, Calmar, win rate
   - **DEFLATED SHARPE RATIO ANALYSIS** — DSR for each strategy
   - **PER-STRATEGY DETAIL** — full metrics per config
4. Generates 3 charts:
   - `output/equity_comparison.png` — equity curves overlaid
   - `output/drawdown_comparison.png` — drawdown curves overlaid
   - `output/sharpe_comparison.png` — horizontal bar chart

**Suppresses verbose output**: Engine fill/bar logs and data handler loading
messages are redirected to `io.StringIO()` during execution, so the report
output is clean.

### 3. Phase 2 Comparison Script (`scripts/run_comparison.py`)

Simpler predecessor: runs 4 Phase 2 configs (baseline, vol, mean-var, risk-parity)
side-by-side. Available as `make run-compare`.

### 4. Additional Makefile Targets

| Target | Command | What |
|--------|---------|------|
| `make run-baseline` | Baseline config (fixed 10%, no risk) |
| `make run-vol` | Volatility sizing + risk management |
| `make run-meanvar` | Mean-variance optimized |
| `make run-riskparity` | Risk parity optimized |
| `make run-compare` | 4-config Phase 2 comparison |
| `make report` | Full 8-config report with DSR |

## The 8 configurations compared

| # | Config | Sizing | Risk | Optimization |
|---|--------|--------|------|-------------|
| 1 | Baseline (Default) | fixed_fraction 10% | ON | none |
| 2 | Baseline (No Risk) | fixed_fraction 10% | OFF | none |
| 3 | Conservative Risk | fixed_fraction 5% | ON | none |
| 4 | Volatility Sizing | volatility 2% risk | ON | none |
| 5 | Kelly Sizing | kelly half-Kelly | ON | none |
| 6 | Vol + Risk Mgmt | volatility 2% risk | ON (12% DD) | none |
| 7 | Mean-Var Optimized | volatility | ON | max_sharpe |
| 8 | Risk Parity | volatility | ON | risk_parity |

## Key findings from the report

1. **No strategy is statistically significant** — all DSR p-values are 0.0000,
   meaning none of the observed Sharpes survive the multiple testing adjustment.

2. **Kelly sizing performs worst** — not a bug. Kelly correctly identifies that the
   SMA crossover has poor win rate (~35-40%) and sizes down to near-zero. It's
   the mathematically optimal response to a strategy without edge.

3. **Risk management reduces drawdown but also reduces return** — the tradeoff is
   visible in the Vol + Risk Mgmt config vs. Volatility Sizing alone.

4. **Portfolio optimization (mean-var, risk-parity) doesn't add alpha** — it changes
   allocation across symbols but can't fix a weak underlying signal.

5. **The SMA crossover is the bottleneck** — all Phase 2 infrastructure works correctly,
   but position sizing and risk management can't create edge from nothing. This is
   exactly what Phase 3's ML models are designed to address.

## Tests

**39 tests** in `tests/test_deflated_sharpe.py`:

| Class | Tests | What |
|-------|-------|------|
| TestDSRResult | 3 | Creation, is_significant true/false |
| TestExpectedMaxSharpe | 9 | Single trial, two trials, monotonic increase, 1000 trials, std scaling, known value, validation errors |
| TestProbabilisticSharpe | 10* | SR=benchmark→0.5, above/below benchmark, few observations, symmetry, skew effect, kurtosis effect, large N, range check |
| TestDeflatedSharpeRatio | 12* | Single trial, many trials, strong Sharpe, significance flag, custom level, fields populated, n_trials stored, zero Sharpe, deflated=difference, skew/kurtosis effects |
| TestInputValidation | 5 | n_trials zero/negative, n_observations zero, significance level 0/1 |
| TestIntegration | 2 | DSR from tracker metrics, DSR with CPCV n_paths |

*PSR tests use SR=0.3 and T=50 (not SR=1.0 and T=252) to avoid saturation where
PSR rounds to 1.0, making comparison assertions like `psr_skewed < psr_normal`
fail as `1.0 < 1.0`.

## Files

| File | Lines | Action |
|------|-------|--------|
| `src/validation/deflated_sharpe.py` | 219 | Created — DSR, PSR, E[max SR] |
| `src/validation/__init__.py` | 25 | Updated — added DSR exports |
| `tests/test_deflated_sharpe.py` | 481 | Created — 39 tests |
| `scripts/run_full_report.py` | 517 | Created — full comparison report |
| `scripts/run_comparison.py` | 226 | Created — Phase 2 comparison |
| `Makefile` | 107 | Updated — added run-*, report targets |

## Running total

**896 tests passing** (793 after Week 5 + 103 validation tests).

`make check` passes: black, ruff, mypy, all tests.

## Usage

```bash
# Full report with charts
make report

# Quick Phase 2 comparison
make run-compare

# Individual configs
make run-baseline
make run-vol
make run-meanvar
make run-riskparity
```

```python
# Use DSR directly
from src.validation import deflated_sharpe_ratio

result = deflated_sharpe_ratio(
    observed_sr=0.68,
    n_trials=8,
    n_observations=1200,
    skewness=-0.5,
    kurtosis=4.2,
)
print(f"Significant: {result.is_significant}")  # False
print(f"p-value: {result.p_value:.4f}")          # 0.0000
```

## References

Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
*Journal of Portfolio Management*, 40(5), 94-107.

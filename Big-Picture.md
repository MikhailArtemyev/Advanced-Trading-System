# Architecting a Production-Grade Automated Trading System
## Software Architecture, ML Integration, and Development Roadmap for Short-Term Systematic Trading

**Version 1.0 | February 2026**

---

## Executive Summary

Short-term systematic trading systems demand a careful balance between flexibility and performance, ML sophistication and robustness, and modularity and operational simplicity. For a solo developer or small team targeting holding periods from minutes to days across crypto, forex, equities, and CFDs, the optimal architecture is an **event-driven modular monolith** that can evolve into selective microservices as scale demands.

The ML stack should combine **gradient-boosted trees** for interpretable signals with **Temporal Fusion Transformers** for multi-horizon forecasting, while **reinforcement learning (PPO)** handles portfolio optimization. Perhaps most critically, the system must embed rigorous overfitting prevention from day one using **combinatorial purged cross-validation**, **deflated Sharpe ratios**, and **walk-forward testing**—because a profitable backtest means nothing without proper validation.

This report provides a comprehensive technical blueprint covering:
- Software architecture and system design
- ML model selection for each trading component
- Validation methodologies and overfitting prevention
- Data infrastructure and storage design
- Execution systems and order management
- Risk management mechanisms
- A phased 18-month development roadmap

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [ML Model Architectures by Use Case](#2-ml-model-architectures-by-use-case)
3. [ML vs Rule-Based Decision Framework](#3-ml-vs-rule-based-decision-framework)
4. [Overfitting Prevention and Validation](#4-overfitting-prevention-and-validation)
5. [Backtesting Architecture](#5-backtesting-architecture)
6. [Data Infrastructure and Storage](#6-data-infrastructure-and-storage)
7. [Execution Layer Architecture](#7-execution-layer-architecture)
8. [Risk Management Mechanisms](#8-risk-management-mechanisms)
9. [Utility Modules Specification](#9-utility-modules-specification)
10. [Development Roadmap](#10-development-roadmap)
11. [Technology Stack Summary](#11-technology-stack-summary)
12. [Risks and Mitigation Strategies](#12-risks-and-mitigation-strategies)

---

## 1. System Architecture

### 1.1 Event-Driven Architecture Foundation

The consensus across production trading systems strongly favors **event-driven architecture (EDA)** as the foundational pattern. This approach provides:

- Same codebase for both backtesting and live trading with minimal component swapping
- Natural prevention of look-ahead bias by treating market data receipt as events
- Realistic order execution and transaction cost modeling
- Clear activity tracking and standardized component interaction

#### Core Event Types and Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          EVENT QUEUE (FIFO)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ MarketEvent │ -> │ SignalEvent │ -> │ OrderEvent  │ -> │  FillEvent  │
│  (triggers  │    │ (portfolio  │    │ (execution  │    │  (updates   │
│  strategy)  │    │  generates) │    │  handler)   │    │  portfolio) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 1.2 Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  Market Data    │  │ Alternative Data│  │ Reference Data  │         │
│  │   Handlers      │  │   Handlers      │  │   Handlers      │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
└───────────┼─────────────────────┼─────────────────────┼─────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING PIPELINE                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   Technical     │  │   Statistical   │  │   ML Feature    │         │
│  │   Indicators    │  │    Features     │  │   Extraction    │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
└───────────┼─────────────────────┼─────────────────────┼─────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MODEL INFERENCE SERVICE                            │
│           (Integrated with MLflow for Model Registry)                   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STRATEGY / SIGNAL GENERATION                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   Alpha Model   │  │ Signal Combiner │  │ Signal Strength │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
└───────────┼─────────────────────┼─────────────────────┼─────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        RISK MANAGEMENT                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  Pre-trade      │  │ Position Limits │  │   Portfolio     │         │
│  │    Checks       │  │                 │  │   Exposure      │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
└───────────┼─────────────────────┼─────────────────────┼─────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PORTFOLIO MANAGEMENT                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │Position Tracker │  │  Order Sizing   │  │ P&L Calculation │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
└───────────┼─────────────────────┼─────────────────────┼─────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXECUTION ENGINE                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  Order Manager  │  │ Smart Order     │  │ Broker Adapters │         │
│  │                 │  │   Router        │  │                 │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Microservices vs Monolith Tradeoffs

For initial development, a **modular monolith** is strongly recommended:

| Aspect | Modular Monolith | Microservices |
|--------|------------------|---------------|
| Initial Development | ✅ Faster | ❌ Slower |
| Operational Complexity | ✅ Lower | ❌ Higher |
| Hot-path Latency | ✅ No network overhead | ❌ Network calls |
| Debugging | ✅ Simpler | ❌ Distributed tracing needed |
| Scaling | ⚠️ Limited | ✅ Independent scaling |

#### Recommended Evolution Path

| Phase | Architecture | Components |
|-------|-------------|------------|
| Phase 1 | Monolithic | Single deployable with clean interfaces |
| Phase 2 | Hybrid | Extract data ingestion, backtesting compute |
| Phase 3 | Selective Services | Keep signal→execution together; separate ML training |

### 1.4 Practical Small-Team Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Docker Compose Environment                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ PostgreSQL/  │  │    Redis     │  │   MLflow     │  │  Airflow   │  │
│  │ TimescaleDB  │  │              │  │              │  │            │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                     Single Python Application                           │
│                   (with Rust Extensions via PyO3)                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Data Handler │ Strategy │ Portfolio │ Risk │ Execution        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ML Model Architectures by Use Case

### 2.1 Model Selection Matrix

| Use Case | Primary Recommendation | Alternative | When to Use Alternative |
|----------|----------------------|-------------|------------------------|
| Regime Detection | HMM + Ensemble Voting | Supervised Classifiers | When labeled regime data available |
| Return Prediction | Temporal Fusion Transformer | XGBoost/LightGBM | High-frequency, structured features |
| Volatility Prediction | GARCH-LSTM Hybrid | HAR-RV + TFT | When interpretability critical |
| Correlation/Stat Arb | TFT-GNN Hybrid | Cointegration + LSTM | Smaller asset universe |
| Portfolio Optimization | PPO (Reinforcement Learning) | Risk Parity + Kelly | Stable markets, simpler strategies |

### 2.2 Regime Detection

**Primary Recommendation: Hidden Markov Models (HMMs)**

**Rationale:**
- Strong theoretical foundation for modeling latent market states
- Natural capture of regime persistence through transition probabilities
- Interpretable output (state probabilities)

**Implementation Specifications:**
- States: 2-3 hidden states (low/high volatility or bull/bear/sideways)
- Training window: 4-year rolling for adaptability
- Features: Daily returns, realized volatility, spread metrics, volume, market breadth

**Enhanced Approach: Ensemble HMM + XGBoost**

Recent research (2024-2025) shows ensemble voting classifiers achieve superior detection:
- HMM provides underlying regime probability
- Tree-based models handle regime-specific predictions
- Voting mechanism combines for final classification

```python
# Conceptual architecture
class RegimeDetector:
    def __init__(self):
        self.hmm = GaussianHMM(n_components=3)
        self.xgb_classifier = XGBClassifier()
        
    def predict_regime(self, features):
        hmm_probs = self.hmm.predict_proba(features)
        xgb_probs = self.xgb_classifier.predict_proba(features)
        return ensemble_vote(hmm_probs, xgb_probs)
```

### 2.3 Return Prediction

**Primary Recommendation: Temporal Fusion Transformer (TFT)**

**Advantages:**
- Handles mixed-frequency data (static + time-varying features)
- Provides quantile predictions for uncertainty estimation
- Variable selection networks automatically identify important features
- SMAPE of 0.0022 on stock prediction tasks (outperforming LSTM, SVR, vanilla Transformers)

**Alternative: XGBoost/LightGBM**

| Model | Best For | Performance |
|-------|----------|-------------|
| LightGBM | High-frequency, structured features | Monthly OOS R² = 2.13%, Sharpe = 1.77 |
| XGBoost | When SHAP interpretability needed | Similar performance, slower training |
| TFT | Multi-horizon forecasting | Superior for 1-5 day horizons |

**Key Research Finding:**
> "Transformers excel at absolute price prediction, LSTMs are superior for predicting price differences."

During COVID-19 crisis: Deep learning models showed only 45% performance degradation vs 100%+ for traditional models.

### 2.4 Volatility Prediction

**Primary Recommendation: GARCH-LSTM Hybrid**

**Architecture:**
```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Raw Returns  │ -> │  GARCH Model  │ -> │     LSTM      │
│               │    │ (preliminary  │    │  (residual    │
│               │    │  volatility)  │    │   patterns)   │
└───────────────┘    └───────────────┘    └───────────────┘
```

**Performance:** 30-50% RMSE improvement over standalone approaches

**Why Hybrid Works:**
- Infuses volatility "stylized facts" (clustering, asymmetric effects, long memory)
- Neural network learns residual patterns GARCH misses
- Maintains interpretability through GARCH component

**Baseline Model:** HAR-RV (Heterogeneous Autoregressive Realized Volatility)
- Strong baseline for realized volatility
- TFT shows excellent results when combined with sectoral pooling

### 2.5 Correlation Modeling / Statistical Arbitrage

**Primary Recommendation: TFT-GNN Hybrid**

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Asset Universe (N assets)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Graph Neural Network                                           │
│  - Nodes: Individual assets                                     │
│  - Edges: Correlation/cointegration relationships               │
│  - Output: Relational features embedded as time-varying inputs  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Temporal Fusion Transformer                                    │
│  - Input: Price data + GNN relational features                  │
│  - Output: Spread predictions, trading signals                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Finding:** GNN-derived relational features assigned greater weight than RSI and MACD in attention analysis, indicating relational structure forms essential part of data-generating process.

**Traditional Enhancement Pipeline:**
1. PCA dimensionality reduction
2. DBSCAN clustering by price features
3. Cointegration testing within clusters
4. LSTM spread prediction
5. RL-based trading rule optimization

### 2.6 Portfolio Weight Optimization

**Primary Recommendation: Proximal Policy Optimization (PPO)**

**Performance Comparison:**

| Method | Cumulative Returns | Sharpe Ratio |
|--------|-------------------|--------------|
| PPO (RL) | 37.5% | 2.15 ± 0.05 |
| Mean-Variance Optimization | 15.1% | 1.2-1.5 |
| Hierarchical Risk Parity | 10.0% | 1.0-1.3 |

**Implementation Framework:**

```python
# State Space
state = {
    'prices': np.array,        # Asset prices
    'indicators': np.array,    # Technical indicators
    'holdings': np.array,      # Current portfolio weights
}

# Action Space
action = np.array  # Continuous weight distribution (sum to 1)

# Reward Function
def reward(returns, holdings):
    sharpe = compute_sharpe(returns)
    drawdown_penalty = compute_drawdown_penalty(holdings)
    return sharpe - lambda * drawdown_penalty
```

**Architecture Notes:**
- CNN-based feature extractors significantly outperform MLP
- Longer lookback periods (28 vs 16 days) yield higher returns
- PPO's clipped objective provides sample efficiency and stable training

### 2.7 Model Comparison Summary

| Model Type | Pros | Cons | Best Use Case |
|------------|------|------|---------------|
| **LSTM** | Captures sequential dependencies; mature ecosystem | Vanishing gradients; slow training | Price difference prediction |
| **Transformer/TFT** | Parallel training; attention interpretability; multi-horizon | Data hungry; computationally expensive | Multi-day return forecasting |
| **XGBoost/LightGBM** | Fast training; SHAP interpretability; handles missing data | No temporal structure; feature engineering required | High-frequency signals |
| **GNN** | Captures asset relationships; dynamic topology | Complex implementation; graph construction sensitive | Correlation modeling |
| **HMM** | Interpretable states; regime persistence | Gaussian assumptions; limited state count | Regime detection |
| **PPO (RL)** | End-to-end optimization; handles transaction costs | Sample inefficient; reward design critical | Portfolio optimization |
| **Gaussian Processes** | Uncertainty quantification; works with small data | Scales poorly O(n³); kernel selection | Volatility with confidence |

---

## 3. ML vs Rule-Based Decision Framework

### 3.1 Recommended Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL GENERATION                               │
│                    (ML-Driven: Return prediction, regime detection)     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PORTFOLIO CONSTRUCTION                             │
│        (Classical: Risk parity, mean-variance with ML-predicted inputs) │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        RISK MANAGEMENT                                  │
│           (Rule-Based: Position limits, stop-losses, VaR limits)        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION                                     │
│         (Classical: TWAP, VWAP with ML-predicted market impact)         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component-Level Recommendations

| Component | Approach | Rationale |
|-----------|----------|-----------|
| Return Prediction | ML (TFT, XGBoost) | Complex patterns, non-linear relationships |
| Regime Detection | ML (HMM, Ensemble) | Latent state modeling |
| Volatility Forecasting | Hybrid (GARCH-LSTM) | Combine stylized facts with pattern learning |
| Correlation Estimation | ML (GNN) or Rolling Window | Dynamic relationships |
| Position Sizing | Classical (Kelly, Vol-based) | Strong theoretical foundation |
| Portfolio Weights | Classical with ML inputs | Robust optimization theory |
| Risk Limits | Rule-Based | Deterministic enforcement required |
| Stop-Loss/Take-Profit | Rule-Based | Immediate execution required |
| Execution Algorithms | Classical (TWAP/VWAP) | Well-understood, predictable |

### 3.3 When Classical Approaches Outperform ML

Classical methods are preferred when:
- Market regimes are stable and well-understood
- Sample sizes are small (< 1000 observations)
- Interpretability is critical for compliance
- Signal-to-noise ratio is very low
- Asset universe is large relative to training data

**Example:** Mean-variance optimization with shrinkage estimators often outperforms deep learning when N (assets) >> T (time periods).

---

## 4. Overfitting Prevention and Validation

### 4.1 Combinatorial Purged Cross-Validation (CPCV)

**Why Standard K-Fold Fails:**
- Financial labels are NOT IID
- Overlapping outcomes create information leakage
- Test performance is systematically overestimated

**CPCV Implementation:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Data Timeline (N Groups)                             │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐        │
│  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  9  │ 10  │        │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘        │
│                                                                         │
│  Combination 1:  [TRAIN] [TRAIN] [TRAIN] [TEST] [TEST] [TRAIN]...      │
│  Combination 2:  [TRAIN] [TEST] [TRAIN] [TRAIN] [TEST] [TRAIN]...      │
│  Combination K:  [TEST] [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]...      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  PURGING: Remove training samples overlapping with test     │       │
│  │  EMBARGO: Add buffer period after test sets                 │       │
│  └─────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Purging Mechanism:**
- If test sample Yⱼ depends on information Φⱼ
- Remove ALL training labels depending on Φⱼ
- Embargo period (1-5% of data) adds temporal buffer

**Minimum Requirements:**
- At least 100 combinations for stable distributions
- Hyperparameters: n_splits, train_size_pct, test_size_pct, purge_size

### 4.2 Deflated Sharpe Ratio (DSR)

**The Problem:** When testing N strategies, probability of false discovery approaches 1.

**DSR Formula:**

```
DSR = Φ[(SR̂ - SR₀) × √(T-1) / √(1 - γ₃×SR̂ + ((γ₄-1)/4)×SR̂²)]

Where:
- SR̂ = Observed Sharpe Ratio
- SR₀ = Expected maximum Sharpe under null (False Strategy Theorem)
- T = Number of observations
- γ₃ = Skewness
- γ₄ = Kurtosis
- Φ = Standard normal CDF
```

**Interpretation:**
| DSR Value | Interpretation |
|-----------|----------------|
| > 0.95 | Likely genuine skill |
| 0.50 - 0.95 | Uncertain |
| < 0.50 | Likely false positive |

**Minimum Track Record Length (MinTRL):**
- For Sharpe = 1.0 at 95% confidence: ~3 years of daily data
- Each additional trial increases requirement

### 4.3 Walk-Forward Validation

**Rolling vs Expanding Window:**

| Method | Training Window | Advantages | Disadvantages |
|--------|----------------|------------|---------------|
| Rolling | Fixed size (2-4 years) | Adapts to regime changes | May lose valuable early data |
| Expanding | Grows from fixed start | Uses all available data | May include outdated patterns |

**Walk-Forward Efficiency (WFE):**

```
WFE = (Annualized OOS Return / Annualized IS Return) × 100%

Passing Threshold: WFE > 50-60%
```

**Parameter Stability Analysis:**
- Coefficient of Variation (CV) across windows
- CV < 0.30 indicates robustness
- CV > 0.50 suggests overfitting

### 4.4 Monte Carlo Validation

**Trade Shuffling (Permutation) Tests:**
- Reshuffle trade sequences across 1,000+ simulations
- Tests path dependency
- Original strategy should outperform 95%+ of shuffled versions

**Parameter Perturbation (Jitter Testing):**
- Add ±10% noise to parameters
- Run 1,000+ simulations
- If Sharpe degrades > 50%, strategy is likely overfit

### 4.5 Validation Checklist

| Requirement | Threshold | Status |
|-------------|-----------|--------|
| Deflated Sharpe Ratio | > 0.95 | ☐ |
| Walk-Forward Efficiency | > 50% | ☐ |
| Monte Carlo 5th Percentile Sharpe | > 0 | ☐ |
| Maximum OOS Degradation | < 50% | ☐ |
| Parameter Stability CV | < 0.30 | ☐ |
| Track Record | ≥ MinTRL | ☐ |

---

## 5. Backtesting Architecture

### 5.1 Event-Driven Simulation Engine

**Main Loop Structure:**

```python
class Backtester:
    def run(self):
        while True:
            # Outer loop: Advance heartbeat (bar-by-bar)
            if self.data_handler.continue_backtest:
                self.data_handler.update_bars()
            else:
                break
            
            # Inner loop: Process all events before advancing
            while True:
                try:
                    event = self.events.get(block=False)
                except Empty:
                    break
                
                if event.type == 'MARKET':
                    self.strategy.calculate_signals(event)
                    self.portfolio.update_timeindex(event)
                elif event.type == 'SIGNAL':
                    self.portfolio.update_signal(event)
                elif event.type == 'ORDER':
                    self.execution_handler.execute_order(event)
                elif event.type == 'FILL':
                    self.portfolio.update_fill(event)
```

### 5.2 Look-Ahead Bias Prevention

**Structural Safeguards:**
1. Data handler only releases data at or before current simulation time
2. Strategies/portfolios cannot access future data by design
3. Point-in-time data with `as_of` timestamps mandatory

```python
class DataHandler:
    def get_latest_bars(self, symbol, N=1):
        """Only returns bars up to current simulation time"""
        return self.bars[symbol][:self.current_index][-N:]
    
    # NEVER expose:
    # def get_all_bars(self, symbol)  # Would allow look-ahead
```

### 5.3 Execution Modeling

**Market Order Fill Model:**

```
fill_price = mid_price + (spread/2) × sign(direction) + market_impact

Market Impact (Almgren Square-Root Model):
Impact = σ × √(Q/V) × (T/τ)^(-0.5)

Where:
- σ = Volatility
- Q = Order quantity
- V = Daily volume
- T = Execution time
- τ = Time constant
```

**Limit Order Fill Probability:**
- Distance from mid-price
- Current volatility
- Time horizon
- Queue position estimation

**Transaction Cost Components:**

| Component | Typical Range |
|-----------|---------------|
| Maker/Taker Fees | 0.1-0.2% |
| Half-Spread | Variable by asset |
| Funding Rates (Perpetuals) | ±0.01-0.03% per 8h |
| Borrow Costs (Shorts) | 0.5-5% annually |

### 5.4 Slippage Assumptions by Asset Class

| Asset Class | Expected Slippage (bps) | Conservative Assumption |
|-------------|------------------------|------------------------|
| Large-Cap Equities | 1-5 | 10 |
| Small-Cap Equities | 10-50 | 75 |
| Forex Majors | 0.5-2 | 5 |
| Forex Exotics | 5-20 | 40 |
| BTC/ETH | 3-10 | 20 |
| Crypto Altcoins | 20-100+ | 150 |

### 5.5 Walk-Forward Backtesting Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Full Historical Data                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Window 1: [=======TRAIN=======][==TEST==]                              │
│  Window 2:      [=======TRAIN=======][==TEST==]                         │
│  Window 3:           [=======TRAIN=======][==TEST==]                    │
│  Window 4:                [=======TRAIN=======][==TEST==]               │
│                                                                         │
│  Typical Ratio: 70-80% Train / 20-30% Test                              │
│  Training: 2-4 years | Testing: 3-6 months                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Aggregate OOS results across all windows for final metrics     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Infrastructure and Storage

### 6.1 Database Selection

**Time-Series Database Comparison:**

| Database | Ingestion Rate | Query Speed | Best For |
|----------|---------------|-------------|----------|
| QuestDB | 7-11M rows/sec | 16-20x faster aggregation | Primary OHLCV/tick storage |
| TimescaleDB | 600K-1.2M rows/sec | Full SQL compatibility | Teams with PostgreSQL expertise |
| ClickHouse | High | 100x faster OLAP | Batch analytics, historical analysis |
| ArcticDB | High | 25x faster than legacy | Quantitative research, petabyte scale |

**Recommendation: QuestDB**
- Native `ASOF JOIN` for point-in-time joins
- `SAMPLE BY` for efficient time bucketing
- 6-13x faster ingestion than TimescaleDB

### 6.2 Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA STORAGE LAYERS                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  L1: In-Memory Cache (Process-Level)                            │   │
│  │  Access: ~100μs | Contents: Latest quotes per symbol            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  L2: Redis Cache                                                │   │
│  │  Access: ~1ms | Contents: Recent OHLCV, positions, features     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  L3: Time-Series Database (QuestDB)                             │   │
│  │  Access: ~10-100ms | Contents: Full historical data             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PostgreSQL: Metadata, configs, trade logs, audit trail         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Feature Store Architecture

**Recommended: Feast (Open Source)**

**Capabilities:**
- Point-in-time correct joins for training data
- Offline/online store separation (Parquet → Redis)
- OpenLineage integration for data lineage
- Pip-installable, no infrastructure required to start

**Custom Feature Store Design:**

```
Raw Data (QuestDB/TSDB)
        │
        ▼
Feature Pipeline (Airflow/dbt)
        │
        ├──► Offline Store (Parquet/S3) ──► Training
        │
        └──► Online Store (Redis) ──► Real-time Serving
```

### 6.4 Redis Key Naming Convention

```
{namespace}:{service}:{entity}:{id}:{params}

Examples:
- market:bar:ohlcv:AAPL:1m:current
- feature:technical:rsi:BTCUSD:14d
- position:portfolio:main:AAPL:quantity
```

**Cache TTL Guidelines:**

| Data Type | TTL | Jitter |
|-----------|-----|--------|
| Real-time Quotes | 100-500ms | ±10% |
| 1-Minute Bars | 60s | ±10% |
| Daily Data | 24 hours | ±10% |
| Features | Match update frequency | ±10% |

### 6.5 Data Versioning with DVC + MLflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DATA VERSIONING WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │  Raw Data   │    │   DVC       │    │   Remote    │                 │
│  │   Files     │ -> │  .dvc files │ -> │  Storage    │                 │
│  │             │    │  (in Git)   │    │  (S3/GCS)   │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MLflow Experiment Tracking                                     │   │
│  │  - Log DVC hash as MLflow tag                                   │   │
│  │  - Full data-to-model traceability                              │   │
│  │  - Parameters, metrics, model registry                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Execution Layer Architecture

### 7.1 Order Management System (OMS)

**Order State Machine:**

```
┌─────────┐    ┌─────────────────┐    ┌───────────┐    ┌──────────────┐
│   NEW   │ -> │ PENDING_SUBMIT  │ -> │ SUBMITTED │ -> │ ACKNOWLEDGED │
└─────────┘    └─────────────────┘    └───────────┘    └──────────────┘
                                                              │
                    ┌─────────────────────────────────────────┤
                    │                                         │
                    ▼                                         ▼
           ┌────────────────────┐                    ┌──────────────┐
           │ PARTIALLY_FILLED   │                    │    FILLED    │
           └────────────────────┘                    └──────────────┘
                    │
    ┌───────────────┼───────────────┬───────────────┐
    │               │               │               │
    ▼               ▼               ▼               ▼
┌────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│ FILLED │    │ CANCELED │    │ REJECTED │    │ EXPIRED │
└────────┘    └──────────┘    └──────────┘    └─────────┘
```

**All state transitions must be logged with timestamps for audit compliance.**

### 7.2 Broker API Abstraction

```python
from abc import ABC, abstractmethod

class BrokerAdapter(ABC):
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker"""
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> OrderId:
        """Submit order, return broker order ID"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: OrderId) -> bool:
        """Cancel pending order"""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[Symbol, Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> AccountBalance:
        """Get account balance and margin info"""
        pass

# Implementations
class InteractiveBrokersAdapter(BrokerAdapter): ...
class AlpacaAdapter(BrokerAdapter): ...
class BinanceAdapter(BrokerAdapter): ...
```

### 7.3 Connection Resilience

| Pattern | Implementation |
|---------|----------------|
| Reconnection | Exponential backoff with jitter |
| Circuit Breaker | Disable after N consecutive failures |
| Cancel-on-Disconnect | Broker-side COD functionality |
| Heartbeat | Regular ping with auto-reconnect |

### 7.4 Position Reconciliation

**Three-Way Matching (Daily):**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Internal Book  │    │ Fund Administrator│   │ Broker/Custodian│
│     (IBOR)      │ <->│                 │ <->│    (Street)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Break Resolution   │
                    │  - Quantity mismatches
                    │  - Price differences
                    │  - Fee discrepancies
                    └─────────────────────┘
```

---

## 8. Risk Management Mechanisms

### 8.1 Real-Time Risk Engine

**VaR Calculation Methods:**

| Method | Formula | Use Case |
|--------|---------|----------|
| Parametric | VaR = Portfolio × z × σ × √horizon | Speed (real-time) |
| Historical | Percentile of historical P&L | Non-normal distributions |
| Monte Carlo | 10,000+ simulated paths | Complex portfolios |

### 8.2 Position Limits Framework

| Limit Type | Typical Threshold | Rationale |
|------------|------------------|-----------|
| Per-Symbol | 5-10% of portfolio | Single-name concentration |
| Per-Sector | 25-30% of portfolio | Sector concentration |
| Per-Asset-Class | 50% of portfolio | Asset class concentration |
| Maximum Leverage | 1-2x | Tail risk management |
| Correlation Limit | No new positions with r > 0.7 | Diversification |

### 8.3 Kill-Switch Implementation

**Trigger Conditions:**

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Daily Loss | 3-5% | Halt new trades |
| Maximum Drawdown | 10-20% | Close all positions |
| Order Rate Anomaly | >50-100 orders/min | Emergency stop |
| Volatility Spike | >3x normal | Reduce positions |
| Strategy Degradation | Rolling Sharpe < -1.0 | Disable strategy |

**Shutdown Procedures:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GRACEFUL SHUTDOWN (5-15 minutes)                     │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Cancel all pending orders                                           │
│  2. Close positions using limit orders (reasonable slippage)            │
│  3. Allow time for position unwinding                                   │
│  4. Maintain risk management during unwind                              │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ If losses accelerating
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EMERGENCY SHUTDOWN (Immediate)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Market order liquidation of all positions                           │
│  2. Cancel all pending orders                                           │
│  3. Disconnect from broker                                              │
│  4. Alert operations team                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.4 Position Sizing Algorithms

**Fractional Kelly:**

```
Full Kelly: f* = (p × b - q) / b

Where:
- p = Win probability
- q = Loss probability (1 - p)
- b = Win/loss ratio

Recommended: Use 0.25-0.5 × Full Kelly

Rationale: Full Kelly can see 50-70% drawdowns
           Quarter Kelly provides better risk-adjusted returns
```

**Volatility-Based Sizing:**

```
Position Size = (Capital × Risk%) / (ATR × Multiplier)

Example:
- Capital: $100,000
- Risk per trade: 1%
- ATR: $2.50
- Multiplier: 2

Position Size = ($100,000 × 0.01) / ($2.50 × 2) = 200 shares
```

---

## 9. Utility Modules Specification

### 9.1 Must-Have Modules

| Module | Priority | Description |
|--------|----------|-------------|
| Risk Engine | P0 | VaR, drawdown, position limits |
| Performance Analytics | P0 | Sharpe, Sortino, Calmar, max DD |
| Order Manager | P0 | Order lifecycle, state machine |
| Position Tracker | P0 | Real-time P&L, exposure |
| Kill Switch | P0 | Automated emergency stop |
| Logging/Audit | P0 | Immutable trade history |
| Alert Service | P1 | PagerDuty/Slack integration |
| Dashboard | P1 | Grafana visualization |
| Anomaly Detection | P1 | Strategy degradation detection |
| Reconciliation | P1 | Position verification |
| Report Generator | P2 | Daily/weekly performance reports |
| Config Manager | P2 | Strategy parameter management |

### 9.2 Performance Metrics Module

```python
class PerformanceMetrics:
    def sharpe_ratio(self, returns, risk_free=0.0, periods=252):
        """Annualized Sharpe Ratio"""
        excess = returns - risk_free / periods
        return np.sqrt(periods) * excess.mean() / excess.std()
    
    def sortino_ratio(self, returns, risk_free=0.0, periods=252):
        """Sortino Ratio (downside deviation)"""
        excess = returns - risk_free / periods
        downside = returns[returns < 0].std()
        return np.sqrt(periods) * excess.mean() / downside
    
    def calmar_ratio(self, returns, periods=252):
        """Calmar Ratio (return / max drawdown)"""
        annual_return = returns.mean() * periods
        max_dd = self.max_drawdown(returns)
        return annual_return / abs(max_dd)
    
    def max_drawdown(self, returns):
        """Maximum Drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def win_rate(self, trades):
        """Percentage of winning trades"""
        winners = sum(1 for t in trades if t.pnl > 0)
        return winners / len(trades) if trades else 0
    
    def profit_factor(self, trades):
        """Gross profit / Gross loss"""
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
```

### 9.3 Anomaly Detection for Strategy Degradation

**Detection Methods:**

| Method | Implementation | Trigger |
|--------|----------------|---------|
| Rolling Sharpe | 30-day rolling window | Sharpe < -1.0 |
| Drawdown Velocity | Rate of drawdown increase | Accelerating losses |
| Fill Rate Change | Execution quality degradation | >20% change |
| Alpha Decay | Predictive power decrease | IC < 0.02 |

---

## 10. Development Roadmap

### 10.1 Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    18-MONTH DEVELOPMENT TIMELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: MVP Backtester          [Months 1-2]   ████░░░░░░░░░░░░░░     │
│  Phase 2: Portfolio Engine        [Months 3-4]   ░░░░████░░░░░░░░░░     │
│  Phase 3: ML Integration          [Months 5-7]   ░░░░░░░░██████░░░░     │
│  Phase 4: Paper Trading           [Months 8-9]   ░░░░░░░░░░░░░░████     │
│  Phase 5: Live Trading            [Months 10-11] ░░░░░░░░░░░░░░░░████   │
│  Phase 6: Monitoring/Risk         [Months 12-13] ░░░░░░░░░░░░░░░░░░██   │
│  Phase 7: Multi-Strategy Scale    [Months 14-18] ░░░░░░░░░░░░░░░░░░████ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Phase 1: MVP Backtester (Months 1-2)

**Objectives:**
- Build minimal viable event-driven backtester
- Single strategy capability
- Basic performance reporting

**Key Modules:**

| Module | Description |
|--------|-------------|
| DataHandler | CSV/API data loading, bar iteration |
| EventQueue | FIFO queue with event types |
| Strategy (base) | Signal generation interface |
| Portfolio | Position tracking, basic P&L |
| ExecutionHandler | Simulated fills with fixed slippage |
| PerformanceReporter | Sharpe, returns, drawdown |

**Tech Stack:**
- Python 3.11+
- pandas, NumPy, matplotlib
- SQLite for initial storage
- YAML for configuration

**Potential Pitfalls:**

| Risk | Mitigation |
|------|------------|
| Look-ahead bias | Use strict `get_latest_bars` pattern |
| Over-engineering | Start simple with Python Queue |
| Ignoring costs | Include basic spread/commission from day one |

**Validation Tests:**
- Unit tests for each component
- Integration test with SMA crossover strategy
- Manual verification of P&L calculation

**Success Metrics:**
- ✅ Backtest runs without errors
- ✅ P&L matches manual calculation
- ✅ Events process in correct order

---

### 10.3 Phase 2: Portfolio Engine (Months 3-4)

**Objectives:**
- Multi-asset support
- Proper position sizing
- Basic risk management

**Key Modules:**

| Module | Description |
|--------|-------------|
| PositionSizer | Kelly criterion, volatility-based sizing |
| RiskManager | Pre-trade checks, position limits |
| PortfolioOptimizer | Basic mean-variance, risk parity |
| Portfolio (enhanced) | Multi-asset tracking, correlation |

**Tech Stack:**
- Add scipy for optimization
- Add PostgreSQL for trade logging
- Configuration schema validation

**Potential Pitfalls:**

| Risk | Mitigation |
|------|------------|
| Future correlation data | Rolling window correlations only |
| Constraint violations | Hard position limits |
| Complexity explosion | Keep optimizer simple initially |

**Validation Tests:**
- Portfolio rebalancing produces expected weights
- Risk limits prevent oversized positions
- Multi-asset P&L tracks correctly

**Success Metrics:**
- ✅ Backtest 10+ asset portfolio
- ✅ Position sizes respect limits
- ✅ Portfolio metrics calculate correctly

---

### 10.4 Phase 3: ML Model Integration (Months 5-7)

**Objectives:**
- ML-based signal generation
- Proper validation methodology
- Experiment tracking

**Key Modules:**

| Module | Description |
|--------|-------------|
| FeatureEngine | Technical indicators, statistical features |
| ModelTrainer | CPCV, walk-forward validation |
| InferenceService | Online prediction |
| RegimeDetector | HMM-based regime classification |
| ReturnPredictor | XGBoost/LightGBM (initially) |
| MLValidation | DSR calculation, Monte Carlo |

**Tech Stack:**
- Add scikit-learn, XGBoost, LightGBM
- MLflow for experiment tracking
- DVC for data versioning
- Consider PyTorch for later deep learning

**Potential Pitfalls:**

| Risk | Mitigation |
|------|------------|
| Training on full dataset | Implement CPCV from start |
| Feature leakage | Strict point-in-time generation |
| Model complexity | Start with gradient boosting |

**Validation Tests:**
- Walk-forward produces WFE > 50%
- DSR calculation matches expected values
- Feature importance is interpretable
- Predictions have correct timestamps

**Success Metrics:**
- ✅ ML model improves Sharpe in walk-forward
- ✅ No detected look-ahead bias
- ✅ Experiment tracking captures all trials

---

### 10.5 Phase 4: Paper Trading (Months 8-9)

**Objectives:**
- Connect to live market data
- Broker API integration (paper mode)
- Real-time operation

**Key Modules:**

| Module | Description |
|--------|-------------|
| LiveDataHandler | WebSocket connections, bar aggregation |
| BrokerAdapter | Abstract interface, paper implementation |
| OrderManager | Order lifecycle, state machine |
| ReconciliationEngine | Position verification |

**Tech Stack:**
- Add asyncio/aiohttp for async operations
- Redis for real-time data caching
- WebSocket libraries for exchanges

**Potential Pitfalls:**

| Risk | Mitigation |
|------|------------|
| WebSocket disconnection | Exponential backoff reconnection |
| Order state sync | Periodic reconciliation checks |
| Data gaps | Gap detection and handling |

**Validation Tests:**
- Paper trades execute at reasonable prices
- Position tracking matches broker
- Strategy signals match backtest logic
- System runs 24+ hours without crashes

**Success Metrics:**
- ✅ 4+ weeks paper trading without critical errors
- ✅ Execution latency < 500ms
- ✅ Fill prices within expected slippage

---

### 10.6 Phase 5: Live Trading (Months 10-11)

**Objectives:**
- Enable real money trading
- Appropriate safeguards
- Disaster recovery

**Key Modules:**

| Module | Description |
|--------|-------------|
| LiveExecutionHandler | Real order submission |
| FillTracker | Actual fill processing |
| CashManagement | Margin, buying power |
| DisasterRecovery | State persistence, restart |

**Tech Stack:**
- Encrypted credential storage (Vault/Secrets Manager)
- ELK stack or similar for logging

**Potential Pitfalls:**

| Risk | Mitigation |
|------|------------|
| API failure handling | Comprehensive try/catch |
| Missing audit trail | Immutable order logging |
| Over-trading bugs | Order rate limits, daily loss limits |

**Validation Tests:**
- Orders execute correctly in live market
- Fills recorded accurately
- Position reconciliation matches broker
- Graceful API error handling

**Success Metrics:**
- ✅ Live trading matches paper trading behavior
- ✅ No unintended trades
- ✅ Complete audit trail
- ✅ First month within expected variance

---

### 10.7 Phase 6: Monitoring and Risk Kill-Switches (Months 12-13)

**Objectives:**
- Production-grade monitoring
- Automated risk controls
- Alerting infrastructure

**Key Modules:**

| Module | Description |
|--------|-------------|
| KillSwitch | Multi-condition automated stop |
| MonitoringDashboard | Grafana integration |
| AlertingService | PagerDuty/Slack |
| PerformanceAnalytics | Rolling metrics, degradation detection |
| AuditLogger | Compliance-ready logging |

**Tech Stack:**
- Prometheus for metrics
- Grafana for visualization
- PagerDuty/Opsgenie for alerting
- Sentry for error tracking

**Potential Pitfalls:**

| Risk | Mitigation |
|------|------------|
| Alert fatigue | Tiered alerting with proper thresholds |
| False kill switch triggers | Confirmation for some conditions |
| Monitoring overhead | Async metrics collection |

**Validation Tests:**
- Kill switch activates under simulated conditions
- Alerts fire correctly
- Dashboard displays accurate real-time data
- Graceful recovery from kill switch

**Success Metrics:**
- ✅ MTTD < 5 minutes for critical issues
- ✅ False positive rate < 5%
- ✅ Complete audit trail for all actions

---

### 10.8 Phase 7: Multi-Strategy Scaling (Months 14-18)

**Objectives:**
- Multiple independent strategies
- Centralized risk management
- Capital allocation across strategies

**Key Modules:**

| Module | Description |
|--------|-------------|
| StrategyOrchestrator | Multi-strategy coordination |
| CentralRiskManager | Portfolio-level limits |
| ResourceAllocator | Capital allocation |
| StrategyHealthMonitor | Per-strategy tracking |
| StrategyMarketplace | Easy strategy deployment |

**Tech Stack:**
- Consider Kubernetes for container orchestration
- Kafka for event streaming at scale
- Rust components for performance-critical paths

**Potential Pitfalls:**

| Risk | Mitigation |
|------|------------|
| Strategy interference | Strict resource isolation |
| Correlated drawdowns | Correlation-based capital allocation |
| Complexity overwhelming team | Automation, documentation |

**Validation Tests:**
- Multiple strategies run independently
- Portfolio-level risk limits enforce correctly
- New strategy deployment without restart
- Performance acceptable with 10+ strategies

**Success Metrics:**
- ✅ 5+ strategies running simultaneously
- ✅ Portfolio-level Sharpe improvement
- ✅ Manageable operational burden

---

## 11. Technology Stack Summary

### 11.1 Core Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.11+ | Strategy logic, research, infrastructure |
| **Performance** | Rust (PyO3) | Performance-critical paths |
| **Time-Series DB** | QuestDB | Tick data, OHLCV storage |
| **Relational DB** | PostgreSQL | Metadata, configs, trade logs |
| **Cache** | Redis | Real-time data, feature serving |
| **Messaging** | Redis Streams → Kafka | Pub/sub (scale up later) |
| **Orchestration** | Apache Airflow | Batch pipelines, ML training |
| **Monitoring** | Prometheus + Grafana | Metrics, dashboards |
| **Alerting** | PagerDuty | On-call notifications |
| **Deployment** | Docker Compose → K8s | Containerization (scale up later) |

### 11.2 ML Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Experiment Tracking** | MLflow | Parameters, metrics, model registry |
| **Data Versioning** | DVC | Large file tracking |
| **Feature Store** | Feast | Point-in-time features |
| **Deep Learning** | PyTorch | TFT, GNN models |
| **Gradient Boosting** | XGBoost, LightGBM | Fast, interpretable models |
| **RL** | Stable-Baselines3 | PPO implementation |

### 11.3 Key Libraries

```python
# Core
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# ML
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
pytorch>=2.0.0
pytorch-forecasting>=1.0.0  # TFT implementation

# Data
sqlalchemy>=2.0.0
redis>=4.5.0
questdb>=1.0.0

# Backtesting
vectorbt>=0.25.0  # Research phase

# Visualization
matplotlib>=3.7.0
plotly>=5.15.0

# MLOps
mlflow>=2.5.0
dvc>=3.0.0
```

---

## 12. Risks and Mitigation Strategies

### 12.1 Risk Matrix

| Risk Category | Risk | Probability | Impact | Mitigation |
|---------------|------|-------------|--------|------------|
| **Technical** | Overfitting | High | Critical | CPCV, DSR, walk-forward from day one |
| **Technical** | Look-ahead bias | Medium | Critical | Structural prevention in data handler |
| **Technical** | System failure | Medium | High | Redundancy, automated failover |
| **Operational** | API outage | Medium | High | Multi-broker support, graceful degradation |
| **Operational** | Data gaps | Medium | Medium | Gap detection, interpolation |
| **Execution** | Slippage | High | Medium | Conservative assumptions (2-3x expected) |
| **Execution** | Partial fills | Medium | Medium | Order management state machine |
| **Market** | Regime change | High | High | Regime detection, strategy disabling |
| **Market** | Model decay | High | High | Rolling validation, continuous retraining |
| **Regulatory** | Compliance | Low | High | Audit logging, position limits |

### 12.2 Critical Success Factors

1. **Validation Discipline:** Never trust a backtest without CPCV, DSR, and walk-forward validation.

2. **Aggressive Simplicity:** Start with gradient boosting and basic features. Add complexity only when simpler approaches demonstrably fail.

3. **Operational Robustness:** Kill switches must function even when other systems fail.

4. **Continuous Monitoring:** Model degradation detection must be automated.

5. **Realistic Expectations:** A simple strategy that survives proper validation is worth infinitely more than a complex strategy with a beautiful but meaningless backtest.

---

## Appendix A: Recommended Reading

1. **López de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley.
2. **López de Prado, M.** (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
3. **Chan, E.** (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
4. **Jansen, S.** (2020). *Machine Learning for Algorithmic Trading*. Packt.

---

## Appendix B: Open Source References

| Project | URL | Use Case |
|---------|-----|----------|
| Nautilus Trader | github.com/nautechsystems/nautilus_trader | Production-grade backtester |
| VectorBT | github.com/polakowo/vectorbt | Research backtesting |
| Freqtrade | github.com/freqtrade/freqtrade | Crypto trading bot |
| MLflow | mlflow.org | Experiment tracking |
| Feast | feast.dev | Feature store |
| QuestDB | questdb.io | Time-series database |

---

*Document Version: 1.0*  
*Last Updated: February 2026*  
*Classification: Technical Research Report*
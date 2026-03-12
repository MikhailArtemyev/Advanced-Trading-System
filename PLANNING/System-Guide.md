# How the Trading System Works
## A Complete Guide to Understanding the Codebase

This document explains everything the system does and why. It starts with the big picture, then walks through each component step by step using concrete examples. Every financial and technical term is explained when it first appears.

---

## Table of Contents

1. [What This System Is](#1-what-this-system-is)
2. [The Core Idea: Event-Driven Backtesting](#2-the-core-idea-event-driven-backtesting)
3. [The Full Lifecycle of a Trade](#3-the-full-lifecycle-of-a-trade)
4. [Data: Where Prices Come From](#4-data-where-prices-come-from)
5. [Strategy: Deciding When to Trade](#5-strategy-deciding-when-to-trade)
6. [Portfolio: Deciding How Much to Trade](#6-portfolio-deciding-how-much-to-trade)
7. [Position Sizing: The Mathematics of "How Much"](#7-position-sizing-the-mathematics-of-how-much)
8. [Risk Management: The Safety Net](#8-risk-management-the-safety-net)
9. [Execution: Simulating Real Market Conditions](#9-execution-simulating-real-market-conditions)
10. [Portfolio Optimization: Balancing Multiple Assets](#10-portfolio-optimization-balancing-multiple-assets)
11. [Feature Engineering: Turning Raw Prices Into Useful Signals](#11-feature-engineering-turning-raw-prices-into-useful-signals)
12. [Machine Learning: Learning Patterns From Data](#12-machine-learning-learning-patterns-from-data)
13. [Validation: Making Sure Results Are Real](#13-validation-making-sure-results-are-real)
14. [Regime Detection: Reading the Market's Mood](#14-regime-detection-reading-the-markets-mood)
15. [Performance Measurement: How Good Is the Strategy?](#15-performance-measurement-how-good-is-the-strategy)
16. [Experiment Tracking: The Lab Notebook](#16-experiment-tracking-the-lab-notebook)
17. [Configuration: Controlling Everything From One File](#17-configuration-controlling-everything-from-one-file)
18. [How It All Fits Together](#18-how-it-all-fits-together)

---

## 1. What This System Is

This is a **backtesting engine** — software that simulates trading strategies against historical market data to see how they would have performed in the past.

**Why backtesting matters:** Before risking real money on a trading strategy, you want to test it. Did "buy when the short-term average crosses above the long-term average" actually make money over the last 5 years? Backtesting answers that question by replaying history day by day, making the decisions your strategy would have made, and tracking the results.

The system is designed to eventually transition from backtesting (historical simulation) to paper trading (live data, simulated trades) and finally to live trading (real money). The architecture makes this transition possible by keeping the core logic identical across all three modes.

---

## 2. The Core Idea: Event-Driven Backtesting

The system is built around **events** — messages that components pass to each other through a central queue, like notes in a suggestion box that get processed one at a time.

There are four types of events:

| Event | What It Means | Who Creates It | Who Receives It |
|-------|---------------|----------------|-----------------|
| **MarketEvent** | "A new price bar is available" | Engine | Strategy |
| **SignalEvent** | "I think we should buy/sell/exit" | Strategy | Portfolio |
| **OrderEvent** | "Buy 100 shares of AAPL at market price" | Portfolio | Risk Manager → Execution |
| **FillEvent** | "100 shares of AAPL were bought at $150.33" | Execution | Portfolio |

**Why events?** Two reasons:

1. **No cheating.** In a simpler system, you might accidentally use tomorrow's price to make today's decision (called **look-ahead bias**). Events enforce strict ordering — the strategy only sees data that was available at that point in time.

2. **Same code, different modes.** The strategy doesn't know whether it's processing historical data or live data. It just receives MarketEvents and emits SignalEvents. To switch from backtesting to live trading, you swap the data source and execution handler, but the strategy code stays identical.

The **EventQueue** (`src/events/queue.py`) is a simple first-in-first-out queue. Components put events in, and the engine pulls them out in order. It tracks how many events have been processed in total.

---

## 3. The Full Lifecycle of a Trade

Here is exactly what happens, step by step, when a new price bar arrives. We will use a concrete example throughout:

> **Scenario:** It's January 15, 2025. We are trading AAPL with a simple moving average crossover strategy. Our portfolio has $100,000 starting capital, and we currently have no positions.

### Step 1: Engine Advances to the Next Bar

The `BacktestEngine` (`src/backtest/engine.py`) is the conductor. It calls `data_handler.update_bars()`, which advances to the next day's data. Today's AAPL bar: Open $148.50, High $151.20, Low $148.10, Close $150.25, Volume 45M shares.

The engine creates a `MarketEvent(timestamp=2025-01-15)` and puts it on the queue.

```
Queue: [MarketEvent(2025-01-15)]
```

### Step 2: Strategy Analyzes the Data

The engine pulls the MarketEvent from the queue and calls `strategy.on_market_data(event, data_handler)`.

The strategy (let's say SMA Crossover with short_window=10, long_window=30) asks the data handler for the last 31 bars of AAPL closing prices. It computes:

- **10-day SMA** (Simple Moving Average): average of the last 10 closing prices = $149.80
- **30-day SMA**: average of the last 30 closing prices = $148.50

The strategy also remembers yesterday's values:
- Yesterday's 10-day SMA was $148.20
- Yesterday's 30-day SMA was $148.60

**What is a moving average?** It's the average price over a recent window. A 10-day SMA is the average of the last 10 closing prices. It smooths out daily noise to show the trend.

**What is a crossover?** When a shorter average crosses above a longer average, it suggests the price is trending upward — recent prices (10-day) are now higher than the longer trend (30-day). This is called a **bullish crossover** ("bullish" means expecting prices to rise, like a bull charging upward).

Yesterday: 10-day ($148.20) was *below* 30-day ($148.60)
Today: 10-day ($149.80) is *above* 30-day ($148.50)

The short-term average just crossed above the long-term average. The strategy creates a `SignalEvent(symbol="AAPL", signal_type=LONG, strength=1.0)` and puts it on the queue.

- **LONG** means "buy and hold, expecting the price to go up."
- **Strength 1.0** means "maximum confidence in this signal." Some strategies use a 0-to-1 scale; rule-based strategies typically use 1.0, while ML strategies might output 0.72 to express partial confidence.

```
Queue: [SignalEvent(AAPL, LONG, strength=1.0)]
```

### Step 3: Portfolio Determines Order Size

The engine pulls the SignalEvent and calls `portfolio.on_signal(event, data_handler)`.

The Portfolio (`src/portfolio/portfolio.py`) now needs to decide: *how many shares?* It delegates this to a **PositionSizer** (we'll use the default FixedFractionSizer at 10%):

- Current equity: $100,000 (our starting capital, no existing positions)
- Target allocation: 10% of equity = $10,000
- Current AAPL price: $150.25
- Shares to buy: floor($10,000 / $150.25) = **66 shares**

The portfolio checks if we have enough cash: 66 × $150.25 × 1.001 (including commission) = $10,026.57. We have $100,000. Plenty.

It creates an `OrderEvent(symbol="AAPL", side=BUY, quantity=66, order_type=MARKET)`.

- **MARKET order** means "buy at whatever the current price is." This is the simplest order type — you get the trade done immediately, but you don't control the exact price.
- A **LIMIT order** would say "buy only if the price is at or below $150.00." You control the price but might not get filled if the price never drops to your level.

```
Queue: [OrderEvent(AAPL, BUY, 66, MARKET)]
```

### Step 4: Risk Manager Checks the Order

The engine pulls the OrderEvent. Before executing it, the engine passes it to the `RiskManager` (`src/risk/risk_manager.py`). The risk manager runs through its checklist:

**Check 1 — Circuit Breaker:** Has the portfolio hit max drawdown? Our peak equity is $100,000, current equity is $100,000. Drawdown is 0%. Limit is 15%. **PASS.**

**Check 2 — Daily Loss Limit:** Have we lost too much today? Daily P&L is $0. Limit is 3% of equity ($3,000). **PASS.**

**Check 3 — Order Rate Limit:** Too many orders today? This is our first. Limit is 100 per day. **PASS.**

**Check 4 — Open Position Count:** Do we have too many open positions? We have 0. Limit is 20. **PASS.**

**Check 5 — Position Concentration:** Would this position be too large relative to our portfolio? 66 shares × $150.25 = $9,916 = 9.9% of equity. Limit is 10%. **PASS** (just under the limit).

**Check 6 — Portfolio Exposure:** Would our total exposure be too high? Current exposure is $0, adding $9,916 = $9,916 = 9.9% of equity. Limit is 100%. **PASS.**

All checks pass. The order proceeds to execution.

> **What if a check fails?** If, say, we already had 20 open positions (check 4), the risk manager would return `RiskCheckResult(approved=False, rejection_reasons=["Max open positions: 20 >= 20"])`. The order would be discarded, and the engine would increment its `rejected_orders` counter. The strategy keeps running — it will generate new signals on the next bar.

### Step 5: Execution Simulates the Fill

The engine passes the approved order to `ExecutionHandler` (`src/execution/execution_handler.py`).

The execution handler simulates what would happen in a real market:

1. **Base price:** The current closing price of AAPL = $150.25

2. **Slippage:** In real markets, when you place a buy order, your demand pushes the price up slightly. The system simulates this:
   - Slippage rate: 0.05% (configurable)
   - Fill price: $150.25 × (1 + 0.0005) = **$150.33**
   - For sell orders, slippage works against you the other way — the price drops slightly.

3. **Commission:** Brokers charge fees for executing trades:
   - Commission rate: 0.1% of trade value (configurable)
   - Trade value: 66 × $150.33 = $9,921.78
   - Commission: $9,921.78 × 0.001 = **$9.92**

The execution handler creates a `FillEvent(symbol="AAPL", side=BUY, quantity=66, fill_price=150.33, commission=9.92)`.

```
Queue: [FillEvent(AAPL, BUY, 66, $150.33, commission=$9.92)]
```

### Step 6: Portfolio Updates Its Books

The engine pulls the FillEvent and calls `portfolio.on_fill(event)`.

The portfolio updates everything:

- **Cash:** $100,000 − (66 × $150.33) − $9.92 = **$90,068.30**
- **Position:** AAPL: 66 shares, average cost $150.33
- **Trade log:** Records the trade with timestamp, price, quantity, commission
- **Equity:** cash ($90,068.30) + market value of positions (66 × $150.25) = **$99,984.80**

Wait — our equity dropped from $100,000 to $99,984.80? Yes, because of slippage ($5.28) and commission ($9.92). These are the costs of trading, and modeling them realistically is critical. A strategy that looks profitable without costs might actually lose money once you account for them.

The engine also tells the strategy: `strategy.update_position("AAPL", 66)`. Now the strategy knows we own 66 shares, so it won't generate another BUY signal until we've exited.

### Step 7: Performance Tracking

The engine calls `performance_tracker.update(timestamp, equity)`, recording today's equity of $99,984.80. After the backtest finishes, the tracker computes performance metrics from this equity history (Sharpe ratio, drawdown, etc. — covered in Section 15).

### Step 8: Risk Manager Updates

The engine updates the risk manager's peak equity tracker: `risk_manager.update_peak_equity(99984.80)`. Since this is below our previous peak of $100,000, the peak stays at $100,000.

**Then the next bar arrives, and it all repeats.**

---

## 4. Data: Where Prices Come From

**Location:** `src/data/`

The `DataHandler` is an **abstract base class (ABC)** — a template that defines *what* a data source must do, without specifying *how*. This allows the system to swap data sources without changing any other code.

Every DataHandler must support:

| Method | Purpose |
|--------|---------|
| `get_latest_bars(symbol, n)` | "Give me the last n bars of data for this symbol" |
| `update_bars()` | "Advance to the next time period" |
| `get_current_timestamp()` | "What time is it in the simulation?" |
| `continue_backtest` | "Is there more data to process?" |
| `reset()` | "Start over from the beginning" |

**What is a "bar"?** A bar is a summary of price activity over a time period (usually one day). It contains:
- **Open:** the first price of the day
- **High:** the highest price reached during the day
- **Low:** the lowest price
- **Close:** the last price of the day (often the most important)
- **Volume:** how many shares were traded

The system has two concrete implementations:

### HistoricalCSVDataHandler
Reads price data from CSV files on disk. Fast, no internet needed, completely reproducible. Each symbol gets its own CSV file with columns: datetime, open, high, low, close, volume.

The handler pre-loads all data into memory, then simulates the passage of time by advancing an internal index one bar at a time. When the strategy calls `get_latest_bars("AAPL", 50)`, it returns only the last 50 bars *up to the current index* — never future data.

### YFinanceDataHandler
Downloads data from Yahoo Finance. Useful for quickly testing against real market data without manually collecting CSV files. Same interface — once downloaded, it behaves identically to the CSV handler.

---

## 5. Strategy: Deciding When to Trade

**Location:** `src/strategy/`

A strategy is the core trading logic — the rules that decide when to buy, sell, or exit. The `Strategy` ABC (`src/strategy/base_strategy.py`) defines the interface:

- `calculate_signals(timestamp, data_handler)` → list of SignalEvents
- `update_position(symbol, quantity)` → tell the strategy about current holdings
- `get_position(symbol)` → check what we currently hold

Every strategy has access to the data handler and can request historical bars. It tracks its own positions so it knows whether it's already long/short before generating signals.

### SMA Crossover Strategy

**Location:** `src/strategy/sma_crossover.py`

The simplest strategy. It uses two **Simple Moving Averages** of different lengths:

**How it works:**
- Compute a "fast" SMA (e.g., 10-day average of closing prices)
- Compute a "slow" SMA (e.g., 30-day average)
- When the fast SMA crosses *above* the slow SMA → BUY signal (LONG)
- When the fast SMA crosses *below* the slow SMA → EXIT signal (sell existing position)

**The intuition:** The fast average reacts to recent price changes quickly. The slow average represents the longer trend. When the fast crosses above the slow, it means recent prices are accelerating upward relative to the longer trend. When it crosses below, the momentum is fading.

**Example with real numbers:**

| Day | Close | 10-day SMA | 30-day SMA | Signal |
|-----|-------|-----------|-----------|--------|
| Mon | $148.00 | $147.50 | $148.00 | — (fast below slow) |
| Tue | $149.50 | $148.80 | $148.10 | — (fast below slow) |
| Wed | $151.00 | $149.20 | $148.15 | BUY (fast crossed above slow) |
| Thu | $150.50 | $149.50 | $148.20 | — (already holding) |
| ... | ... | ... | ... | ... |
| Fri | $145.00 | $147.80 | $148.30 | EXIT (fast crossed below slow) |

**Why crossovers work (sometimes):** Markets often exhibit momentum — prices that have been rising tend to continue rising for a while. The crossover captures the beginning of these trends. It doesn't work in choppy, directionless markets where the averages keep crossing back and forth ("whipsawing"), generating many small losses.

### Multi-Asset SMA Strategy

**Location:** `src/strategy/multi_asset_sma.py`

Same logic as the single-symbol version, but runs independently for each symbol in the portfolio. If you're trading AAPL, MSFT, and GOOGL, it checks for crossovers in each one separately.

### ML Strategy

**Location:** `src/ml/ml_strategy.py`

Instead of fixed rules like "crossover = buy," this strategy uses a machine learning model to decide. Covered in detail in Section 12.

---

## 6. Portfolio: Deciding How Much to Trade

**Location:** `src/portfolio/portfolio.py`

The Portfolio is the bookkeeper. It receives signals from the strategy, determines order sizes, and tracks every aspect of the portfolio's state.

### What the Portfolio Tracks

**Positions:** For each symbol, the portfolio maintains:
- **Quantity:** How many shares we hold. Positive = long (we own them, expecting price to rise). Negative = short (we've borrowed and sold shares, expecting price to fall, planning to buy them back cheaper).
- **Average cost:** The average price we paid per share. Used to calculate profit/loss.
- **Market value:** Current price × quantity. What our position is worth right now.
- **Unrealized P&L:** Profit or loss we would realize if we closed the position right now. If we bought 100 shares at $150 and the price is now $155, our unrealized P&L is +$500.
- **Realized P&L:** Profit or loss from trades we've already closed.

**Cash:** Money available for new trades. Starts at initial capital, decreases with buys, increases with sells.

**Equity:** The total portfolio value: cash + market value of all positions. This is the number that matters — if it goes up, the strategy is working.

**Equity curve:** A time series of equity values. Each bar, the portfolio records the current equity. This curve is the raw material for all performance metrics.

### How Order Generation Works

When the portfolio receives a LONG signal:
1. Ask the PositionSizer how many shares to buy (Section 7)
2. Check if we have enough cash
3. If cash is insufficient, reduce the quantity to what we can afford
4. Create an OrderEvent with the final quantity

When the portfolio receives an EXIT signal:
1. Check if we have a position to close
2. If long, create a SELL order for the full quantity
3. If short, create a BUY order to cover the short

When the portfolio receives a SHORT signal:
1. Only allowed if we have no existing position (the system won't flip from long to short in one step — it requires an EXIT first)
2. Ask the PositionSizer for the short quantity
3. Create a SELL order (selling shares we don't own — we borrow them)

### How Fill Processing Works

When a BUY fill arrives:
- **Adding to a long position or opening new long:** Cash decreases by (quantity × price + commission). The position's average cost is recalculated as a weighted average.
- **Covering a short:** Cash decreases (we're buying back shares we borrowed). Profit = (original sell price − buy-back price) × quantity. P&L is realized.

When a SELL fill arrives:
- **Closing a long position:** Cash increases by (quantity × price − commission). Profit = (sell price − average cost) × quantity. P&L is realized.
- **Opening a short position:** Cash increases (we receive money from selling borrowed shares). Average cost is set to the sell price.

### Correlation Tracking

**Location:** `src/portfolio/correlation.py`

When trading multiple assets, it's important to know if they move together. If AAPL and MSFT are 95% correlated (they almost always move in the same direction by similar amounts), owning both doesn't truly diversify your risk — it's almost like having one bigger position.

The `CorrelationTracker` maintains a rolling window of prices for each symbol and computes **Pearson correlation** between their returns on demand.

**What is correlation?** A number from −1 to +1:
- **+1:** Perfect positive correlation. When A goes up 2%, B goes up ~2%.
- **0:** No relationship. A and B move independently.
- **−1:** Perfect negative correlation. When A goes up 2%, B goes down ~2%.

Correlation is computed on *returns* (percentage changes), not raw prices, because prices trend over time and would always appear correlated.

---

## 7. Position Sizing: The Mathematics of "How Much"

**Location:** `src/risk/position_sizer.py`

Position sizing is arguably more important than the strategy itself. A good strategy with poor sizing can blow up; a mediocre strategy with disciplined sizing can survive long enough to profit. The system provides three sizing methods.

### Fixed Fraction Sizer

The simplest approach: invest a fixed percentage of your portfolio in each trade.

**Formula:**
```
target_value = equity × fraction
quantity = floor(target_value / current_price)
```

**Example:** With $100,000 equity and a 10% fraction, you invest $10,000 per trade. At $150/share, that's 66 shares.

**Why it works:** As your portfolio grows, your positions grow proportionally. As your portfolio shrinks from losses, positions shrink too, automatically reducing risk. You can never bet everything on one trade.

### Volatility-Based Sizer

Sizes positions inversely proportional to how volatile the asset is. More volatile assets get smaller positions.

**Key concept — ATR (Average True Range):** ATR measures how much an asset's price typically moves in a day. It's the average of the "true range" over a lookback period (typically 14 days).

**True Range** is the largest of:
- Today's high minus today's low
- Today's high minus yesterday's close (gap up)
- Yesterday's close minus today's low (gap down)

If a stock has an ATR of $3, it typically moves about $3 per day. If another stock has an ATR of $0.50, it's much calmer.

**Formula:**
```
risk_per_share = ATR × multiplier     (e.g., 2 × ATR = $6 per share)
max_risk = equity × risk_fraction     (e.g., 2% of $100,000 = $2,000)
quantity = floor(max_risk / risk_per_share)   (e.g., $2,000 / $6 = 333 shares)
```

**Why it matters:** If you use fixed fractions, a $10,000 position in a calm stock (ATR $0.50) risks about $100/day, while the same $10,000 in a volatile stock (ATR $3.00) risks about $600/day. Volatility-based sizing makes the *risk* equal, not the dollar amount.

### Kelly Criterion Sizer

The mathematically optimal bet size, given your strategy's historical win rate and payoff ratio.

**The Kelly Formula:**
```
Kelly % = W − (1 − W) / R
```
Where:
- **W** = Win rate (what fraction of your trades are profitable)
- **R** = Win/loss ratio (average winning trade ÷ average losing trade)

**Example:** If your strategy wins 55% of the time (W = 0.55) and average wins are $200 while average losses are $100 (R = 2.0):
```
Kelly % = 0.55 − (1 − 0.55) / 2.0 = 0.55 − 0.225 = 0.325
```
You should bet 32.5% of your portfolio per trade.

**Fractional Kelly:** Full Kelly is aggressive — it can experience 50-70% drawdowns. The system uses **half-Kelly** by default (Kelly fraction = 0.5), so in the example above, you'd bet 16.25% instead of 32.5%.

**Signal strength scaling:** Kelly sizing is further scaled by the signal strength. If the strategy gives a signal with strength 0.7 instead of 1.0, the position is 30% smaller.

**Minimum trades requirement:** Kelly estimates are unreliable with few trades. The sizer requires at least 20 completed trades before using the Kelly formula; before that, it falls back to a default fixed fraction.

---

## 8. Risk Management: The Safety Net

**Location:** `src/risk/risk_manager.py`

The RiskManager sits between the Portfolio (which creates orders) and the ExecutionHandler (which fills them). It can reject or shrink any order that violates safety limits.

### The Seven Risk Checks (in priority order)

**Check 1 — Circuit Breaker (Max Drawdown Halt)**

**What is drawdown?** The percentage decline from the portfolio's peak value. If your portfolio peaked at $120,000 and is now at $102,000, your drawdown is ($120,000 − $102,000) / $120,000 = 15%.

If drawdown exceeds the limit (default: 15%), the system halts *all trading*. No more orders will be approved until the halt is manually reset. This prevents a catastrophic losing streak from destroying the portfolio.

**Check 2 — Daily Loss Limit**

Tracks realized P&L for the current day. If cumulative losses today exceed the limit (default: 3% of equity), all further orders are rejected until the next trading day.

**Why daily limits matter:** Even good strategies have bad days. A daily loss limit prevents a single disastrous day from doing permanent damage. Professional trading desks almost always have daily loss limits.

**Check 3 — Order Rate Limit**

Maximum number of orders per day (default: 100). Prevents runaway strategies that might submit thousands of orders due to bugs or whipsaw conditions.

**Check 4 — Open Position Count**

Maximum simultaneous open positions (default: 20). Prevents over-diversification (too many small positions are hard to manage) and limits exposure breadth.

**Check 5 — Position Concentration (Adjustable)**

No single position can exceed a percentage of portfolio equity (default: 10%). Unlike checks 1-4 which reject outright, this check can *reduce* the order quantity. If you try to buy 100 shares but that would make the position 12% of equity, it reduces to a quantity that keeps it at 10%.

**Check 6 — Portfolio Exposure (Adjustable)**

Total exposure (sum of all position market values) cannot exceed a percentage of equity (default: 100%). Like check 5, this can reduce order quantity rather than rejecting entirely. An exposure limit of 100% means you can't use leverage (borrow money to invest more than you have).

**What is exposure?** The total dollar value of all your positions relative to your equity. 60% exposure means 60% of your portfolio is invested and 40% is in cash. 150% exposure would mean you've borrowed money (leverage) to invest 1.5× your equity.

**Check 7 — Correlation (Placeholder)**

Reserved for future implementation — would reject orders that would increase correlation concentration (e.g., refusing to buy MSFT when you already own a lot of AAPL, since they're highly correlated).

### Day Rollover

The risk manager detects when a new trading day begins and automatically resets the daily P&L counter and order count.

---

## 9. Execution: Simulating Real Market Conditions

**Location:** `src/execution/execution_handler.py`

In backtesting, we need to simulate what would actually happen when an order hits the market. The ExecutionHandler handles two realities of trading:

### Slippage

**What is slippage?** The difference between the price you expected and the price you actually got. In real markets, when you buy, your order pushes the price up slightly (you're adding demand). When you sell, you push the price down slightly.

The system models this with a simple percentage:
- BUY orders: fill_price = close_price × (1 + slippage_pct)
- SELL orders: fill_price = close_price × (1 − slippage_pct)

Default slippage is 0.05% (5 basis points). On a $150 stock, that's about $0.075 per share. Seems tiny, but across hundreds of trades it adds up significantly.

**A "basis point"** is 1/100th of a percent, or 0.01%. Finance uses basis points because percentage changes in rates and costs are often very small. 50 basis points = 0.50%.

### Commission

**What is commission?** The fee your broker charges for executing a trade. Modeled as a percentage of trade value (default: 0.1%).

Example: buying 66 shares at $150.33 = $9,921.78 trade value. Commission = $9.92.

### Why Costs Matter

Consider a strategy that generates 500 trades per year:
- Commission: 500 × $10 average = $5,000/year
- Slippage: 500 × $5 average = $2,500/year
- Total cost: $7,500/year

On a $100,000 portfolio, that's 7.5% annually eaten by costs alone. Your strategy needs to earn more than 7.5% just to break even. Many strategies that look profitable in a cost-free simulation actually lose money in practice. This is why realistic cost modeling is non-negotiable.

---

## 10. Portfolio Optimization: Balancing Multiple Assets

**Location:** `src/optimization/`

When trading multiple assets, you need to decide what percentage of your portfolio goes to each one. This is called **asset allocation** or **portfolio optimization**.

### The Problem

Say you're trading AAPL, MSFT, GOOGL, and AMZN. Should you put 25% in each? Maybe, but what if AAPL and MSFT are highly correlated? Then you're not really diversified — two of your four bets are essentially the same bet.

### Mean-Variance Optimization (Markowitz)

**Location:** `src/optimization/mean_variance.py`

**The idea:** Find the portfolio weights that give the best trade-off between expected return and risk (measured as volatility). Invented by Harry Markowitz in 1952 (won the Nobel Prize for it).

**Inputs:**
- Expected returns for each asset (estimated from historical average daily returns × 252 trading days)
- **Covariance matrix:** how assets move together. If AAPL and MSFT both go up on the same days, they have positive covariance. If they move independently, covariance is near zero.

**Two optimization targets:**
1. **Max Sharpe:** Find weights that maximize the **Sharpe ratio** (return per unit of risk). This is the "best risk-adjusted return" portfolio.
2. **Min Variance:** Find weights that minimize total portfolio volatility. This is the "safest" portfolio.

**Constraints:** Each weight is bounded (default: 0% minimum, 30% maximum per asset) and all weights must sum to 100%.

**Ledoit-Wolf Shrinkage:** When estimating covariance from limited data (say, 100 daily returns for 10 assets), the sample covariance matrix can be noisy and unstable. Shrinkage blends the sample covariance toward a simpler "target" matrix (scaled identity — assumes all assets have the same variance and zero correlation). This produces more stable, reliable estimates.

**Example output:**
```
AAPL: 30%  (hit the max weight cap)
MSFT: 15%
GOOGL: 25%
AMZN: 30%  (hit the max weight cap)
Expected return: 12.3% annualized
Expected volatility: 18.1% annualized
Sharpe ratio: 0.57
```

### Risk Parity

**Location:** `src/optimization/risk_parity.py`

**The idea:** Instead of maximizing returns, make each asset contribute *equally to portfolio risk*. This way, no single asset dominates your risk exposure.

**Why risk parity?** Mean-variance optimization is theoretically optimal but very sensitive to estimation errors in expected returns. Risk parity doesn't need return estimates at all — it only uses the covariance matrix, which is much more stable to estimate.

**How it works:** The optimizer finds weights such that each asset's **marginal contribution to risk** (how much adding one more dollar to that asset changes total portfolio volatility) is equal for all assets. In practice, this means high-volatility assets get lower weights and low-volatility assets get higher weights.

**Example:** If bonds have 5% volatility and stocks have 20% volatility, risk parity would allocate roughly 4× more to bonds than stocks, so both contribute equally to portfolio risk.

---

## 11. Feature Engineering: Turning Raw Prices Into Useful Signals

**Location:** `src/features/`

Raw OHLCV data (open, high, low, close, volume) is not directly useful for machine learning. Feature engineering transforms it into meaningful numerical signals that ML models can learn from.

### The Pipeline

The `FeaturePipeline` (`src/features/pipeline.py`) orchestrates multiple feature generators. You configure which features to compute, and the pipeline runs them all, combines the results, and aligns everything with a target variable.

### Technical Indicator Features

**Location:** `src/features/technical.py`

These are classic trading indicators that quantify different aspects of price behavior:

**SMA (Simple Moving Average):** Already covered in the strategy section. The feature pipeline computes SMAs at multiple windows (5, 10, 20, 50 days) and also the **SMA ratio** (close / SMA), which measures how far the current price is from its average. A ratio above 1.0 means the price is above average (potentially overbought); below 1.0 means below average (potentially oversold).

**RSI (Relative Strength Index):** Measures the speed and magnitude of recent price changes on a scale from 0 to 100.
- RSI > 70: "overbought" — the price has risen fast and may be due for a pullback
- RSI < 30: "oversold" — the price has fallen fast and may bounce back
- RSI around 50: neutral

The formula tracks the ratio of average upward moves to average downward moves over a period (typically 14 days).

**MACD (Moving Average Convergence Divergence):** Three values:
- **MACD line:** difference between the 12-day and 26-day exponential moving averages (EMAs). An EMA is like an SMA but gives more weight to recent prices.
- **Signal line:** a 9-day EMA of the MACD line (smoothed version of the MACD)
- **Histogram:** MACD − Signal. Positive = bullish momentum, negative = bearish.

When the MACD crosses above its signal line, it's a bullish sign. When it crosses below, it's bearish. Similar to SMA crossover but responds faster due to the exponential weighting.

**Bollinger Bands:** A band drawn around the price using a moving average ± 2 standard deviations.
- **BB Width:** How wide the bands are. Narrow bands = low volatility (a big move may be coming). Wide bands = high volatility.
- **%B (Percent B):** Where the current price sits within the bands. %B = 0 means at the lower band, %B = 1 means at the upper band, %B = 0.5 means at the middle.

**ATR (Average True Range):** Already covered in position sizing. The pipeline computes it as a feature too, normalized by the current price, so the model can learn "this stock is currently more/less volatile than usual."

### Statistical Features

**Location:** `src/features/statistical.py`

These capture the statistical properties of returns:

**Z-Score:** How many standard deviations the current return is from the mean. A z-score of +2 means the price moved up much more than usual — potentially an anomaly or the start of a trend.

**Rolling Skewness:** Whether recent returns are symmetric or lopsided. Negative skewness means there have been more extreme negative days than positive ones — the distribution of returns has a "fat left tail."

**Rolling Kurtosis:** How "fat-tailed" the return distribution is. High kurtosis means extreme moves (both up and down) are more common than a normal bell curve would predict. Financial data almost always has higher kurtosis than normal.

**Hurst Exponent:** Measures whether the price is trending or mean-reverting:
- Hurst > 0.5: Trending (up moves tend to follow up moves)
- Hurst = 0.5: Random walk (no predictable pattern)
- Hurst < 0.5: Mean-reverting (up moves tend to be followed by down moves)

### Target Variable

The pipeline also computes the **target** — what we're trying to predict. Typically this is the forward return: "Did the price go up or down over the next N days?" For classification models, this becomes a binary label (1 = price went up, 0 = price went down).

---

## 12. Machine Learning: Learning Patterns From Data

**Location:** `src/ml/`

Instead of manually defining rules ("buy when SMA crosses"), ML models learn the rules from historical data. You show them thousands of examples of "here are the features on this day, and here's what happened next," and they figure out which patterns are predictive.

### MLModel ABC

**Location:** `src/ml/base_model.py`

All ML models implement two core methods:
- `train(features, target)` → learns patterns from historical data
- `predict(features)` → produces predictions for new data

Each prediction includes:
- **Signal:** a number indicating direction and magnitude (-1 to +1 range typically)
- **Confidence:** how confident the model is (0 to 1)

### XGBoost and LightGBM

**Location:** `src/ml/xgboost_model.py`, `src/ml/lightgbm_model.py`

Both are **gradient-boosted tree** models — the workhorse of modern ML for structured data (tables of numbers, as opposed to images or text).

**What is a decision tree?** A flowchart of yes/no questions. "Is RSI > 70? → Yes → Is MACD positive? → No → Predict: price will drop." A single tree is simple but crude.

**What is boosting?** Building many trees sequentially, where each new tree focuses on fixing the mistakes of the previous ones. Tree 1 makes predictions. Tree 2 is trained on the errors of Tree 1. Tree 3 is trained on the remaining errors, and so on. The final prediction is the sum of all trees.

**XGBoost vs LightGBM:** Both implement gradient boosting but with different algorithms for building the trees. LightGBM is generally faster on large datasets; XGBoost is more established. Both are excellent. The system supports both so you can compare.

**Two modes:**
- **Classification:** Predict a class label (1 = price goes up, 0 = price goes down). Confidence is the probability the model assigns to the predicted class.
- **Regression:** Predict a continuous number (e.g., the expected forward return of 0.3%). Signal is the prediction itself.

### MLStrategy: Bridging ML to the Event Loop

**Location:** `src/ml/ml_strategy.py`

The MLStrategy plugs an ML model into the event-driven system. At each bar:

1. Fetch enough historical bars for feature computation
2. Run the feature pipeline to compute today's features
3. Feed features to the trained model for prediction
4. Map the prediction to a trading signal:
   - prediction.signal > threshold (e.g., 0.1) → LONG signal
   - prediction.signal < −threshold → EXIT or SHORT signal
   - |prediction.signal| < exit_threshold → EXIT (model is uncertain)

The model's confidence becomes the signal strength, which influences position sizing. A high-confidence prediction leads to a larger position.

---

## 13. Validation: Making Sure Results Are Real

**Location:** `src/validation/`

This is perhaps the most important part of the system. A strategy that "backtests well" might just be lucky, or it might be overfitting to the specific historical data. The validation framework provides mathematical tools to distinguish skill from luck.

### The Overfitting Problem

**What is overfitting?** When a model learns the noise in historical data rather than genuine patterns. An overfitted model performs brilliantly on the data it was trained on but terribly on new, unseen data.

**Analogy:** Imagine studying for an exam by memorizing every answer to every practice test. You'd ace the practice tests but fail a new exam with different questions. You memorized the answers instead of learning the concepts.

In trading, overfitting means your model learned "AAPL went up on March 12, 2020 when RSI was 34" — a specific historical coincidence — rather than the general pattern "deeply oversold stocks tend to bounce."

### Combinatorial Purged Cross-Validation (CPCV)

**Location:** `src/validation/cpcv.py`

Standard machine learning validation (k-fold cross-validation) doesn't work properly for financial time series because of **information leakage.**

**The leakage problem:** Suppose you're predicting 5-day forward returns. If Day 100 is in your training set and Day 102 is in your test set, their target values overlap — they're both measuring returns partially over the same future days. The model has effectively "seen" part of the test answer during training.

**CPCV fixes this with three mechanisms:**

1. **Purging:** Removes training samples that are close in time to the test set boundary. If the target horizon is 5 days, samples within 5 days of the test boundary are removed from training. This eliminates the overlap.

2. **Embargo:** Adds an additional buffer after each test block. Even after purging, financial returns exhibit serial correlation (today's return is slightly predictive of tomorrow's). The embargo provides extra separation.

3. **Combinatorial paths:** Standard cross-validation gives you K results (one per fold). CPCV generates C(N, k) paths — all possible combinations of N groups taken k at a time. With 6 groups and 2 test groups, that's C(6,2) = 15 paths. This gives a much richer picture of strategy performance across many different train/test splits.

**Example:** With 1000 days of data, 6 groups, 2 test groups:
- Data is split into 6 contiguous blocks of ~167 days each
- 15 different train/test combinations are evaluated
- Each combination has purging and embargo applied
- You get 15 performance scores instead of just one

If the model scores well across all 15 paths, it's likely learning genuine patterns. If it scores great on some paths and terribly on others, it's probably overfitting.

### Deflated Sharpe Ratio (DSR)

**Location:** `src/validation/deflated_sharpe.py`

**The multiple testing problem:** If you test 100 different parameter combinations for your strategy and pick the one with the best Sharpe ratio, you've engaged in **data mining**. Even with purely random returns, the best of 100 random Sharpe ratios will be significantly positive — it's not evidence of skill, just luck.

**What is the Sharpe ratio?** The most common measure of risk-adjusted return. It's the average excess return (return above the risk-free rate) divided by the standard deviation of returns. A Sharpe of 1.0 means you earned 1 unit of return per unit of risk. Higher is better.

The DSR adjusts for this by computing:

1. **Expected maximum Sharpe under the null:** If you run N independent trials with no true skill, what Sharpe would you expect the best one to have? This is computed from the statistics of order statistics (the expected maximum of N standard normal draws).

2. **Probabilistic Sharpe:** Given the observed Sharpe ratio, how likely is it that the true Sharpe exceeds the null benchmark? This accounts for the uncertainty in the Sharpe estimate, including the effects of non-normal returns (skewness and kurtosis).

**Example:**
- You tested 50 parameter combinations
- The best one has an observed Sharpe of 1.2
- Expected max Sharpe under the null (no skill): 0.85
- Deflated Sharpe: 1.2 − 0.85 = 0.35
- p-value: 0.72

A p-value of 0.72 means there's a 72% chance the true Sharpe exceeds the null benchmark. That's not great — you'd typically want >95% confidence. The strategy's impressive-looking Sharpe of 1.2 is partially explained by having tried 50 combinations.

**Contrast:** If you'd only tested 3 parameter combinations:
- Expected max Sharpe under the null: 0.32
- Deflated Sharpe: 1.2 − 0.32 = 0.88
- p-value: 0.97

Now you have 97% confidence. The fewer things you try, the more meaningful each result is.

---

## 14. Regime Detection: Reading the Market's Mood

**Location:** `src/regime/hmm.py`

Markets don't behave the same way all the time. Sometimes they trend steadily upward (bull market), sometimes they crash (bear market), sometimes they chop sideways. A strategy that works in one regime may fail in another.

### Hidden Markov Model (HMM)

**What is an HMM?** A statistical model that assumes the system being modeled switches between a fixed number of hidden "states" over time. "Hidden" means we can't directly observe which state we're in — we can only see the data it produces and infer the state.

**Analogy:** Imagine a weather model with two hidden states: "summer weather pattern" and "winter weather pattern." You can't see the pattern directly, but you can observe the temperature and humidity. From those observations, you can infer which pattern is active.

For markets, the hidden states are regimes:
- **Bull:** High average returns, moderate volatility
- **Bear:** Negative average returns, high volatility
- **Sideways:** Near-zero returns, low volatility

### How It Works

The `RegimeDetector` uses two observable features:
1. **Daily returns:** How much the price changed today (percentage)
2. **Rolling volatility:** Standard deviation of returns over a window (default 20 days)

It fits a Gaussian HMM to these observations, which learns:
- The typical return and volatility distributions for each regime
- The **transition matrix:** the probability of switching from one regime to another

**Transition matrix example:**
```
          Tomorrow→  Bull  Sideways  Bear
Today ↓
Bull                 0.92    0.06    0.02
Sideways             0.05    0.90    0.05
Bear                 0.03    0.07    0.90
```

This says: if we're in a bull regime today, there's a 92% chance we'll still be in a bull regime tomorrow, 6% chance of switching to sideways, and 2% chance of switching to bear. Regimes are "sticky" — they tend to persist.

### Why Regime Detection Matters

A momentum strategy (like SMA crossover) might work great in bull markets but lose money in choppy sideways markets. If the regime detector says we're in a sideways market, the system could:
- Switch to a mean-reversion strategy instead
- Reduce position sizes
- Tighten risk limits
- Stop trading entirely

---

## 15. Performance Measurement: How Good Is the Strategy?

**Location:** `src/performance/metrics.py`

After the backtest finishes, the PerformanceTracker computes a comprehensive set of metrics. Here's what each one means and why it matters:

### Return Metrics

**Total Return:** How much the portfolio grew overall, as a percentage.
- Formula: (final_equity / initial_capital − 1) × 100
- Example: $100,000 → $115,000 = 15% total return

**Annualized Return:** The total return normalized to a per-year rate.
- A 15% return over 2 years ≈ 7.2% annualized
- This allows comparing strategies run over different time periods
- Formula: (1 + total_return)^(1 / years) − 1

### Risk Metrics

**Volatility:** How much the portfolio's value fluctuates, annualized.
- Computed as the standard deviation of daily returns × √252
- Example: daily std of 0.8% → annualized volatility of 12.7%
- Lower volatility means smoother performance — less stomach-churning ups and downs

**Maximum Drawdown:** The worst peak-to-trough decline.
- If your portfolio goes from $120,000 to $96,000 before recovering, the max drawdown is −20%
- This is arguably the most important risk metric because it measures the worst pain you'd actually experience
- It's always negative (or zero if you never lose money)

### Risk-Adjusted Metrics

These combine return and risk into a single number, because a 20% return with 5% volatility is very different from a 20% return with 40% volatility.

**Sharpe Ratio:** (Annualized return − risk-free rate) / annualized volatility
- Measures units of return per unit of risk
- Sharpe > 1.0: good
- Sharpe > 2.0: excellent
- Sharpe < 0: you'd have been better off in risk-free bonds
- The risk-free rate (default 2%) represents what you could earn with zero risk (e.g., Treasury bonds)

**Sortino Ratio:** Like Sharpe, but only penalizes *downside* volatility.
- The Sharpe ratio treats upside volatility (big gains) the same as downside volatility (big losses). But who complains about unexpectedly large gains?
- Sortino only divides by the standard deviation of negative returns
- A strategy with occasional huge wins and small losses has a better Sortino than Sharpe

**Calmar Ratio:** Annualized return / |max drawdown|
- Measures return relative to the worst-case pain
- Example: 12% annualized return, 20% max drawdown → Calmar = 0.6
- A Calmar of 1.0 means you earned at least as much annually as your worst drawdown

### Trade Metrics

**Win Rate:** Percentage of trades that were profitable.
- Win rate alone is not very informative. A strategy can have 30% win rate but be very profitable if the average win is 3× the average loss.

**Profit Factor:** Gross profit / gross loss.
- Profit factor > 1.0: strategy is profitable overall
- Profit factor of 2.0: the strategy makes $2 for every $1 it loses
- Formula: sum of all winning trades / |sum of all losing trades|

**Average Win / Average Loss:** The typical size of winning and losing trades.
- Combined with win rate, these tell the full story
- A strategy with 40% win rate but average win 3× average loss is solid

---

## 16. Experiment Tracking: The Lab Notebook

**Location:** `src/tracking/mlflow_tracker.py`

When you're testing dozens of strategy variations (different parameters, different models, different feature sets), it's easy to lose track of what you tried and what worked. The ExperimentTracker wraps MLflow to maintain a complete record.

**What is MLflow?** An open-source platform for managing the ML lifecycle. It stores:
- **Parameters:** What configuration was used (strategy name, short_window=10, commission=0.001, etc.)
- **Metrics:** What results were achieved (Sharpe=0.68, max_drawdown=−12.5%, total_return=8.2%)
- **Artifacts:** Files produced by the run (equity curve plots, trained model files)

**How it's used in this system:**

```
with tracker.run("sma_20_50_test"):
    tracker.log_params({
        "strategy": "sma_crossover",
        "short_window": 20,
        "long_window": 50,
        "sizing": "fixed_fraction",
        "fraction": 0.10,
    })
    # ... run backtest ...
    tracker.log_metrics({
        "sharpe_ratio": 0.68,
        "max_drawdown_pct": -12.5,
        "total_return_pct": 8.2,
        "total_trades": 34,
    })
```

After many runs, you can query MLflow to compare results: "Show me all runs with Sharpe > 0.5 sorted by max drawdown." This prevents you from repeating experiments and helps identify which parameter choices matter most.

---

## 17. Configuration: Controlling Everything From One File

**Location:** `src/config.py`

Every aspect of the system is controlled by a YAML configuration file. This makes it easy to run different experiments without changing code.

### Configuration Sections

**`data`** — Where to get prices:
```yaml
data:
  symbols: ["AAPL", "MSFT", "GOOGL"]     # Which assets to trade
  start_date: "2020-01-01"                 # Backtest start date
  end_date: "2024-12-31"                   # Backtest end date
  data_source: "csv"                       # csv or yfinance
  data_path: "data/prices"                 # Where CSV files live
```

**`execution`** — Capital and trading costs:
```yaml
execution:
  initial_capital: 100000.0                # Starting cash
  commission_pct: 0.001                    # 0.1% commission
  slippage_pct: 0.0005                     # 0.05% slippage
```

**`strategy`** — Which strategy and its parameters:
```yaml
strategy:
  name: "sma_crossover"                   # or "ml_strategy"
  parameters:
    short_window: 10
    long_window: 30
```

**`sizing`** — How to size positions:
```yaml
sizing:
  method: "volatility"                     # fixed_fraction, volatility, or kelly
  parameters:
    risk_fraction: 0.02                    # Risk 2% per trade
    atr_period: 14
```

**`risk`** — Safety limits:
```yaml
risk:
  enabled: true
  max_position_pct: 0.10                   # Max 10% in one position
  max_drawdown_pct: 0.15                   # Halt at 15% drawdown
  max_daily_loss_pct: 0.03                 # Halt at 3% daily loss
  max_open_positions: 20
```

**`optimization`** — Portfolio weight allocation:
```yaml
optimization:
  method: "mean_variance"                  # none, mean_variance, or risk_parity
  rebalance_frequency: 20                  # Rebalance every 20 bars
```

**`features`** — Which features to compute for ML:
```yaml
features:
  technical:
    - name: "sma"
      windows: [5, 10, 20, 50]
    - name: "rsi"
      period: 14
    - name: "macd"
    - name: "bollinger"
    - name: "atr"
  statistical:
    - name: "zscore"
    - name: "skewness"
    - name: "kurtosis"
    - name: "hurst"
  target:
    horizon: 5                             # Predict 5-day forward return
```

**`ml`** — Machine learning model:
```yaml
ml:
  model: "xgboost"                         # xgboost or lightgbm
  mode: "classification"                   # classification or regression
  parameters:
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
```

**`validation`** — How to validate results:
```yaml
validation:
  method: "cpcv"
  n_splits: 6
  n_test_splits: 2
  purge_window: 5
  embargo_pct: 0.01
```

**`regime`** — Regime detection:
```yaml
regime:
  enabled: true
  n_regimes: 3                             # Bull, bear, sideways
  vol_window: 20
```

**`tracking`** — Experiment logging:
```yaml
tracking:
  enabled: true
  experiment_name: "trading_system"
```

All sections except `data`, `execution`, and `strategy` have defaults, so a minimal config file only needs those three sections and everything else uses sensible defaults.

---

## 18. How It All Fits Together

Here is the complete flow for a single backtest run, showing how every component interacts:

### Setup Phase

1. **Load config** from YAML file → typed `BacktestConfig` object
2. **Build DataHandler:** load CSV files or download from Yahoo Finance
3. **Build Strategy:** SMA crossover, multi-asset SMA, or ML strategy
4. **Build PositionSizer:** fixed fraction, volatility-based, or Kelly
5. **Build Portfolio:** with initial capital, symbols, and the chosen sizer
6. **Build RiskManager:** with configured limits
7. **Build ExecutionHandler:** with commission and slippage percentages
8. **Build PerformanceTracker:** with initial capital
9. **(If ML):** Build FeaturePipeline, train ML model on historical data
10. **(If regime):** Fit RegimeDetector on historical data
11. **Create BacktestEngine:** wire all components together

### Simulation Phase (repeats for each bar)

```
┌──────────────────────────────────────────────────────────────────┐
│  Engine: data_handler.update_bars() → advance to next day       │
│  Engine: create MarketEvent → put on queue                      │
│                                                                  │
│  ┌─ Process Queue ──────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  MarketEvent → Strategy.on_market_data()                  │   │
│  │    → Strategy pulls bars, computes indicators/ML features │   │
│  │    → If signal detected: put SignalEvent on queue         │   │
│  │                                                           │   │
│  │  SignalEvent → Portfolio.on_signal()                       │   │
│  │    → PositionSizer.calculate_size()                       │   │
│  │    → If valid order: put OrderEvent on queue              │   │
│  │                                                           │   │
│  │  OrderEvent → RiskManager.check_order()                   │   │
│  │    → 7 risk checks (circuit breaker, daily loss, etc.)    │   │
│  │    → If approved: ExecutionHandler.execute_order()         │   │
│  │    → If rejected: log and discard                         │   │
│  │                                                           │   │
│  │  ExecutionHandler → apply slippage + commission            │   │
│  │    → Put FillEvent on queue                               │   │
│  │                                                           │   │
│  │  FillEvent → Portfolio.on_fill()                           │   │
│  │    → Update cash, positions, P&L                          │   │
│  │    → Record trade in history                              │   │
│  │    → Strategy.update_position() (sync holdings)           │   │
│  │    → RiskManager.update_daily_pnl()                       │   │
│  │                                                           │   │
│  └── Queue empty → continue to next bar ─────────────────────┘   │
│                                                                  │
│  Portfolio.update_timeindex() → recalculate all market values    │
│  RiskManager.update_peak_equity() → track drawdown               │
│  PerformanceTracker.update() → record equity for this bar        │
└──────────────────────────────────────────────────────────────────┘
```

### Analysis Phase (after simulation completes)

1. **Calculate performance metrics:** Sharpe, Sortino, Calmar, drawdown, win rate, etc.
2. **Print report:** formatted summary of all metrics
3. **(If CPCV):** Run combinatorial purged cross-validation for robust out-of-sample estimates
4. **(If DSR):** Compute deflated Sharpe ratio to adjust for multiple testing
5. **(If tracking):** Log all parameters and metrics to MLflow
6. **Generate charts:** equity curve, drawdown chart, trade annotations

### What You Get Out

A dictionary of results:
```python
{
    "bars_processed": 1258,              # ~5 years of daily data
    "events_processed": 4832,            # All events handled
    "final_equity": 115230.50,           # Portfolio end value
    "positions": {"AAPL": {...}},        # Final positions
    "trade_history": [Trade(...), ...],  # Every trade made
    "rejected_orders": 3,               # Orders blocked by risk manager
    "metrics": {
        "total_return_pct": 15.23,
        "annualized_return_pct": 2.88,
        "sharpe_ratio": 0.68,
        "sortino_ratio": 0.95,
        "max_drawdown_pct": -12.45,
        "volatility_pct": 14.32,
        "calmar_ratio": 0.23,
    },
    "equity_curve": DataFrame(...)       # Daily equity values
}
```

This tells you everything you need to know about whether the strategy is worth pursuing further. If the Sharpe is positive, the drawdown is tolerable, and the CPCV/DSR validation confirms the results aren't just luck — you have a candidate strategy worth paper trading.

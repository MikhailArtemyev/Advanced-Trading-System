# Phase 5: Live Broker Integration & Production Readiness
## Step-by-Step Implementation Guide

**Duration:** 8 Weeks (Months 10-11)
**Goal:** Connect the paper trading engine to a real broker (Alpaca) for live-paper and live-real execution, add a monitoring dashboard, database-backed trade journal, alerting system, and advanced strategy templates

**Prerequisites:** Phase 4 complete — working paper trading engine with LiveDataHandler, PaperBroker, OrderManager, StateManager, HealthMonitor, and 1349+ tests

---

## Overview

```
Week 1:   Alpaca Broker Adapter (REST + WebSocket, live-paper mode)
Week 2:   Real-Time Data Feed (Alpaca market data, polygon fallback)
Week 3:   Database Trade Journal (SQLite, trade/order/snapshot storage)
Week 4:   Alerting & Notifications (Slack, email, webhook alerts)
Week 5:   Strategy Templates (momentum, mean reversion, pairs trading)
Week 6:   Monitoring Dashboard (terminal UI, real-time metrics)
Week 7:   Deployment & Configuration (Docker, environment management, secrets)
Week 8:   End-to-End Testing, Soak Testing & Documentation
```

---

## What Changes From Phase 4

Phase 4 established the paper trading infrastructure with simulated execution. Phase 5 transitions to real broker connectivity, persistent storage, production alerting, and deployment readiness:

| Component | Phase 4 State | Phase 5 Target |
|-----------|--------------|----------------|
| Broker | PaperBroker only (simulated fills) | Alpaca adapter (live-paper + live-real), PaperBroker preserved |
| Data Feed | Synthetic ticks / WebSocket ABC | Alpaca real-time market data + polygon fallback |
| Storage | JSON snapshots (StateManager) | SQLite database for trades, orders, snapshots, performance |
| Alerts | HealthMonitor callbacks (in-process only) | Slack, email, and webhook notifications |
| Strategies | SMA crossover only | + Momentum, mean reversion, pairs trading templates |
| Dashboard | Log output only | Terminal UI with real-time equity, positions, health |
| Deployment | Local execution only | Docker container, env-based config, secret management |
| Testing | 1349 tests | Target 1600+ with broker, DB, alert, and strategy coverage |

### New Project Structure (additions to Phase 4)

```
src/
├── ... (existing Phase 1-4 modules)
├── broker/
│   ├── ... (existing)
│   └── alpaca_broker.py           # Alpaca API broker adapter
├── live/
│   ├── ... (existing)
│   └── alpaca_feed.py             # Alpaca real-time market data feed
├── storage/
│   ├── __init__.py
│   ├── database.py                # SQLite database manager
│   ├── trade_journal.py           # Trade/order persistence
│   └── models.py                  # SQLAlchemy models (trades, orders, snapshots)
├── alerts/
│   ├── __init__.py
│   ├── base_alert.py              # AlertChannel ABC
│   ├── slack_alert.py             # Slack webhook notifications
│   ├── email_alert.py             # SMTP email notifications
│   └── webhook_alert.py           # Generic HTTP webhook
├── strategy/
│   ├── ... (existing)
│   ├── momentum.py                # Momentum strategy (cross-sectional + time-series)
│   ├── mean_reversion.py          # Mean reversion (z-score based)
│   └── pairs_trading.py           # Statistical pairs trading (cointegration)
├── dashboard/
│   ├── __init__.py
│   └── terminal_ui.py             # Rich-based terminal dashboard
tests/
├── ... (existing Phase 1-4 tests)
├── test_alpaca_broker.py
├── test_alpaca_feed.py
├── test_database.py
├── test_trade_journal.py
├── test_alerts.py
├── test_momentum_strategy.py
├── test_mean_reversion.py
├── test_pairs_trading.py
├── test_dashboard.py
├── test_phase5_integration.py
configs/
├── ... (existing configs)
├── live_alpaca_config.yaml        # Alpaca paper trading
└── pairs_trading_config.yaml      # Pairs trading strategy
```

---

## Week 1: Alpaca Broker Adapter

### Step 1.1: Alpaca Broker
**Time:** Day 1-3

**File: `src/broker/alpaca_broker.py`**

Implements `BrokerAdapter` ABC using the Alpaca Trade API:

**Class: `AlpacaBroker(BrokerAdapter)`**

Constructor Args:
- `api_key: str` — Alpaca API key
- `api_secret: str` — Alpaca secret key
- `base_url: str` — API base URL (paper: `https://paper-api.alpaca.markets`, live: `https://api.alpaca.markets`)
- `paper_mode: bool = True` — Whether using paper trading endpoint

Methods (all async, implementing BrokerAdapter ABC):
- `connect()` — Initialize Alpaca REST client, verify credentials with `get_account()`
- `disconnect()` — Close HTTP session
- `submit_order(order)` — Submit via `POST /v2/orders`, map BrokerOrder → Alpaca order params, return updated BrokerOrder with broker_order_id
- `cancel_order(order_id)` — Cancel via `DELETE /v2/orders/{id}`
- `get_order_status(order_id)` — Query `GET /v2/orders/{id}`, map Alpaca status → OrderStatus enum
- `get_positions()` — Query `GET /v2/positions`, return dict of positions
- `get_account_info()` — Query `GET /v2/account`, return cash/equity/buying_power

Private helpers:
- `_map_order_type(order_type)` — Map internal order type to Alpaca format
- `_map_order_status(alpaca_status)` — Map Alpaca status string to OrderStatus enum
- `_build_headers()` — API key authentication headers

**Tests (`tests/test_alpaca_broker.py`):** ~25 tests
- Connection with valid/invalid credentials (mocked HTTP)
- Order submission (market, limit)
- Order cancellation
- Position retrieval
- Account info
- Status mapping (all Alpaca statuses → OrderStatus)
- Error handling (network errors, API errors, rate limits)

### Step 1.2: Config Extension
**Time:** Day 4

**File: `src/config.py`** (modified)

Add `alpaca` as a valid broker_type. Add optional fields:
- `BrokerConfig.paper_mode: bool = True`

**File: `scripts/run_paper_trading.py`** (modified)

Add broker builder that selects PaperBroker or AlpacaBroker based on config:
```python
if config.live.broker.broker_type == "alpaca":
    broker = AlpacaBroker(
        api_key=config.live.broker.api_key,
        api_secret=config.live.broker.api_secret,
        base_url=config.live.broker.base_url,
        paper_mode=config.live.broker.paper_mode,
    )
```

### Step 1.3: Dependencies
**Time:** Day 5

Add to `requirements.txt`:
```
alpaca-trade-api>=3.0.0
```

**Target: ~25 tests**

---

## Week 2: Real-Time Market Data Feed

### Step 2.1: Alpaca Data Feed
**Time:** Day 1-3

**File: `src/live/alpaca_feed.py`**

Implements `LiveDataFeed` ABC using Alpaca's real-time market data WebSocket:

**Class: `AlpacaDataFeed(LiveDataFeed)`**

Constructor Args:
- `api_key: str`
- `api_secret: str`
- `feed: str = "iex"` — Data feed (`iex` for free, `sip` for premium)
- `base_url: str = "wss://stream.data.alpaca.markets/v2"`

Methods:
- `async connect()` — Open WebSocket, authenticate, start message loop
- `async disconnect()` — Close WebSocket cleanly
- `async subscribe(symbols)` — Send subscribe message for trade/quote updates
- `async unsubscribe(symbols)` — Send unsubscribe message
- `_on_message(msg)` — Parse Alpaca message, create Tick, notify listeners

Message types handled:
- `t` (trade) → Tick with price, volume
- `q` (quote) → Tick with bid/ask
- `b` (bar) → Completed bar notification

### Step 2.2: Data Feed Factory
**Time:** Day 4-5

**File: `scripts/run_paper_trading.py`** (modified)

Add data feed builder:
```python
if config.live.data.feed_type == "alpaca":
    feed = AlpacaDataFeed(
        api_key=config.live.broker.api_key,
        api_secret=config.live.broker.api_secret,
    )
elif config.live.data.feed_type == "websocket":
    feed = WebSocketDataFeed(url=config.live.data.url)
```

Wire feed → BarAggregator → LiveDataHandler.

**Tests (`tests/test_alpaca_feed.py`):** ~20 tests
- Connection and authentication (mocked WebSocket)
- Subscribe/unsubscribe
- Trade message → Tick conversion
- Quote message → Tick conversion
- Reconnection on disconnect
- Error handling

**Target: ~20 tests**

---

## Week 3: Database Trade Journal

### Step 3.1: SQLAlchemy Models
**Time:** Day 1-2

**File: `src/storage/models.py`**

SQLAlchemy ORM models:

```python
class TradeRecord(Base):
    __tablename__ = "trades"
    id: int (PK, autoincrement)
    timestamp: datetime
    symbol: str
    side: str           # BUY or SELL
    quantity: int
    price: float
    commission: float
    pnl: float
    session_id: str     # links trades to a session
    strategy: str

class OrderRecord(Base):
    __tablename__ = "orders"
    id: int (PK, autoincrement)
    order_id: str (unique)
    symbol: str
    side: str
    order_type: str
    quantity: int
    status: str
    submitted_at: datetime
    filled_at: datetime | None
    filled_quantity: int
    filled_avg_price: float
    session_id: str

class SnapshotRecord(Base):
    __tablename__ = "snapshots"
    id: int (PK, autoincrement)
    timestamp: datetime
    session_id: str
    equity: float
    cash: float
    positions_json: str  # JSON blob
    statistics_json: str # JSON blob

class SessionRecord(Base):
    __tablename__ = "sessions"
    id: str (PK)         # UUID
    started_at: datetime
    ended_at: datetime | None
    strategy: str
    config_json: str     # full config as JSON
    final_equity: float | None
    total_trades: int
    status: str          # running, completed, crashed
```

### Step 3.2: Database Manager
**Time:** Day 2-3

**File: `src/storage/database.py`**

**Class: `DatabaseManager`**

Constructor Args:
- `db_url: str = "sqlite:///trading.db"` — SQLAlchemy connection URL

Methods:
- `initialize()` — Create tables if not exist
- `get_session()` — Return SQLAlchemy session (context manager)
- `close()` — Close engine

### Step 3.3: Trade Journal
**Time:** Day 3-5

**File: `src/storage/trade_journal.py`**

**Class: `TradeJournal`**

Constructor Args:
- `db: DatabaseManager`
- `session_id: str`

Methods:
- `record_trade(trade)` — Insert TradeRecord
- `record_order(order)` — Insert/update OrderRecord
- `record_snapshot(engine_state)` — Insert SnapshotRecord
- `start_session(strategy, config)` — Create SessionRecord
- `end_session(final_equity, total_trades)` — Update SessionRecord
- `get_trades(symbol=None, start=None, end=None)` — Query trades with filters
- `get_orders(status=None)` — Query orders
- `get_equity_curve(session_id)` — Query snapshots → equity DataFrame
- `get_session_summary(session_id)` — Aggregate stats for a session
- `get_all_sessions()` — List all sessions with summary stats

**Tests (`tests/test_database.py`, `tests/test_trade_journal.py`):** ~30 tests
- Table creation
- Trade insert and query
- Order insert, update, and query
- Snapshot insert and equity curve retrieval
- Session lifecycle (start → end)
- Filtering (by symbol, date range, status)
- Multiple sessions
- Edge cases (empty DB, duplicate order_id)

**Target: ~30 tests**

---

## Week 4: Alerting & Notifications

### Step 4.1: Alert Channel ABC
**Time:** Day 1

**File: `src/alerts/base_alert.py`**

```python
class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class AlertMessage:
    level: AlertLevel
    title: str
    body: str
    timestamp: datetime
    metadata: dict[str, Any]

class AlertChannel(ABC):
    @abstractmethod
    async def send(self, message: AlertMessage) -> bool: ...

    @abstractmethod
    async def test_connection(self) -> bool: ...
```

### Step 4.2: Alert Implementations
**Time:** Day 2-3

**File: `src/alerts/slack_alert.py`**

**Class: `SlackAlert(AlertChannel)`**
- Constructor: `webhook_url: str, channel: str = ""`
- `send()` — POST to Slack webhook with formatted message blocks
- `test_connection()` — Send test message

**File: `src/alerts/email_alert.py`**

**Class: `EmailAlert(AlertChannel)`**
- Constructor: `smtp_host, smtp_port, username, password, to_addresses`
- `send()` — Send email via SMTP (async with aiosmtplib)
- `test_connection()` — Verify SMTP connection

**File: `src/alerts/webhook_alert.py`**

**Class: `WebhookAlert(AlertChannel)`**
- Constructor: `url: str, headers: dict = {}, method: str = "POST"`
- `send()` — POST/PUT JSON payload to URL
- `test_connection()` — Send health check request

### Step 4.3: Alert Manager
**Time:** Day 4-5

**File: `src/alerts/alert_manager.py`**

**Class: `AlertManager`**

Constructor Args:
- `channels: list[AlertChannel]`
- `min_level: AlertLevel = AlertLevel.WARNING`
- `cooldown_seconds: float = 300.0` — Min time between repeated alerts

Methods:
- `async send_alert(message)` — Send to all channels (respects min_level and cooldown)
- `async send_trade_alert(trade)` — Format and send trade notification
- `async send_health_alert(report)` — Format and send health status change
- `async send_drawdown_alert(drawdown_pct)` — Alert on significant drawdown

Integration with PaperTradingEngine:
- Register AlertManager as a HealthMonitor callback
- Hook into fill events for trade notifications
- Hook into risk manager for drawdown alerts

**Config extension (`src/config.py`):**

```python
class AlertConfig(BaseModel):
    enabled: bool = False
    channels: list[dict[str, Any]] = Field(default_factory=list)
    min_level: str = "warning"
    cooldown_seconds: float = 300.0
```

**Tests (`tests/test_alerts.py`):** ~25 tests
- AlertLevel enum
- AlertMessage creation
- SlackAlert send (mocked HTTP)
- EmailAlert send (mocked SMTP)
- WebhookAlert send (mocked HTTP)
- AlertManager routing (min_level filtering, cooldown)
- Trade/health/drawdown alert formatting
- Multiple channels
- Connection test

**Target: ~25 tests**

---

## Week 5: Strategy Templates

### Step 5.1: Momentum Strategy
**Time:** Day 1-2

**File: `src/strategy/momentum.py`**

**Class: `MomentumStrategy(Strategy)`**

Cross-sectional momentum: rank symbols by trailing return, go long top N, short bottom N.

Parameters:
- `lookback_period: int = 20` — Return lookback window
- `top_n: int = 3` — Number of long positions
- `bottom_n: int = 0` — Number of short positions (0 = long-only)
- `rebalance_frequency: int = 5` — Bars between rebalances
- `min_bars: int = 30` — Minimum bars before trading

### Step 5.2: Mean Reversion Strategy
**Time:** Day 2-3

**File: `src/strategy/mean_reversion.py`**

**Class: `MeanReversionStrategy(Strategy)`**

Z-score based mean reversion: enter when price deviates from mean, exit when it reverts.

Parameters:
- `lookback_period: int = 20` — Z-score window
- `entry_threshold: float = 2.0` — Z-score to enter (long below -2, short above +2)
- `exit_threshold: float = 0.5` — Z-score to exit (near mean)
- `max_holding_period: int = 10` — Force exit after N bars

### Step 5.3: Pairs Trading Strategy
**Time:** Day 3-5

**File: `src/strategy/pairs_trading.py`**

**Class: `PairsTradingStrategy(Strategy)`**

Statistical arbitrage: find cointegrated pairs, trade the spread.

Parameters:
- `pair: tuple[str, str]` — Symbol pair to trade
- `lookback_period: int = 60` — Cointegration/spread window
- `entry_threshold: float = 2.0` — Spread z-score to enter
- `exit_threshold: float = 0.5` — Spread z-score to exit
- `hedge_ratio_method: str = "ols"` — OLS or Kalman filter

Methods:
- `_calculate_spread(bars_a, bars_b)` — Compute spread and z-score
- `_check_cointegration(bars_a, bars_b)` — Engle-Granger test
- `_calculate_hedge_ratio(bars_a, bars_b)` — OLS regression

Register new strategies in `scripts/run_backtest.py` `build_strategy()`:
- `"momentum"` → `MomentumStrategy`
- `"mean_reversion"` → `MeanReversionStrategy`
- `"pairs_trading"` → `PairsTradingStrategy`

Add to `StrategyConfig.validate_name()` in `src/config.py`.

**Tests:** ~35 tests
- `tests/test_momentum_strategy.py` — ~12 tests: ranking, signal generation, rebalance timing, long-only/long-short
- `tests/test_mean_reversion.py` — ~12 tests: z-score entry/exit, max holding period, both directions
- `tests/test_pairs_trading.py` — ~11 tests: spread calculation, cointegration check, hedge ratio, signal generation

**Target: ~35 tests**

---

## Week 6: Monitoring Dashboard

### Step 6.1: Terminal Dashboard
**Time:** Day 1-4

**File: `src/dashboard/terminal_ui.py`**

Rich-based terminal UI showing real-time trading session status.

**Dependencies:** `rich>=13.0.0`

**Class: `TradingDashboard`**

Constructor Args:
- `engine: PaperTradingEngine`
- `health_monitor: HealthMonitor`
- `refresh_rate: float = 1.0` — Seconds between refreshes

Layout panels:
- **Header** — Session info (strategy, symbols, start time)
- **Equity** — Current equity, P&L, return %, max drawdown
- **Positions** — Table: symbol, quantity, avg_cost, market_value, unrealized P&L
- **Recent Trades** — Last 10 trades with timestamps
- **Health** — Status indicator, bar age, latency, fill rate
- **Statistics** — Bars processed, events, orders submitted/rejected/filled

Methods:
- `async start()` — Begin live-updating dashboard (runs in asyncio loop)
- `stop()` — Stop dashboard
- `_build_layout()` — Compose Rich Layout with panels
- `_update()` — Refresh all panels from engine/monitor state

### Step 6.2: Integration
**Time:** Day 5

Wire dashboard into `scripts/run_paper_trading.py`:
- Add `--dashboard` flag to enable terminal UI
- Dashboard runs as another task in `asyncio.gather()`
- Graceful cleanup on shutdown

Add `make paper-dashboard` target to Makefile.

**Tests (`tests/test_dashboard.py`):** ~10 tests
- Dashboard creation
- Layout building
- Panel content generation from mock engine state
- Start/stop lifecycle
- Handles missing data gracefully

**Target: ~10 tests**

---

## Week 7: Deployment & Configuration

### Step 7.1: Environment-Based Configuration
**Time:** Day 1-2

**File: `src/config.py`** (modified)

Add environment variable support for secrets:
- `ALPACA_API_KEY`, `ALPACA_API_SECRET`
- `SLACK_WEBHOOK_URL`
- `SMTP_PASSWORD`
- `DATABASE_URL`

Config loader checks env vars when YAML values are empty:
```python
api_key = config.live.broker.api_key or os.environ.get("ALPACA_API_KEY", "")
```

### Step 7.2: Docker Setup
**Time:** Day 2-3

**File: `Dockerfile`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "scripts/run_paper_trading.py", "--config", "configs/live_alpaca_config.yaml"]
```

**File: `docker-compose.yml`**

```yaml
services:
  paper-trader:
    build: .
    env_file: .env
    volumes:
      - ./state:/app/state
      - ./data:/app/data
    restart: unless-stopped
```

**File: `.env.example`**

```
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
SLACK_WEBHOOK_URL=
DATABASE_URL=sqlite:///trading.db
```

### Step 7.3: Startup Validation
**Time:** Day 4-5

**File: `src/config.py`** (modified)

Add `validate_for_live()` method to `BacktestConfig`:
- Verify API credentials are set when broker_type is not "paper"
- Verify symbols are valid
- Verify alert channels are reachable (optional)
- Return list of warnings/errors

**Tests:** ~10 tests
- Env var loading
- Missing credential detection
- Validation warnings
- Docker config verification (file exists, syntax correct)

**Target: ~10 tests**

---

## Week 8: End-to-End Testing & Documentation

### Step 8.1: Integration Tests
**Time:** Day 1-3

**File: `tests/test_phase5_integration.py`**

End-to-end tests:
1. Full pipeline with Alpaca broker (mocked API)
2. Trade journal records from paper trading session
3. Alert firing on health degradation
4. Strategy switching (SMA → momentum → mean reversion)
5. Database persistence across sessions
6. Dashboard renders with live engine data
7. Config validation catches missing credentials
8. Graceful shutdown saves final state to DB

**Target: ~25 integration tests**

### Step 8.2: Documentation
**Time:** Day 4-5

- Weekly notes: `PLANNING/NOTES-Phase5-Week{1-8}.md`
- Update `README.md` with Phase 5 components, configs, strategies
- Update `CLAUDE.md` with new packages and commands
- Update project status table

---

## Test Count Summary

| Week | Component | New Tests |
|------|-----------|-----------|
| 1 | Alpaca Broker | ~25 |
| 2 | Alpaca Data Feed | ~20 |
| 3 | Database & Trade Journal | ~30 |
| 4 | Alerting & Notifications | ~25 |
| 5 | Strategy Templates | ~35 |
| 6 | Monitoring Dashboard | ~10 |
| 7 | Deployment & Config | ~10 |
| 8 | Integration Tests | ~25 |
| **Total** | | **~180** |
| **Cumulative (with Phase 4)** | | **~1529** |

---

## Success Metrics

- All `make check` passes (black, ruff, mypy, pytest)
- 1500+ tests, all green
- Alpaca broker adapter connects in paper mode (with valid credentials)
- Trades persisted to SQLite with full audit trail
- Alerts fire on health degradation and significant drawdowns
- Three new strategy templates work in both backtest and paper trading modes
- Terminal dashboard shows live updating equity, positions, and health
- Docker container runs paper trading session end-to-end
- All Phase 1-4 configs and tests still work unchanged

---

## Dependencies Added

```
# Broker
alpaca-trade-api>=3.0.0

# Dashboard
rich>=13.0.0

# Alerts
aiosmtplib>=2.0.0

# Already present from Phase 1-4
sqlalchemy>=2.0.0
aiohttp>=3.9.0
```

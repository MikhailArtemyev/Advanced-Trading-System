# Phase 4: Paper Trading & Live Data Integration
## Step-by-Step Implementation Guide

**Duration:** 8 Weeks (Months 8-9)
**Goal:** Connect the backtesting engine to live market data, implement broker API integration in paper mode, and run strategies in real-time with simulated execution

**Prerequisites:** Phase 3 complete — working ML-enabled backtester with feature engineering, regime detection, CPCV validation, experiment tracking, and 997+ tests

---

## Overview

```
Week 1-2: Async Infrastructure & Live Data Handler (WebSocket, bar aggregation)
Week 3-4: Broker Adapter & Paper Execution (abstract broker interface, simulated fills)
Week 5:   Order Manager & Order Lifecycle (state machine, order tracking)
Week 6:   Paper Trading Engine (real-time event loop, strategy integration)
Week 7:   Reconciliation, State Persistence & Monitoring
Week 8:   Integration Testing, Stability Testing & Documentation
```

---

## What Changes From Phase 3

Phase 3 established ML-driven signal generation with rigorous validation. Phase 4 transitions from historical replay to real-time operation with live market data and simulated execution:

| Component | Phase 3 State | Phase 4 Target |
|-----------|--------------|----------------|
| Data | Historical CSV/yfinance download | Live WebSocket feeds + bar aggregation |
| Execution | Simulated fills from historical bars | Paper broker with realistic fill simulation |
| Engine | Synchronous `BacktestEngine.run()` loop | Async `PaperTradingEngine` with continuous event loop |
| Orders | Fire-and-forget (instant fill) | Full lifecycle: submitted → accepted → filled/cancelled |
| State | In-memory only, lost on exit | Persistent state with crash recovery |
| Monitoring | Post-hoc metrics only | Real-time health checks, latency tracking |
| Config | Backtest-only YAML | + `live` section for broker, data feed, persistence settings |
| Testing | 997 tests | Target 1200+ with async, broker, and integration coverage |

### New Project Structure (additions to Phase 3)

```
src/
├── ... (existing Phase 1-3 modules)
├── live/
│   ├── __init__.py
│   ├── data_feed.py              # LiveDataFeed ABC + WebSocket implementation
│   ├── bar_aggregator.py         # Tick/quote → OHLCV bar aggregation
│   └── data_handler.py           # LiveDataHandler (implements DataHandler ABC)
├── broker/
│   ├── __init__.py
│   ├── base_broker.py            # BrokerAdapter ABC
│   ├── paper_broker.py           # Paper trading broker (simulated fills)
│   └── order_manager.py          # Order lifecycle state machine
├── engine/
│   ├── __init__.py
│   ├── paper_engine.py           # PaperTradingEngine (async event loop)
│   └── state_manager.py          # State persistence & crash recovery
├── monitoring/
│   ├── __init__.py
│   └── health.py                 # Health checks, latency tracking, alerting
tests/
├── ... (existing Phase 1-3 tests)
├── test_data_feed.py
├── test_bar_aggregator.py
├── test_live_data_handler.py
├── test_paper_broker.py
├── test_order_manager.py
├── test_paper_engine.py
├── test_state_manager.py
├── test_monitoring.py
├── test_phase4_integration.py
configs/
├── ... (existing configs)
├── paper_trading_config.yaml     # Paper trading configuration
└── paper_ml_config.yaml          # Paper trading with ML strategy
```

---

## Week 1: Async Infrastructure & Live Data Feed

### Step 1.1: Live Data Feed ABC & WebSocket Implementation
**Time:** Day 1-3

**File: `src/live/__init__.py`**
```python
from .bar_aggregator import BarAggregator
from .data_feed import LiveDataFeed, WebSocketDataFeed
from .data_handler import LiveDataHandler

__all__ = [
    "BarAggregator",
    "LiveDataFeed",
    "WebSocketDataFeed",
    "LiveDataHandler",
]
```

**File: `src/live/data_feed.py`**
```python
"""Live market data feed abstraction.

Provides an ABC for streaming market data and a WebSocket implementation.
Data feeds emit raw ticks/quotes that the BarAggregator converts to OHLCV bars.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class FeedStatus(Enum):
    """Connection status of a data feed."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()


@dataclass(frozen=True)
class Tick:
    """A single price update from the market.

    Attributes:
        symbol: Instrument identifier.
        timestamp: Exchange timestamp of the tick.
        price: Last trade price.
        volume: Trade volume (0 for quotes without volume).
        bid: Best bid price (None if unavailable).
        ask: Best ask price (None if unavailable).
    """

    symbol: str
    timestamp: datetime
    price: float
    volume: float = 0.0
    bid: float | None = None
    ask: float | None = None


class LiveDataFeed(ABC):
    """Abstract base class for live market data feeds.

    Subclasses must implement connect/disconnect and subscribe/unsubscribe.
    When ticks arrive, subclasses call _on_tick() to notify listeners.
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[Tick], None]] = []
        self._status: FeedStatus = FeedStatus.DISCONNECTED
        self._symbols: set[str] = set()

    @property
    def status(self) -> FeedStatus:
        return self._status

    @property
    def subscribed_symbols(self) -> set[str]:
        return self._symbols.copy()

    def add_listener(self, callback: Callable[[Tick], None]) -> None:
        """Register a callback invoked on each tick."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[Tick], None]) -> None:
        """Unregister a tick callback."""
        self._listeners.remove(callback)

    def _on_tick(self, tick: Tick) -> None:
        """Dispatch a tick to all registered listeners."""
        for listener in self._listeners:
            listener(tick)

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        ...

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to live data for the given symbols."""
        ...

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from live data for the given symbols."""
        ...


class WebSocketDataFeed(LiveDataFeed):
    """WebSocket-based live data feed.

    Connects to a WebSocket endpoint, subscribes to symbols,
    and parses incoming messages into Tick objects. Includes
    automatic reconnection with exponential backoff.

    Constructor Args:
        url: WebSocket server URL.
        parse_message: Callable that converts a raw WebSocket message
            (str or bytes) into a list of Tick objects. Allows plugging
            in exchange-specific message formats.
        max_reconnect_attempts: Max consecutive reconnection attempts
            before giving up (0 = unlimited).
        initial_reconnect_delay: Starting delay in seconds for
            exponential backoff reconnection.
    """

    def __init__(
        self,
        url: str,
        parse_message: Callable[[str | bytes], list[Tick]],
        max_reconnect_attempts: int = 10,
        initial_reconnect_delay: float = 1.0,
    ) -> None:
        super().__init__()
        self._url = url
        self._parse_message = parse_message
        self._max_reconnect_attempts = max_reconnect_attempts
        self._initial_reconnect_delay = initial_reconnect_delay
        self._ws: Any = None  # aiohttp.ClientWebSocketResponse
        self._session: Any = None  # aiohttp.ClientSession
        self._listen_task: asyncio.Task[None] | None = None
        self._reconnect_count = 0
        self._should_run = False

    async def connect(self) -> None:
        """Connect to the WebSocket server and start listening."""
        import aiohttp

        self._should_run = True
        self._status = FeedStatus.CONNECTING
        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(self._url)
            self._status = FeedStatus.CONNECTED
            self._reconnect_count = 0
            self._listen_task = asyncio.create_task(self._listen_loop())
            logger.info("WebSocket connected to %s", self._url)
        except Exception:
            self._status = FeedStatus.ERROR
            logger.exception("Failed to connect to %s", self._url)
            raise

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._should_run = False
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        self._status = FeedStatus.DISCONNECTED
        logger.info("WebSocket disconnected from %s", self._url)

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols via WebSocket.

        Sends a JSON subscribe message. Override _build_subscribe_message()
        for exchange-specific formats.
        """
        import json

        self._symbols.update(symbols)
        if self._ws and not self._ws.closed:
            msg = json.dumps(
                {"action": "subscribe", "symbols": symbols}
            )
            await self._ws.send_str(msg)
            logger.info("Subscribed to %s", symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols via WebSocket."""
        import json

        self._symbols -= set(symbols)
        if self._ws and not self._ws.closed:
            msg = json.dumps(
                {"action": "unsubscribe", "symbols": symbols}
            )
            await self._ws.send_str(msg)
            logger.info("Unsubscribed from %s", symbols)

    async def _listen_loop(self) -> None:
        """Read messages from WebSocket and dispatch ticks."""
        import aiohttp

        while self._should_run:
            try:
                if self._ws is None or self._ws.closed:
                    await self._reconnect()
                    continue

                msg = await self._ws.receive(timeout=30.0)

                if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    ticks = self._parse_message(msg.data)
                    for tick in ticks:
                        self._on_tick(tick)
                elif msg.type == aiohttp.WSMsgType.PING:
                    await self._ws.pong()
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    logger.warning("WebSocket closed by server")
                    await self._reconnect()
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error: %s", self._ws.exception())
                    await self._reconnect()

            except asyncio.TimeoutError:
                # No message in 30s — send ping or reconnect
                logger.debug("WebSocket receive timeout, continuing")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in WebSocket listen loop")
                await self._reconnect()

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        if not self._should_run:
            return

        self._status = FeedStatus.RECONNECTING

        if (
            self._max_reconnect_attempts > 0
            and self._reconnect_count >= self._max_reconnect_attempts
        ):
            self._status = FeedStatus.ERROR
            logger.error(
                "Max reconnect attempts (%d) reached",
                self._max_reconnect_attempts,
            )
            return

        delay = self._initial_reconnect_delay * (2 ** self._reconnect_count)
        delay = min(delay, 60.0)  # cap at 60 seconds
        self._reconnect_count += 1
        logger.info(
            "Reconnecting in %.1fs (attempt %d)",
            delay,
            self._reconnect_count,
        )
        await asyncio.sleep(delay)

        try:
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._session and not self._session.closed:
                self._ws = await self._session.ws_connect(self._url)
            else:
                import aiohttp
                self._session = aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(self._url)

            self._status = FeedStatus.CONNECTED
            self._reconnect_count = 0

            # Re-subscribe to all symbols
            if self._symbols:
                await self.subscribe(list(self._symbols))

            logger.info("Reconnected successfully")
        except Exception:
            logger.exception("Reconnection failed")
            self._status = FeedStatus.ERROR
```

**Tests to write (`tests/test_data_feed.py`):**
- `TestTick`: frozen dataclass, default values, optional bid/ask
- `TestFeedStatus`: enum values
- `TestLiveDataFeed`: listener add/remove, `_on_tick` dispatches to all listeners
- `TestWebSocketDataFeed`: connect/disconnect lifecycle, subscribe/unsubscribe sends messages, `_listen_loop` parses ticks and dispatches, reconnection with exponential backoff, max reconnect attempts stops after limit, re-subscribe after reconnect

**Target: ~30 tests, all passing**

---

### Step 1.2: Bar Aggregator
**Time:** Day 3-5

**File: `src/live/bar_aggregator.py`**
```python
"""Aggregates raw ticks into OHLCV bars on a configurable time interval.

The BarAggregator collects ticks and emits completed bars when the
time boundary is crossed. Supports any interval (1s, 1m, 5m, 1h, etc.).
Partial bars are available via current_bar for real-time display.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd

from .data_feed import Tick

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single OHLCV bar.

    Attributes:
        symbol: Instrument identifier.
        timestamp: Bar open time (start of the interval).
        open: Opening price.
        high: Highest price during the bar.
        low: Lowest price during the bar.
        close: Closing price (last tick price).
        volume: Total volume during the bar.
        tick_count: Number of ticks aggregated.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int


class BarAggregator:
    """Aggregates ticks into time-based OHLCV bars.

    Constructor Args:
        interval: Bar duration (e.g. timedelta(minutes=1)).
        on_bar: Callback invoked when a bar completes.
    """

    def __init__(
        self,
        interval: timedelta,
        on_bar: Callable[[Bar], None] | None = None,
    ) -> None:
        if interval.total_seconds() <= 0:
            raise ValueError("interval must be positive")
        self._interval = interval
        self._on_bar = on_bar
        # Current partial bars keyed by symbol
        self._current_bars: dict[str, Bar] = {}
        # Completed bars history
        self._completed_bars: dict[str, list[Bar]] = {}

    @property
    def interval(self) -> timedelta:
        return self._interval

    def on_tick(self, tick: Tick) -> None:
        """Process a tick, updating the current bar or completing one.

        This method is designed to be registered as a LiveDataFeed listener.
        """
        bar_start = self._floor_timestamp(tick.timestamp)

        if tick.symbol not in self._current_bars:
            # First tick for this symbol — start a new bar
            self._current_bars[tick.symbol] = Bar(
                symbol=tick.symbol,
                timestamp=bar_start,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.volume,
                tick_count=1,
            )
            return

        current = self._current_bars[tick.symbol]

        if bar_start > current.timestamp:
            # New bar period — emit the completed bar and start fresh
            self._emit_bar(current)
            self._current_bars[tick.symbol] = Bar(
                symbol=tick.symbol,
                timestamp=bar_start,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.volume,
                tick_count=1,
            )
        else:
            # Same bar period — update OHLCV
            current.high = max(current.high, tick.price)
            current.low = min(current.low, tick.price)
            current.close = tick.price
            current.volume += tick.volume
            current.tick_count += 1

    def get_current_bar(self, symbol: str) -> Bar | None:
        """Return the in-progress (partial) bar for a symbol."""
        return self._current_bars.get(symbol)

    def get_completed_bars(self, symbol: str, n: int | None = None) -> list[Bar]:
        """Return the last n completed bars for a symbol."""
        bars = self._completed_bars.get(symbol, [])
        if n is not None:
            return bars[-n:]
        return list(bars)

    def get_bars_as_dataframe(self, symbol: str, n: int | None = None) -> pd.DataFrame:
        """Return completed bars as a DataFrame with OHLCV columns.

        Compatible with the DataHandler interface — same column names
        as HistoricalCSVDataHandler.
        """
        bars = self.get_completed_bars(symbol, n)
        if not bars:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
        records = [
            {
                "datetime": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        df = pd.DataFrame(records)
        df.set_index("datetime", inplace=True)
        return df

    def flush(self, symbol: str | None = None) -> None:
        """Force-complete the current partial bar(s).

        Useful at session end or when switching to a new trading day.
        """
        if symbol is not None:
            bar = self._current_bars.pop(symbol, None)
            if bar is not None:
                self._emit_bar(bar)
        else:
            for sym in list(self._current_bars.keys()):
                bar = self._current_bars.pop(sym)
                self._emit_bar(bar)

    def reset(self) -> None:
        """Clear all bars and state."""
        self._current_bars.clear()
        self._completed_bars.clear()

    def _floor_timestamp(self, ts: datetime) -> datetime:
        """Round timestamp down to the nearest bar boundary."""
        seconds = int(self._interval.total_seconds())
        epoch = ts.timestamp()
        floored = (int(epoch) // seconds) * seconds
        return datetime.fromtimestamp(floored, tz=ts.tzinfo)

    def _emit_bar(self, bar: Bar) -> None:
        """Store a completed bar and notify the callback."""
        if bar.symbol not in self._completed_bars:
            self._completed_bars[bar.symbol] = []
        self._completed_bars[bar.symbol].append(bar)

        if self._on_bar is not None:
            self._on_bar(bar)

        logger.debug(
            "Bar completed: %s %s O=%.4f H=%.4f L=%.4f C=%.4f V=%.1f (%d ticks)",
            bar.symbol,
            bar.timestamp,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            bar.tick_count,
        )
```

**Tests to write (`tests/test_bar_aggregator.py`):**
- `TestBar`: dataclass fields
- `TestBarAggregator`:
  - First tick creates new bar
  - Ticks within same interval update OHLCV correctly (high/low/close/volume)
  - Tick crossing interval boundary emits completed bar and starts new one
  - `on_bar` callback is invoked on completion
  - `get_completed_bars()` returns history, respects `n` limit
  - `get_bars_as_dataframe()` returns proper DataFrame with correct columns and DatetimeIndex
  - `flush()` emits partial bar, clears current
  - `flush(symbol)` only flushes that symbol
  - `reset()` clears everything
  - `_floor_timestamp()` floors correctly for various intervals (1m, 5m, 1h)
  - Zero-interval raises ValueError
  - Multiple symbols tracked independently

**Target: ~35 tests, all passing**

---

## Week 2: Live Data Handler

### Step 2.1: LiveDataHandler (implements DataHandler ABC)
**Time:** Day 1-3

**File: `src/live/data_handler.py`**
```python
"""LiveDataHandler — bridges live data feeds to the DataHandler interface.

Wraps a BarAggregator to provide the same get_latest_bars() / update_bars()
interface that strategies and the engine expect. This allows strategies to
run unchanged against both historical and live data.
"""

import logging
from datetime import datetime

import pandas as pd

from src.data.data_handler import DataHandler

from .bar_aggregator import Bar, BarAggregator

logger = logging.getLogger(__name__)


class LiveDataHandler(DataHandler):
    """DataHandler implementation backed by a live data feed.

    Wraps a BarAggregator and presents completed bars through the
    standard DataHandler interface. Strategies using get_latest_bars()
    work identically to backtest mode.

    Constructor Args:
        symbols: List of symbols to track.
        bar_aggregator: BarAggregator receiving live ticks.
        max_history: Maximum number of bars to retain per symbol.
    """

    def __init__(
        self,
        symbols: list[str],
        bar_aggregator: BarAggregator,
        max_history: int = 5000,
    ) -> None:
        self._symbols = symbols
        self._bar_aggregator = bar_aggregator
        self._max_history = max_history
        self._current_timestamp: datetime | None = None
        self._bar_count = 0
        self._new_bar_available = False

        # Register as a bar listener
        self._bar_aggregator._on_bar = self._on_new_bar

    def _on_new_bar(self, bar: Bar) -> None:
        """Called by the BarAggregator when a bar completes."""
        self._current_timestamp = bar.timestamp
        self._bar_count += 1
        self._new_bar_available = True
        logger.debug("New bar: %s at %s", bar.symbol, bar.timestamp)

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return the last n completed bars as a DataFrame.

        Returns the same format as HistoricalCSVDataHandler:
        columns = [open, high, low, close, volume], DatetimeIndex.
        """
        return self._bar_aggregator.get_bars_as_dataframe(symbol, n)

    def update_bars(self) -> bool:
        """Check if new bars are available since last call.

        In live mode this doesn't advance an index — it checks whether
        the BarAggregator has emitted new bars since the last call.
        Returns True if new data was available, False otherwise.
        """
        if self._new_bar_available:
            self._new_bar_available = False
            return True
        return False

    def get_current_timestamp(self) -> datetime:
        """Return the timestamp of the most recent completed bar."""
        if self._current_timestamp is None:
            return datetime.now()
        return self._current_timestamp

    @property
    def continue_backtest(self) -> bool:
        """Always True in live mode — runs until explicitly stopped."""
        return True

    def reset(self) -> None:
        """Reset the live data handler state."""
        self._bar_aggregator.reset()
        self._current_timestamp = None
        self._bar_count = 0
        self._new_bar_available = False

    def get_all_symbols(self) -> list[str]:
        """Return all tracked symbols."""
        return list(self._symbols)

    def get_latest_bar_value(self, symbol: str, field: str) -> float:
        """Return a single field from the most recent bar."""
        bars = self.get_latest_bars(symbol, n=1)
        if bars.empty:
            raise ValueError(f"No bars available for {symbol}")
        return float(bars[field].iloc[-1])
```

**Tests to write (`tests/test_live_data_handler.py`):**
- `get_latest_bars()` returns DataFrame with correct columns from aggregator
- `update_bars()` returns True after new bar, False on subsequent call
- `continue_backtest` is always True
- `get_current_timestamp()` returns latest bar timestamp
- `get_latest_bar_value()` returns correct field, raises on empty
- `reset()` clears state
- `get_all_symbols()` returns configured symbols
- Integration: tick → aggregator → handler → get_latest_bars flow

**Target: ~25 tests, all passing**

---

### Step 2.2: Data Feed Gap Detection
**Time:** Day 3-4

Add gap detection to `BarAggregator`:

```python
# Additional fields on BarAggregator.__init__:
self._expected_next: dict[str, datetime] = {}
self._gaps: list[dict[str, Any]] = []

def get_gaps(self) -> list[dict[str, Any]]:
    """Return detected data gaps: [{symbol, expected, actual, duration}]."""
    return list(self._gaps)
```

When a new bar arrives, if the timestamp is more than 2× the interval after the expected time, record a gap. This is critical for detecting WebSocket disconnection data loss.

**Tests:**
- Gap detected when bar timestamp jumps by >2× interval
- No gap on normal consecutive bars
- Gap records symbol, expected time, actual time, duration
- Multiple symbols track gaps independently

**Target: ~10 tests**

---

### Step 2.3: Config Extension
**Time:** Day 5

**Add to `src/config.py`:**
```python
class LiveDataConfig(BaseModel):
    """Configuration for live data feeds."""

    feed_type: str = "websocket"  # websocket | polling
    url: str = ""
    symbols: list[str] = []
    bar_interval_seconds: int = 60  # 1-minute bars
    max_history: int = 5000
    reconnect_attempts: int = 10
    reconnect_delay: float = 1.0


class BrokerConfig(BaseModel):
    """Configuration for broker connection."""

    broker_type: str = "paper"  # paper | (future: alpaca, ibkr)
    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    paper_mode: bool = True
    fill_delay_ms: int = 100  # simulated fill delay for paper
    slippage_model: str = "fixed"  # fixed | proportional | volume


class PersistenceConfig(BaseModel):
    """Configuration for state persistence."""

    enabled: bool = True
    state_dir: str = "state/"
    save_interval_seconds: int = 300  # save every 5 minutes
    max_snapshots: int = 10


class LiveConfig(BaseModel):
    """Top-level live/paper trading configuration."""

    data: LiveDataConfig = LiveDataConfig()
    broker: BrokerConfig = BrokerConfig()
    persistence: PersistenceConfig = PersistenceConfig()
```

Add `live: LiveConfig = LiveConfig()` to `BacktestConfig` (with default so existing configs load unchanged).

**Tests:**
- LiveConfig defaults load cleanly
- Existing backtest configs still load (backward compatibility)
- All fields validate correctly

**Target: ~15 tests**

---

## Week 3: Broker Adapter & Paper Execution

### Step 3.1: Broker Adapter ABC
**Time:** Day 1-2

**File: `src/broker/__init__.py`**
```python
from .base_broker import BrokerAdapter, BrokerFill, BrokerOrder, OrderStatus
from .order_manager import OrderManager
from .paper_broker import PaperBroker

__all__ = [
    "BrokerAdapter",
    "BrokerFill",
    "BrokerOrder",
    "OrderManager",
    "OrderStatus",
    "PaperBroker",
]
```

**File: `src/broker/base_broker.py`**
```python
"""Abstract broker adapter interface.

Defines the contract for all broker implementations (paper, Alpaca, IBKR).
The adapter translates between our internal OrderEvent/FillEvent types
and the broker's API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto


class OrderStatus(Enum):
    """Lifecycle states for a broker order."""

    PENDING = auto()       # Created, not yet submitted
    SUBMITTED = auto()     # Sent to broker
    ACCEPTED = auto()      # Acknowledged by broker
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


@dataclass
class BrokerOrder:
    """Internal representation of an order submitted to a broker.

    Attributes:
        order_id: Unique identifier (assigned by broker adapter).
        symbol: Instrument to trade.
        side: "BUY" or "SELL".
        order_type: "MARKET" or "LIMIT".
        quantity: Number of shares/units.
        limit_price: Limit price (None for market orders).
        status: Current lifecycle state.
        submitted_at: When the order was submitted.
        filled_at: When the order was fully filled (None if not yet).
        filled_quantity: Cumulative filled quantity.
        filled_avg_price: Volume-weighted average fill price.
        broker_order_id: External order ID from the broker (if any).
    """

    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: int
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    filled_quantity: int = 0
    filled_avg_price: float = 0.0
    broker_order_id: str | None = None


@dataclass(frozen=True)
class BrokerFill:
    """A fill notification from the broker.

    Attributes:
        order_id: Our internal order ID.
        symbol: Instrument filled.
        side: "BUY" or "SELL".
        quantity: Quantity filled in this notification.
        price: Execution price.
        commission: Commission charged.
        timestamp: Fill timestamp.
    """

    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    timestamp: datetime


class BrokerAdapter(ABC):
    """Abstract interface for broker interactions.

    All broker implementations (paper, live) must implement these methods.
    The adapter handles order submission, cancellation, and status queries.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the broker."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        ...

    @abstractmethod
    async def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        """Submit an order to the broker.

        Returns the order with updated status and broker_order_id.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel a pending/submitted order.

        Returns the order with updated status.
        """
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> BrokerOrder:
        """Query the current status of an order."""
        ...

    @abstractmethod
    async def get_positions(self) -> dict[str, dict[str, float]]:
        """Get current positions from the broker.

        Returns: {symbol: {quantity, avg_cost, market_value}}
        """
        ...

    @abstractmethod
    async def get_account_info(self) -> dict[str, float]:
        """Get account information (cash, equity, buying power).

        Returns: {cash, equity, buying_power}
        """
        ...
```

**Tests to write:**
- `OrderStatus` enum values
- `BrokerOrder` dataclass defaults, status transitions
- `BrokerFill` frozen dataclass

**Target: ~10 tests**

---

### Step 3.2: Paper Broker Implementation
**Time:** Day 2-5

**File: `src/broker/paper_broker.py`**
```python
"""Paper trading broker — simulates order execution without real money.

Fills orders based on the last known price from the data handler,
applying configurable slippage and commission models. Supports market
and limit orders, partial fills, and realistic fill delays.
"""

import asyncio
import logging
import uuid
from datetime import datetime

from src.data.data_handler import DataHandler

from .base_broker import BrokerAdapter, BrokerFill, BrokerOrder, OrderStatus

logger = logging.getLogger(__name__)


class PaperBroker(BrokerAdapter):
    """Simulated broker for paper trading.

    Constructor Args:
        initial_capital: Starting cash balance.
        commission_pct: Commission as a fraction of trade value.
        slippage_pct: Slippage as a fraction of price.
        fill_delay_seconds: Simulated delay before fills (0 for instant).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        fill_delay_seconds: float = 0.0,
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._commission_pct = commission_pct
        self._slippage_pct = slippage_pct
        self._fill_delay_seconds = fill_delay_seconds
        self._connected = False

        # Order tracking
        self._orders: dict[str, BrokerOrder] = {}
        self._fills: list[BrokerFill] = []
        self._positions: dict[str, dict[str, float]] = {}
        # {symbol: {quantity, avg_cost, market_value}}

        # Fill callbacks
        self._fill_callbacks: list[
            "asyncio.Callable[[BrokerFill], None]"
        ] = []

        # Price source — set via set_data_handler
        self._data_handler: DataHandler | None = None

    def set_data_handler(self, data_handler: DataHandler) -> None:
        """Set the data handler for price lookups."""
        self._data_handler = data_handler

    def add_fill_callback(
        self, callback: "asyncio.Callable[[BrokerFill], None]"
    ) -> None:
        """Register a callback invoked when an order is filled."""
        self._fill_callbacks.append(callback)

    async def connect(self) -> None:
        self._connected = True
        logger.info("Paper broker connected (capital=%.2f)", self._cash)

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Paper broker disconnected")

    async def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        """Submit an order for simulated execution.

        Market orders fill immediately (after optional delay).
        Limit orders fill if the price condition is met.
        """
        if not self._connected:
            order.status = OrderStatus.REJECTED
            return order

        order.order_id = order.order_id or str(uuid.uuid4())
        order.broker_order_id = f"PAPER-{order.order_id[:8]}"
        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED
        self._orders[order.order_id] = order

        logger.info(
            "Order submitted: %s %s %d %s @ %s",
            order.side,
            order.symbol,
            order.quantity,
            order.order_type,
            order.limit_price or "MARKET",
        )

        # Simulate fill
        if order.order_type == "MARKET":
            if self._fill_delay_seconds > 0:
                await asyncio.sleep(self._fill_delay_seconds)
            await self._execute_fill(order)
        elif order.order_type == "LIMIT":
            # Check if limit can be filled at current price
            await self._try_limit_fill(order)

        return order

    async def cancel_order(self, order_id: str) -> BrokerOrder:
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Unknown order: {order_id}")
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return order
        order.status = OrderStatus.CANCELLED
        logger.info("Order cancelled: %s", order_id)
        return order

    async def get_order_status(self, order_id: str) -> BrokerOrder:
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Unknown order: {order_id}")
        return order

    async def get_positions(self) -> dict[str, dict[str, float]]:
        return {k: dict(v) for k, v in self._positions.items()}

    async def get_account_info(self) -> dict[str, float]:
        total_value = sum(
            pos.get("market_value", 0.0) for pos in self._positions.values()
        )
        equity = self._cash + total_value
        return {
            "cash": self._cash,
            "equity": equity,
            "buying_power": self._cash,
            "initial_capital": self._initial_capital,
        }

    async def _execute_fill(self, order: BrokerOrder) -> None:
        """Fill a market order at the current price with slippage."""
        price = self._get_current_price(order.symbol)
        if price is None:
            order.status = OrderStatus.REJECTED
            logger.warning(
                "No price available for %s — order rejected", order.symbol
            )
            return

        # Apply slippage
        if order.side == "BUY":
            fill_price = price * (1 + self._slippage_pct)
        else:
            fill_price = price * (1 - self._slippage_pct)

        commission = abs(fill_price * order.quantity * self._commission_pct)

        # Update order
        order.filled_quantity = order.quantity
        order.filled_avg_price = fill_price
        order.filled_at = datetime.now()
        order.status = OrderStatus.FILLED

        # Update cash
        if order.side == "BUY":
            self._cash -= fill_price * order.quantity + commission
        else:
            self._cash += fill_price * order.quantity - commission

        # Update positions
        self._update_position(order.symbol, order.side, order.quantity, fill_price)

        # Create fill
        fill = BrokerFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=order.filled_at,
        )
        self._fills.append(fill)

        # Notify callbacks
        for cb in self._fill_callbacks:
            cb(fill)

        logger.info(
            "Order filled: %s %s %d @ %.4f (commission=%.2f)",
            order.side,
            order.symbol,
            order.quantity,
            fill_price,
            commission,
        )

    async def _try_limit_fill(self, order: BrokerOrder) -> None:
        """Check if a limit order can be filled at the current price."""
        price = self._get_current_price(order.symbol)
        if price is None:
            return  # Leave as submitted — will check on next tick

        can_fill = False
        if order.side == "BUY" and price <= order.limit_price:
            can_fill = True
        elif order.side == "SELL" and price >= order.limit_price:
            can_fill = True

        if can_fill:
            if self._fill_delay_seconds > 0:
                await asyncio.sleep(self._fill_delay_seconds)
            # Fill at limit price (better execution)
            order.filled_quantity = order.quantity
            order.filled_avg_price = order.limit_price
            order.filled_at = datetime.now()
            order.status = OrderStatus.FILLED

            commission = abs(
                order.limit_price * order.quantity * self._commission_pct
            )

            if order.side == "BUY":
                self._cash -= order.limit_price * order.quantity + commission
            else:
                self._cash += order.limit_price * order.quantity - commission

            self._update_position(
                order.symbol, order.side, order.quantity, order.limit_price
            )

            fill = BrokerFill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.limit_price,
                commission=commission,
                timestamp=order.filled_at,
            )
            self._fills.append(fill)
            for cb in self._fill_callbacks:
                cb(fill)

    def _get_current_price(self, symbol: str) -> float | None:
        """Get the latest price from the data handler."""
        if self._data_handler is None:
            return None
        try:
            return self._data_handler.get_latest_bar_value(symbol, "close")
        except (ValueError, KeyError):
            return None

    def _update_position(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
    ) -> None:
        """Update position tracking after a fill."""
        if symbol not in self._positions:
            self._positions[symbol] = {
                "quantity": 0.0,
                "avg_cost": 0.0,
                "market_value": 0.0,
            }
        pos = self._positions[symbol]
        if side == "BUY":
            total_cost = pos["avg_cost"] * pos["quantity"] + price * quantity
            pos["quantity"] += quantity
            if pos["quantity"] > 0:
                pos["avg_cost"] = total_cost / pos["quantity"]
        else:
            pos["quantity"] -= quantity
            if pos["quantity"] <= 0:
                pos["quantity"] = 0.0
                pos["avg_cost"] = 0.0
        pos["market_value"] = pos["quantity"] * price
```

**Tests to write (`tests/test_paper_broker.py`):**
- Connect/disconnect lifecycle
- Market order: submitted → filled, cash deducted, position updated
- Limit order: fill when price condition met, no fill when not met
- Slippage applied correctly (buy = price up, sell = price down)
- Commission calculated and deducted
- Cancel order changes status
- Get positions returns current state
- Get account info: cash + market_value = equity
- Fill callback invoked
- Order rejected when not connected
- Multiple orders same symbol accumulate position
- Sell reduces position, avg_cost resets at zero
- Order rejected when no price available

**Target: ~40 tests, all passing**

---

## Week 4: Order Manager

### Step 4.1: Order Lifecycle State Machine
**Time:** Day 1-3

**File: `src/broker/order_manager.py`**
```python
"""Order lifecycle management with state machine.

Tracks all orders through their lifecycle, validates state transitions,
and provides query interfaces for order status. Bridges between the
event-driven engine (OrderEvent) and the broker adapter (BrokerOrder).
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from src.events.event import FillEvent, OrderEvent, OrderSide

from .base_broker import BrokerAdapter, BrokerFill, BrokerOrder, OrderStatus

logger = logging.getLogger(__name__)

# Valid state transitions
VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING: {OrderStatus.SUBMITTED, OrderStatus.REJECTED},
    OrderStatus.SUBMITTED: {
        OrderStatus.ACCEPTED,
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.ACCEPTED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.FILLED: set(),      # terminal
    OrderStatus.CANCELLED: set(),   # terminal
    OrderStatus.REJECTED: set(),    # terminal
    OrderStatus.EXPIRED: set(),     # terminal
}


class OrderManager:
    """Manages the full order lifecycle.

    Converts OrderEvents from the engine into BrokerOrders,
    submits them via the BrokerAdapter, tracks state transitions,
    and converts BrokerFills back into FillEvents for the engine.

    Constructor Args:
        broker: The broker adapter to submit orders through.
    """

    def __init__(self, broker: BrokerAdapter) -> None:
        self._broker = broker
        self._orders: dict[str, BrokerOrder] = {}
        self._order_history: list[dict[str, Any]] = []
        self._pending_cancels: set[str] = set()

    @property
    def active_orders(self) -> list[BrokerOrder]:
        """Return all non-terminal orders."""
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        return [o for o in self._orders.values() if o.status not in terminal]

    @property
    def all_orders(self) -> dict[str, BrokerOrder]:
        """Return all orders (active and historical)."""
        return dict(self._orders)

    async def submit_order_event(self, order_event: OrderEvent) -> BrokerOrder:
        """Convert an OrderEvent to a BrokerOrder and submit it.

        This is the main entry point from the engine's event loop.
        """
        order_id = str(uuid.uuid4())
        broker_order = BrokerOrder(
            order_id=order_id,
            symbol=order_event.symbol,
            side=order_event.side.name,
            order_type=order_event.order_type.name,
            quantity=order_event.quantity,
            limit_price=order_event.limit_price,
        )

        self._orders[order_id] = broker_order
        self._record_transition(order_id, None, OrderStatus.PENDING)

        # Submit through broker
        updated = await self._broker.submit_order(broker_order)
        self._orders[order_id] = updated

        return updated

    async def cancel_order(self, order_id: str) -> BrokerOrder:
        """Request cancellation of an order."""
        if order_id not in self._orders:
            raise ValueError(f"Unknown order: {order_id}")

        order = self._orders[order_id]
        old_status = order.status
        updated = await self._broker.cancel_order(order_id)

        if updated.status != old_status:
            self._record_transition(order_id, old_status, updated.status)

        self._orders[order_id] = updated
        return updated

    def on_fill(self, fill: BrokerFill) -> FillEvent:
        """Convert a BrokerFill into a FillEvent for the engine.

        Called by the paper trading engine when it receives a fill
        notification from the broker.
        """
        side = OrderSide.BUY if fill.side == "BUY" else OrderSide.SELL

        return FillEvent(
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            side=side,
            quantity=fill.quantity,
            fill_price=fill.price,
            commission=fill.commission,
        )

    def get_order(self, order_id: str) -> BrokerOrder | None:
        """Look up an order by ID."""
        return self._orders.get(order_id)

    def get_orders_for_symbol(self, symbol: str) -> list[BrokerOrder]:
        """Return all orders for a given symbol."""
        return [o for o in self._orders.values() if o.symbol == symbol]

    def get_order_history(self) -> list[dict[str, Any]]:
        """Return the full state transition history."""
        return list(self._order_history)

    def reset(self) -> None:
        """Clear all order state."""
        self._orders.clear()
        self._order_history.clear()
        self._pending_cancels.clear()

    def _record_transition(
        self,
        order_id: str,
        from_status: OrderStatus | None,
        to_status: OrderStatus,
    ) -> None:
        """Record an order state transition for audit trail."""
        self._order_history.append(
            {
                "order_id": order_id,
                "from": from_status.name if from_status else None,
                "to": to_status.name,
                "timestamp": datetime.now(),
            }
        )
        logger.debug(
            "Order %s: %s → %s",
            order_id[:8],
            from_status.name if from_status else "NEW",
            to_status.name,
        )
```

**Tests to write (`tests/test_order_manager.py`):**
- `VALID_TRANSITIONS`: verify terminal states have empty sets
- `submit_order_event()`: converts OrderEvent fields correctly, assigns UUID
- `cancel_order()`: calls broker, records transition
- `cancel_order()` on unknown order raises ValueError
- `on_fill()`: converts BrokerFill → FillEvent with correct fields
- `active_orders` filters out terminal states
- `get_orders_for_symbol()` filters correctly
- `get_order_history()` records all transitions
- `reset()` clears everything
- Integration: submit → fill → order status is FILLED

**Target: ~25 tests, all passing**

---

## Week 5: Paper Trading Engine

### Step 5.1: Paper Trading Engine (Async Event Loop)
**Time:** Day 1-4

**File: `src/engine/__init__.py`**
```python
from .paper_engine import PaperTradingEngine
from .state_manager import StateManager

__all__ = ["PaperTradingEngine", "StateManager"]
```

**File: `src/engine/paper_engine.py`**
```python
"""PaperTradingEngine — real-time event loop for paper trading.

Mirrors BacktestEngine's event-driven architecture but runs
asynchronously against live data feeds. The core event cycle
is identical: MarketEvent → Strategy → SignalEvent → Portfolio →
OrderEvent → Broker → FillEvent → Portfolio.

The engine runs continuously until stopped, processing events
as they arrive from the live data feed.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from src.data.data_handler import DataHandler
from src.events.event import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from src.events.queue import EventQueue
from src.risk.risk_manager import RiskManager

from src.broker.base_broker import BrokerFill
from src.broker.order_manager import OrderManager

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Async event-driven engine for paper trading.

    Reuses the same Strategy, Portfolio, and RiskManager components
    from backtesting. The key differences from BacktestEngine:
    - Runs asynchronously with asyncio
    - Receives bars from LiveDataHandler instead of replaying history
    - Submits orders through BrokerAdapter instead of simulating fills
    - Runs continuously until explicitly stopped

    Constructor Args:
        data_handler: LiveDataHandler providing real-time bars.
        strategy: Strategy instance (same as backtest).
        portfolio: Portfolio instance (same as backtest).
        order_manager: OrderManager wrapping a BrokerAdapter.
        risk_manager: Optional RiskManager for pre-trade checks.
        event_poll_interval: Seconds between event queue polls.
        bar_poll_interval: Seconds between checking for new bars.
    """

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Any,  # StrategyProtocol
        portfolio: Any,  # PortfolioProtocol
        order_manager: OrderManager,
        risk_manager: RiskManager | None = None,
        event_poll_interval: float = 0.01,
        bar_poll_interval: float = 0.1,
    ) -> None:
        self._data_handler = data_handler
        self._strategy = strategy
        self._portfolio = portfolio
        self._order_manager = order_manager
        self._risk_manager = risk_manager
        self._event_poll_interval = event_poll_interval
        self._bar_poll_interval = bar_poll_interval

        self._events = EventQueue()
        self._running = False
        self._started_at: datetime | None = None

        # Statistics
        self.bars_processed = 0
        self.events_processed = 0
        self.orders_submitted = 0
        self.orders_rejected = 0
        self.fills_processed = 0

        # Wire up components
        self._strategy.set_event_queue(self._events)
        self._portfolio.set_event_queue(self._events)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def uptime_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return (datetime.now() - self._started_at).total_seconds()

    async def start(self) -> None:
        """Start the paper trading engine.

        Runs two concurrent tasks:
        1. Bar poller: checks for new bars and emits MarketEvents
        2. Event processor: processes events from the queue
        """
        self._running = True
        self._started_at = datetime.now()
        logger.info("Paper trading engine started")

        try:
            await asyncio.gather(
                self._bar_poll_loop(),
                self._event_process_loop(),
            )
        except asyncio.CancelledError:
            logger.info("Paper trading engine cancelled")
        finally:
            self._running = False
            logger.info(
                "Engine stopped. Bars=%d Events=%d Orders=%d Fills=%d",
                self.bars_processed,
                self.events_processed,
                self.orders_submitted,
                self.fills_processed,
            )

    async def stop(self) -> None:
        """Signal the engine to stop gracefully."""
        self._running = False
        logger.info("Stop signal sent to paper trading engine")

    async def _bar_poll_loop(self) -> None:
        """Poll for new bars and emit MarketEvents."""
        while self._running:
            if self._data_handler.update_bars():
                timestamp = self._data_handler.get_current_timestamp()
                event = MarketEvent(timestamp=timestamp)
                self._events.put(event)
                self.bars_processed += 1
            await asyncio.sleep(self._bar_poll_interval)

    async def _event_process_loop(self) -> None:
        """Process events from the queue."""
        while self._running:
            if self._events.empty():
                await asyncio.sleep(self._event_poll_interval)
                continue

            event = self._events.get()
            if event is None:
                continue

            self.events_processed += 1

            if event.event_type == EventType.MARKET:
                self._handle_market(event)
            elif event.event_type == EventType.SIGNAL:
                self._handle_signal(event)
            elif event.event_type == EventType.ORDER:
                await self._handle_order(event)
            elif event.event_type == EventType.FILL:
                self._handle_fill(event)

    def _handle_market(self, event: MarketEvent) -> None:
        """Dispatch market event to strategy."""
        self._strategy.on_market_data(event, self._data_handler)
        self._portfolio.update_timeindex(event.timestamp)

    def _handle_signal(self, event: SignalEvent) -> None:
        """Dispatch signal event to portfolio for order generation."""
        self._portfolio.on_signal(event, self._data_handler)

    async def _handle_order(self, event: OrderEvent) -> None:
        """Check order with risk manager, then submit to broker."""
        # Risk check
        if self._risk_manager is not None:
            equity = self._portfolio.get_equity()
            positions = self._portfolio.get_positions()
            try:
                current_price = self._data_handler.get_latest_bar_value(
                    event.symbol, "close"
                )
            except (ValueError, KeyError):
                logger.warning(
                    "No price for %s — rejecting order", event.symbol
                )
                self.orders_rejected += 1
                return

            result = self._risk_manager.check_order(
                event, equity, positions, current_price
            )
            if not result.approved:
                logger.warning(
                    "Order rejected by risk manager: %s",
                    result.rejection_reasons,
                )
                self.orders_rejected += 1
                return

        # Submit to broker via order manager
        try:
            broker_order = await self._order_manager.submit_order_event(event)
            self.orders_submitted += 1
            logger.info(
                "Order submitted: %s %s %d %s",
                broker_order.side,
                broker_order.symbol,
                broker_order.quantity,
                broker_order.status.name,
            )

            # If filled immediately (paper broker market orders), emit FillEvent
            if broker_order.status.name == "FILLED":
                fill = BrokerFill(
                    order_id=broker_order.order_id,
                    symbol=broker_order.symbol,
                    side=broker_order.side,
                    quantity=broker_order.filled_quantity,
                    price=broker_order.filled_avg_price,
                    commission=abs(
                        broker_order.filled_avg_price
                        * broker_order.filled_quantity
                        * 0.001
                    ),
                    timestamp=broker_order.filled_at or datetime.now(),
                )
                fill_event = self._order_manager.on_fill(fill)
                self._events.put(fill_event)

        except Exception:
            logger.exception("Failed to submit order")
            self.orders_rejected += 1

    def _handle_fill(self, event: FillEvent) -> None:
        """Dispatch fill event to portfolio."""
        self._portfolio.on_fill(event)
        self.fills_processed += 1

        # Update strategy position tracking
        quantity = event.quantity
        if event.side.name == "SELL":
            quantity = -quantity
        current = self._strategy.get_position(event.symbol) if hasattr(
            self._strategy, "get_position"
        ) else 0
        self._strategy.update_position(event.symbol, current + quantity)

    def get_statistics(self) -> dict[str, Any]:
        """Return engine runtime statistics."""
        return {
            "is_running": self._running,
            "uptime_seconds": self.uptime_seconds,
            "bars_processed": self.bars_processed,
            "events_processed": self.events_processed,
            "orders_submitted": self.orders_submitted,
            "orders_rejected": self.orders_rejected,
            "fills_processed": self.fills_processed,
            "equity": self._portfolio.get_equity(),
            "positions": self._portfolio.get_positions(),
        }
```

**Tests to write (`tests/test_paper_engine.py`):**
- Engine starts and stops cleanly
- MarketEvent → Strategy.on_market_data called
- SignalEvent → Portfolio.on_signal called
- OrderEvent → risk check → broker submission
- OrderEvent rejected by risk manager does not reach broker
- FillEvent → Portfolio.on_fill called, strategy position updated
- `get_statistics()` returns correct counts
- `uptime_seconds` tracks running time
- Bar poll loop emits MarketEvent when update_bars returns True
- Full cycle: market → signal → order → fill integration
- Engine stops on `stop()` call

**Target: ~35 tests, all passing**

---

## Week 6: Reconciliation & State Persistence

### Step 6.1: State Manager
**Time:** Day 1-3

**File: `src/engine/state_manager.py`**
```python
"""State persistence and crash recovery.

Periodically snapshots engine state (positions, cash, equity, orders)
to disk as JSON. On restart, can restore from the latest snapshot to
resume paper trading without losing position/P&L tracking.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StateManager:
    """Manages engine state persistence.

    Constructor Args:
        state_dir: Directory to store state snapshots.
        max_snapshots: Maximum snapshots to retain (oldest deleted).
    """

    def __init__(
        self,
        state_dir: str = "state/",
        max_snapshots: int = 10,
    ) -> None:
        self._state_dir = Path(state_dir)
        self._max_snapshots = max_snapshots
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, state: dict[str, Any]) -> Path:
        """Save a state snapshot to disk.

        Args:
            state: Dictionary containing engine state. Must be
                JSON-serializable. Typically includes: positions,
                cash, equity, orders, statistics, timestamp.

        Returns:
            Path to the saved snapshot file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.json"
        filepath = self._state_dir / filename

        # Add metadata
        state["_snapshot_timestamp"] = datetime.now().isoformat()
        state["_snapshot_version"] = 1

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info("State snapshot saved: %s", filepath)

        # Prune old snapshots
        self._prune_snapshots()

        return filepath

    def load_latest_snapshot(self) -> dict[str, Any] | None:
        """Load the most recent state snapshot.

        Returns:
            State dictionary, or None if no snapshots exist.
        """
        snapshots = self._list_snapshots()
        if not snapshots:
            return None

        latest = snapshots[-1]
        with open(latest) as f:
            state = json.load(f)

        logger.info("State restored from: %s", latest)
        return state

    def load_snapshot(self, filepath: str | Path) -> dict[str, Any]:
        """Load a specific snapshot by path."""
        with open(filepath) as f:
            return json.load(f)

    def list_snapshots(self) -> list[dict[str, Any]]:
        """Return metadata for all available snapshots."""
        snapshots = self._list_snapshots()
        result = []
        for path in snapshots:
            stat = path.stat()
            result.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime
                    ).isoformat(),
                }
            )
        return result

    def clear_snapshots(self) -> int:
        """Delete all snapshots. Returns count deleted."""
        snapshots = self._list_snapshots()
        for path in snapshots:
            path.unlink()
        logger.info("Cleared %d snapshots", len(snapshots))
        return len(snapshots)

    def _list_snapshots(self) -> list[Path]:
        """List snapshot files sorted by name (oldest first)."""
        return sorted(self._state_dir.glob("snapshot_*.json"))

    def _prune_snapshots(self) -> None:
        """Remove oldest snapshots beyond max_snapshots."""
        snapshots = self._list_snapshots()
        while len(snapshots) > self._max_snapshots:
            oldest = snapshots.pop(0)
            oldest.unlink()
            logger.debug("Pruned old snapshot: %s", oldest)
```

**Tests to write (`tests/test_state_manager.py`):**
- `save_snapshot()` creates JSON file with metadata
- `load_latest_snapshot()` returns most recent
- `load_latest_snapshot()` returns None when no snapshots
- Snapshot pruning deletes oldest when exceeding max_snapshots
- `list_snapshots()` returns metadata for all files
- `clear_snapshots()` deletes all, returns count
- State survives save → load round-trip (positions, cash, etc.)
- State dir created automatically if missing
- `load_snapshot()` loads specific file
- Snapshot version field included

**Target: ~20 tests, all passing**

---

### Step 6.2: Reconciliation
**Time:** Day 3-5

Add reconciliation logic to `PaperTradingEngine` or as a standalone utility. In paper mode, reconciliation compares the engine's internal Portfolio positions with the PaperBroker's position tracking.

```python
# In src/engine/paper_engine.py or standalone utility

async def reconcile_positions(self) -> dict[str, Any]:
    """Compare engine positions with broker positions.

    Returns a report of discrepancies. In paper mode this should
    always match. In live mode, discrepancies indicate bugs.
    """
    engine_positions = self._portfolio.get_positions()
    broker_positions = await self._order_manager._broker.get_positions()

    discrepancies = []
    all_symbols = set(engine_positions.keys()) | set(broker_positions.keys())

    for symbol in all_symbols:
        engine_qty = engine_positions.get(symbol, {}).get("quantity", 0)
        broker_qty = broker_positions.get(symbol, {}).get("quantity", 0)
        if engine_qty != broker_qty:
            discrepancies.append({
                "symbol": symbol,
                "engine_quantity": engine_qty,
                "broker_quantity": broker_qty,
                "difference": engine_qty - broker_qty,
            })

    return {
        "timestamp": datetime.now().isoformat(),
        "matched": len(discrepancies) == 0,
        "symbols_checked": len(all_symbols),
        "discrepancies": discrepancies,
    }
```

**Tests:**
- Matching positions return `matched=True`, empty discrepancies
- Mismatched quantity detected
- Symbol in engine but not broker detected
- Symbol in broker but not engine detected
- Multiple symbols checked independently

**Target: ~10 tests**

---

## Week 7: Monitoring & Health Checks

### Step 7.1: Health Monitor
**Time:** Day 1-3

**File: `src/monitoring/__init__.py`**
```python
from .health import HealthMonitor, HealthStatus

__all__ = ["HealthMonitor", "HealthStatus"]
```

**File: `src/monitoring/health.py`**
```python
"""Health monitoring for the paper trading engine.

Tracks key metrics: data feed status, event processing latency,
order fill rates, and system uptime. Provides a unified health
status and alerting callbacks.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall system health status."""

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()


@dataclass
class HealthReport:
    """Snapshot of system health metrics.

    Attributes:
        status: Overall health status.
        timestamp: When the report was generated.
        uptime_seconds: Engine uptime.
        data_feed_connected: Whether the data feed is connected.
        last_bar_age_seconds: Seconds since the last bar was received.
        event_latency_ms: Recent average event processing latency.
        bars_per_minute: Recent bar processing rate.
        fill_rate: Fraction of orders that were filled (vs rejected).
        warnings: List of warning messages.
        errors: List of error messages.
    """

    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float = 0.0
    data_feed_connected: bool = True
    last_bar_age_seconds: float = 0.0
    event_latency_ms: float = 0.0
    bars_per_minute: float = 0.0
    fill_rate: float = 1.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class HealthMonitor:
    """Monitors system health and exposes metrics.

    Constructor Args:
        max_bar_age_seconds: Max seconds without a bar before
            status degrades.
        max_event_latency_ms: Max event latency before warning.
        history_window: Number of recent measurements to keep.
    """

    def __init__(
        self,
        max_bar_age_seconds: float = 120.0,
        max_event_latency_ms: float = 500.0,
        history_window: int = 100,
    ) -> None:
        self._max_bar_age = max_bar_age_seconds
        self._max_event_latency = max_event_latency_ms
        self._history_window = history_window

        self._last_bar_time: datetime | None = None
        self._event_latencies: deque[float] = deque(maxlen=history_window)
        self._bar_times: deque[datetime] = deque(maxlen=history_window)
        self._data_feed_connected = True
        self._uptime_start: datetime | None = None

        # Counters
        self._orders_total = 0
        self._orders_filled = 0

        # Alert callbacks
        self._alert_callbacks: list[Callable[[HealthReport], None]] = []

    def start(self) -> None:
        """Mark the start of monitoring."""
        self._uptime_start = datetime.now()

    def record_bar(self, timestamp: datetime) -> None:
        """Record that a bar was processed."""
        self._last_bar_time = datetime.now()
        self._bar_times.append(self._last_bar_time)

    def record_event_latency(self, latency_ms: float) -> None:
        """Record the processing time for a single event."""
        self._event_latencies.append(latency_ms)

    def record_order_result(self, filled: bool) -> None:
        """Record whether an order was filled or rejected."""
        self._orders_total += 1
        if filled:
            self._orders_filled += 1

    def set_data_feed_status(self, connected: bool) -> None:
        """Update data feed connection status."""
        self._data_feed_connected = connected

    def add_alert_callback(
        self, callback: Callable[[HealthReport], None]
    ) -> None:
        """Register a callback for health alerts."""
        self._alert_callbacks.append(callback)

    def get_health_report(self) -> HealthReport:
        """Generate a current health report."""
        warnings: list[str] = []
        errors: list[str] = []

        # Uptime
        uptime = 0.0
        if self._uptime_start:
            uptime = (datetime.now() - self._uptime_start).total_seconds()

        # Data feed
        if not self._data_feed_connected:
            errors.append("Data feed disconnected")

        # Bar age
        bar_age = 0.0
        if self._last_bar_time:
            bar_age = (datetime.now() - self._last_bar_time).total_seconds()
            if bar_age > self._max_bar_age:
                warnings.append(
                    f"No bars received for {bar_age:.0f}s "
                    f"(threshold: {self._max_bar_age:.0f}s)"
                )

        # Event latency
        avg_latency = 0.0
        if self._event_latencies:
            avg_latency = sum(self._event_latencies) / len(
                self._event_latencies
            )
            if avg_latency > self._max_event_latency:
                warnings.append(
                    f"High event latency: {avg_latency:.1f}ms "
                    f"(threshold: {self._max_event_latency:.1f}ms)"
                )

        # Bar rate
        bars_per_min = 0.0
        if len(self._bar_times) >= 2:
            span = (
                self._bar_times[-1] - self._bar_times[0]
            ).total_seconds()
            if span > 0:
                bars_per_min = (len(self._bar_times) - 1) / span * 60

        # Fill rate
        fill_rate = 1.0
        if self._orders_total > 0:
            fill_rate = self._orders_filled / self._orders_total

        # Determine status
        if errors:
            status = HealthStatus.UNHEALTHY
        elif warnings:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        report = HealthReport(
            status=status,
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            data_feed_connected=self._data_feed_connected,
            last_bar_age_seconds=bar_age,
            event_latency_ms=avg_latency,
            bars_per_minute=bars_per_min,
            fill_rate=fill_rate,
            warnings=warnings,
            errors=errors,
        )

        # Fire alerts if unhealthy
        if status != HealthStatus.HEALTHY:
            for cb in self._alert_callbacks:
                cb(report)

        return report

    def reset(self) -> None:
        """Reset all monitoring state."""
        self._last_bar_time = None
        self._event_latencies.clear()
        self._bar_times.clear()
        self._data_feed_connected = True
        self._uptime_start = None
        self._orders_total = 0
        self._orders_filled = 0
```

**Tests to write (`tests/test_monitoring.py`):**
- `HealthStatus` enum values
- HEALTHY when all metrics normal
- DEGRADED when bar age exceeds threshold
- DEGRADED when event latency exceeds threshold
- UNHEALTHY when data feed disconnected
- `record_bar()` updates last bar time
- `record_event_latency()` tracked, average computed
- `record_order_result()` updates fill rate
- `bars_per_minute` calculation correct
- Alert callback fired on non-healthy status
- `reset()` clears everything
- Multiple warnings combine correctly

**Target: ~25 tests, all passing**

---

### Step 7.2: Paper Trading Script
**Time:** Day 3-5

**File: `scripts/run_paper_trading.py`**
```python
"""Main entry point for paper trading.

Wires up all components from config and starts the async engine.
Handles graceful shutdown on SIGINT/SIGTERM.

Usage:
    python scripts/run_paper_trading.py configs/paper_trading_config.yaml
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import timedelta

from src.broker.order_manager import OrderManager
from src.broker.paper_broker import PaperBroker
from src.config import load_config
from src.engine.paper_engine import PaperTradingEngine
from src.engine.state_manager import StateManager
from src.live.bar_aggregator import BarAggregator
from src.live.data_handler import LiveDataHandler
from src.monitoring.health import HealthMonitor
from src.portfolio.portfolio import Portfolio
from src.risk.risk_manager import RiskLimits, RiskManager


# Builder functions mirror scripts/run_backtest.py pattern
# (build_strategy, build_position_sizer, etc. reused)

async def main(config_path: str) -> None:
    config = load_config(config_path)

    # Build components
    # ... (component wiring from config, similar to run_backtest.py)

    engine = PaperTradingEngine(...)
    state_manager = StateManager(...)
    health_monitor = HealthMonitor(...)

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(engine.stop()))

    # Restore state if available
    snapshot = state_manager.load_latest_snapshot()
    if snapshot:
        logging.info("Restored state from snapshot")
        # Apply snapshot to portfolio/positions...

    # Periodic state save
    async def periodic_save():
        while engine.is_running:
            await asyncio.sleep(config.live.persistence.save_interval_seconds)
            state = engine.get_statistics()
            state_manager.save_snapshot(state)

    # Run engine + periodic save
    await asyncio.gather(
        engine.start(),
        periodic_save(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument("config", help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    asyncio.run(main(args.config))
```

**Config file: `configs/paper_trading_config.yaml`**
```yaml
data:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  data_source: "yfinance"

execution:
  initial_capital: 100000.0
  commission_pct: 0.001
  slippage_pct: 0.0005

strategy:
  name: "sma_crossover"
  parameters:
    fast_period: 10
    slow_period: 30

sizing:
  method: "fixed_fraction"
  parameters:
    fraction: 0.05

risk:
  enabled: true
  max_position_pct: 0.10
  max_portfolio_exposure_pct: 0.80
  max_daily_loss_pct: 0.02
  max_drawdown_pct: 0.10
  max_open_positions: 10
  max_orders_per_day: 50

live:
  data:
    feed_type: "websocket"
    bar_interval_seconds: 60
    max_history: 5000
    reconnect_attempts: 10
  broker:
    broker_type: "paper"
    paper_mode: true
    fill_delay_ms: 100
  persistence:
    enabled: true
    state_dir: "state/"
    save_interval_seconds: 300
    max_snapshots: 10
```

---

## Week 8: Integration Testing & Stability

### Step 8.1: Integration Tests
**Time:** Day 1-3

**File: `tests/test_phase4_integration.py`**

End-to-end integration tests covering:

1. **Full pipeline (mock data):** Create a mock data feed that emits scripted ticks → BarAggregator → LiveDataHandler → PaperTradingEngine with SMA strategy → verify signals generated, orders submitted, fills processed, positions updated
2. **State persistence round-trip:** Run engine → save snapshot → create new engine → restore snapshot → verify positions and equity match
3. **Reconciliation:** Run trades → reconcile engine vs broker → verify no discrepancies
4. **Reconnection:** Simulate data feed disconnect → verify reconnection → verify no data loss or duplicate bars
5. **Risk manager integration:** Submit order exceeding limits → verify rejection → verify strategy continues
6. **Health monitoring:** Run engine with stale data → verify DEGRADED status
7. **Multiple symbols:** Run with 3+ symbols → verify independent position tracking
8. **Graceful shutdown:** Start engine → stop → verify clean state, no pending orders

**Target: ~30 integration tests, all passing**

---

### Step 8.2: Stability & Soak Testing
**Time:** Day 3-4

Create a lightweight soak test that runs the paper trading engine for a configurable duration against mock data, verifying:

- No memory leaks (RSS stays bounded)
- No event queue growth (queue size stays near 0)
- No dropped events
- Consistent processing latency
- State snapshots written on schedule

```python
# tests/test_stability.py (or scripts/soak_test.py)
async def test_soak_30_seconds():
    """Run paper engine for 30s with mock feed, verify stability."""
    # Set up mock feed emitting 1 tick/10ms
    # Assert: events_processed > 0, queue empty, memory bounded
```

**Target: ~5 stability tests**

---

### Step 8.3: Documentation & Config Validation
**Time:** Day 5

- Verify all new config sections have defaults (backward compatibility with Phase 1-3 configs)
- Verify `make check` passes: black, ruff, mypy, pytest
- Verify all new modules have `__init__.py` with proper exports
- Verify all `zip()` calls use `strict=True`

---

## Test Count Summary

| Week | Module | Tests |
|------|--------|-------|
| 1 | Data Feed (WebSocket, reconnection) | ~30 |
| 1 | Bar Aggregator | ~35 |
| 2 | Live Data Handler | ~25 |
| 2 | Gap Detection | ~10 |
| 2 | Config Extension | ~15 |
| 3 | Broker Adapter (base) | ~10 |
| 3 | Paper Broker | ~40 |
| 4 | Order Manager | ~25 |
| 5 | Paper Trading Engine | ~35 |
| 6 | State Manager | ~20 |
| 6 | Reconciliation | ~10 |
| 7 | Health Monitor | ~25 |
| 8 | Integration | ~30 |
| 8 | Stability | ~5 |
| **Total** | **New tests** | **~315** |
| | **Cumulative (997 + 315)** | **~1312** |

---

## Dependencies to Add

```
# requirements.txt additions
aiohttp>=3.9.0         # Async HTTP + WebSocket client
```

No Redis required in Phase 4 — the BarAggregator handles in-memory bar storage. Redis can be added in Phase 5 if needed for multi-process architectures.

---

## Key Design Decisions

1. **Reuse existing protocols.** LiveDataHandler implements the same `DataHandler` ABC. Strategies work unchanged. Portfolio works unchanged. Only the engine and execution layer are new.

2. **Async only where needed.** The broker layer is async (network I/O). Strategy/Portfolio remain synchronous — called from the async engine but don't need to be async themselves.

3. **No external broker API yet.** Phase 4 uses `PaperBroker` only. The `BrokerAdapter` ABC is designed so Phase 5 can add `AlpacaBroker`, `IBKRBroker` etc. without changing the engine.

4. **State persistence is JSON.** Simple, debuggable, sufficient for paper trading. Phase 5 can upgrade to SQLite or Redis if needed.

5. **WebSocket parse function is pluggable.** `WebSocketDataFeed` takes a `parse_message` callable — exchange-specific parsing is injected, not hardcoded. This avoids coupling to any single exchange's message format.

6. **No new event types.** The existing MarketEvent/SignalEvent/OrderEvent/FillEvent cycle is preserved exactly. The paper engine routes events through the same path as BacktestEngine.

---

## Success Metrics

- All `make check` passes (black, ruff, mypy, pytest)
- 1300+ tests, all green
- Paper trading engine runs for 30+ seconds in soak test without errors
- State snapshot save/restore round-trip works
- Position reconciliation passes (engine matches broker)
- Health monitor correctly detects degraded/unhealthy states
- Existing Phase 1-3 configs load without changes
- Execution latency < 500ms in integration tests

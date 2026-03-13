# Phase 5 — Week 2: Real-Time Market Data Feed & Historical Backfill

**Status:** COMPLETE
**Tests added:** 72 (42 + 30)
**Cumulative tests:** 1456

---

## What Was Built

### `src/live/alpaca_feed.py` — AlpacaDataFeed

Implements `LiveDataFeed` ABC using Alpaca's real-time market data WebSocket (`wss://stream.data.alpaca.markets/v2`).

**Constructor args:**
- `api_key`, `api_secret` — Alpaca credentials
- `feed` — data tier: `"iex"` (free) or `"sip"` (premium)
- `base_ws_url` — optional override for testing
- `max_reconnect_attempts`, `initial_reconnect_delay` — reconnect behavior

**Connection flow:**
1. Open WebSocket to `{base_url}/{feed}` (e.g. `.../v2/iex`)
2. Receive welcome message: `[{"T": "success", "msg": "connected"}]`
3. Send auth: `{"action": "auth", "key": "...", "secret": "..."}`
4. Receive auth response: `[{"T": "success", "msg": "authenticated"}]`
5. Start background listen loop

**Methods:**

| Method | Description |
|--------|-------------|
| `connect()` | WebSocket connect + authenticate |
| `disconnect()` | Cancel listen task, close WebSocket |
| `subscribe(symbols)` | Send subscribe for trades + quotes |
| `unsubscribe(symbols)` | Send unsubscribe |

**Message types handled:**

| Alpaca `T` field | Type | Tick mapping |
|------------------|------|-------------|
| `t` | Trade | `price=p`, `volume=s` |
| `q` | Quote | `price=mid(bp,ap)`, `bid=bp`, `ask=ap` |
| `b` | Bar | `price=c` (close), `volume=v` |
| `success`, `error`, `subscription` | Control | Logged, no tick emitted |

**Reconnection:** exponential backoff (1s → 2s → 4s ... capped at 60s), re-authenticates and re-subscribes on reconnect.

**Module-level helpers:** `_parse_trade()`, `_parse_quote()`, `_parse_bar()`, `_parse_timestamp()`, `_is_message_type()`, `_get_message_field()`

### `src/live/alpaca_historical.py` — AlpacaHistoricalClient & backfill_aggregator

Fetches historical OHLCV bars from Alpaca's REST Market Data API so strategies and ML models have data from bar one.

**AlpacaHistoricalClient:**

| Method | Alpaca Endpoint | Description |
|--------|----------------|-------------|
| `open()` | — | Creates aiohttp session with auth headers |
| `close()` | — | Closes session |
| `fetch_bars(symbol, ...)` | `GET /v2/stocks/{symbol}/bars` | Fetch bars with timeframe, limit, pagination |
| `fetch_bars_multi(symbols, ...)` | (loops `fetch_bars`) | Fetch for multiple symbols |

Supports pagination via `next_page_token`, configurable timeframe (`1Min`, `5Min`, `1Hour`, `1Day`), and limit up to 10000 bars.

**`backfill_aggregator()` — standalone async function:**

```
backfill_aggregator(aggregator, client, symbols, num_bars=500, timeframe="1Min")
```

Seeds a `BarAggregator` by inserting historical bars directly into `_completed_bars`. This makes `get_latest_bars()` return data immediately, so strategies don't need to wait for enough live bars to accumulate.

**Startup flow:**
```
1. Create AlpacaHistoricalClient
2. Call backfill_aggregator(aggregator, client, symbols, num_bars=500)
   → GET /v2/stocks/AAPL/bars?limit=500&timeframe=1Min
   → Bars inserted into aggregator._completed_bars
3. Connect AlpacaDataFeed (WebSocket)
4. Subscribe to symbols
5. Live ticks flow → BarAggregator appends new bars after historical ones
```

### `src/live/__init__.py` — Updated

Added `AlpacaDataFeed`, `AlpacaHistoricalClient`, `backfill_aggregator` to exports.

### `src/config.py` — Updated

Added `"alpaca"` as valid `feed_type` in `LiveDataConfig.validate_feed_type()`.

---

## Tests

### `tests/test_alpaca_feed.py` — 42 Tests

| Test Class | Count | Coverage |
|------------|-------|----------|
| `TestInit` | 7 | URL defaults, feed tier, initial state |
| `TestConnect` | 5 | Success, auth send, auth failure, bad welcome, URL includes feed |
| `TestDisconnect` | 1 | Clears status and auth flag |
| `TestSubscribe` | 3 | Sends message, updates symbols, no-ws safety |
| `TestUnsubscribe` | 2 | Sends message, removes symbol |
| `TestHandleMessage` | 7 | Trade/quote/bar ticks, multi-message arrays, control msgs, invalid JSON, missing fields |
| `TestParseTrade` | 3 | Valid, missing symbol, missing price |
| `TestParseQuote` | 4 | Valid (mid-price), missing bid/ask/symbol |
| `TestParseBar` | 3 | Valid (close price), missing close/symbol |
| `TestParseTimestamp` | 4 | Z suffix, offset, None, invalid |
| `TestConfigFeedType` | 3 | alpaca valid, websocket valid, bogus rejected |

### `tests/test_alpaca_historical.py` — 30 Tests

| Test Class | Count | Coverage |
|------------|-------|----------|
| `TestInit` | 4 | URL defaults, custom URL, feed tier |
| `TestOpenClose` | 3 | Session creation, close, close-when-none |
| `TestFetchBars` | 8 | Returns bars, preserves OHLCV, respects limit, empty/null response, HTTP error, params, URL |
| `TestFetchBarsMulti` | 2 | Multiple symbols, empty list |
| `TestParseBar` | 7 | Valid, missing timestamp/OHLC, Z/offset parsing, default volume/tick_count |
| `TestBackfillAggregator` | 6 | Populates bars, DataFrame access, multi-symbol, preserves existing, empty response, count dict |

---

## Design Decisions

1. **Separate WebSocket feed and REST historical client** — different concerns (streaming vs batch), different endpoints, independently testable
2. **Backfill inserts directly into `_completed_bars`** — avoids simulating ticks through the aggregator, which would trigger gap detection and on_bar callbacks for historical data
3. **Historical bars prepended** — backfilled bars come before any live bars, maintaining chronological order
4. **Quote mid-price** — quotes emit a Tick with `price = (bid + ask) / 2`, plus `bid` and `ask` fields for spread analysis
5. **IEX default** — free tier for development; SIP available for production use via `feed="sip"`
6. **No real API key needed** — all 72 tests use mocked HTTP/WebSocket responses

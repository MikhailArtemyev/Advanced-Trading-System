# Phase 5 — Week 1: Alpaca Broker Adapter

**Status:** COMPLETE
**Tests added:** 35
**Cumulative tests:** 1384

---

## What Was Built

### `src/broker/alpaca_broker.py` — AlpacaBroker

Implements the `BrokerAdapter` ABC using Alpaca's REST Trade API via aiohttp.

**Key components:**

- **Constants:** `ALPACA_PAPER_URL`, `ALPACA_LIVE_URL` — endpoint URLs for paper and live trading
- **`_STATUS_MAP`** — maps all 16 Alpaca order status strings (`new`, `accepted`, `filled`, `canceled`, `rejected`, etc.) to the internal `OrderStatus` enum
- **Constructor:** accepts `api_key`, `api_secret`, `base_url` (optional override), `paper_mode` (default `True`). Auto-selects paper or live URL when no `base_url` is given

**Methods (all async, implementing BrokerAdapter ABC):**

| Method | Alpaca Endpoint | Description |
|--------|----------------|-------------|
| `connect()` | `GET /v2/account` | Creates aiohttp session, verifies credentials |
| `disconnect()` | — | Closes HTTP session |
| `submit_order(order)` | `POST /v2/orders` | Submits order, maps response to BrokerOrder |
| `cancel_order(order_id)` | `DELETE /v2/orders/{id}` | Cancels order by broker ID |
| `get_order_status(order_id)` | `GET /v2/orders/{id}` | Queries status, fills, prices |
| `get_positions()` | `GET /v2/positions` | Returns `{symbol: {quantity, avg_cost, market_value}}` |
| `get_account_info()` | `GET /v2/account` | Returns `{cash, equity, buying_power}` |

**Private helpers:**
- `_build_headers()` — API key auth headers (`APCA-API-KEY-ID`, `APCA-API-SECRET-KEY`)
- `_map_order_type()` — `MARKET→market`, `LIMIT→limit`, `STOP→stop`, `STOP_LIMIT→stop_limit`
- `_ensure_connected()` — raises `ConnectionError` if not connected

**Order tracking:** internal `_orders` dict (order_id → BrokerOrder) and `_order_map` (order_id → alpaca_order_id) allow mapping between our IDs and Alpaca's.

### `src/broker/__init__.py` — Updated

Added `AlpacaBroker` to imports and `__all__`.

### `tests/test_alpaca_broker.py` — 35 Tests

All tests use mocked aiohttp responses — no real API calls.

| Test Class | Count | Coverage |
|------------|-------|----------|
| `TestInit` | 5 | Paper/live URL, custom URL, initial state |
| `TestConnect` | 3 | Success, 401 invalid creds, 403 forbidden |
| `TestDisconnect` | 1 | Clears connected flag |
| `TestSubmitOrder` | 6 | Market, limit, rejected (403/422), not connected, tracking |
| `TestCancelOrder` | 3 | Success, unknown order, not connected |
| `TestGetOrderStatus` | 3 | Filled, partially filled, unknown order |
| `TestGetPositions` | 3 | Multiple positions, empty, not connected |
| `TestGetAccountInfo` | 2 | Success, failure returns zeros |
| `TestStatusMapping` | 4 | All statuses mapped, specific mappings |
| `TestOrderTypeMapping` | 4 | MARKET, LIMIT, STOP, unknown defaults |
| `TestHeaders` | 1 | Correct header keys |

---

## Design Decisions

1. **aiohttp over alpaca-trade-api SDK** — direct HTTP calls give us full control over error handling, retries, and mocking without adding a heavy dependency
2. **Status map covers all 16 Alpaca statuses** — including edge cases like `pending_cancel`, `pending_replace`, `stopped`, `calculated`
3. **No real API key needed** — all functionality is testable with mocks; real integration deferred to Week 8 (e2e tests)
4. **Order tracking is in-memory** — sufficient for a single session; persistent storage comes in Week 3 (database trade journal)

# Phase 4, Weeks 3-4 — Broker Adapter & Order Management

## What this is

The broker abstraction layer and order lifecycle management. Defines a `BrokerAdapter`
ABC for broker implementations (paper, Alpaca, IBKR), a `PaperBroker` for simulated
execution, and an `OrderManager` that bridges the event system to broker operations
with a strict state machine.

**Before:**
```
Orders go directly from Portfolio to ExecutionHandler (backtest only)
No broker abstraction — execution is just a function call
No order state tracking — orders are fire-and-forget
```

**After:**
```
BrokerAdapter ABC — plug in any broker (paper, Alpaca, IBKR)
PaperBroker — simulated execution with slippage, commission, position tracking
OrderManager — state machine for order lifecycle (PENDING → SUBMITTED → FILLED)
BrokerOrder/BrokerFill dataclasses for typed order/fill representation
OrderEvent ↔ BrokerOrder conversion, BrokerFill → FillEvent conversion
```

## What was built

### 1. Broker Abstractions (`src/broker/base_broker.py`)

- **`OrderStatus`** — 8-state enum:
  ```
  PENDING → SUBMITTED → ACCEPTED → FILLED (terminal)
                                  → CANCELLED (terminal)
                                  → REJECTED (terminal)
                                  → EXPIRED (terminal)
                    → PARTIALLY_FILLED → FILLED
  ```
- **`BrokerOrder`** — mutable dataclass tracking order lifecycle: order_id, symbol, side, order_type, quantity, status, filled_quantity, filled_avg_price, timestamps
- **`BrokerFill`** — frozen dataclass for fill notifications: order_id, symbol, side, quantity, price, commission, timestamp
- **`BrokerAdapter`** — ABC requiring: connect, disconnect, submit_order, cancel_order, get_order_status, get_positions, get_account_info

### 2. Paper Broker (`src/broker/paper_broker.py`)

Simulates a real broker for paper trading:

| Feature | Implementation |
|---------|---------------|
| Market orders | Fill immediately at current price ± slippage |
| Limit orders | Check price condition (buy ≤ limit, sell ≥ limit) |
| Slippage | Configurable percentage applied to fill price |
| Commission | Percentage of trade value |
| Positions | Dict tracking quantity, avg_cost, market_value per symbol |
| Fill delay | Configurable delay before fill (simulates network latency) |
| Price source | DataHandler or manual `set_price()` for testing |
| Fill callbacks | Notify listeners on fill events |

### 3. Order Manager (`src/broker/order_manager.py`)

State machine bridging the event system to broker operations:

```python
VALID_TRANSITIONS = {
    OrderStatus.PENDING: {SUBMITTED, REJECTED},
    OrderStatus.SUBMITTED: {ACCEPTED, REJECTED, CANCELLED, FILLED},
    OrderStatus.ACCEPTED: {FILLED, CANCELLED, PARTIALLY_FILLED},
    OrderStatus.PARTIALLY_FILLED: {FILLED, CANCELLED},
    OrderStatus.FILLED: set(),      # terminal
    OrderStatus.CANCELLED: set(),   # terminal
    OrderStatus.REJECTED: set(),    # terminal
    OrderStatus.EXPIRED: set(),     # terminal
}
```

Key methods:
- `submit_order_event(event)` — converts OrderEvent → BrokerOrder, submits to broker
- `on_fill(fill)` — converts BrokerFill → FillEvent for engine consumption
- `cancel_order(order_id)` — request cancellation
- `active_orders` / `all_orders` — query order state
- `_order_history` — audit trail of all state transitions

## Tests

- **`tests/test_paper_broker.py`** — ~48 tests: market/limit orders, slippage, commission, cancellation, position tracking, account info, fill callbacks
- **`tests/test_order_manager.py`** — ~33 tests: state transitions, submit/cancel/fill, order queries, integration with PaperBroker

## Files changed

| File | Action | What |
|------|--------|------|
| `src/broker/__init__.py` | Created | Package exports |
| `src/broker/base_broker.py` | Created | OrderStatus, BrokerOrder, BrokerFill, BrokerAdapter ABC |
| `src/broker/paper_broker.py` | Created | PaperBroker implementation |
| `src/broker/order_manager.py` | Created | OrderManager state machine |
| `tests/test_paper_broker.py` | Created | ~48 tests |
| `tests/test_order_manager.py` | Created | ~33 tests |

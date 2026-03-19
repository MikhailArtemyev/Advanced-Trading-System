"""Shared test helpers — importable from any test module.

For pytest fixtures, see conftest.py.
"""

from datetime import datetime

import pandas as pd

from src.events.event import OrderEvent, OrderSide, OrderType, SignalEvent, SignalType

# ---------------------------------------------------------------------------
# Fake DataHandler — lightweight stub for strategy/portfolio tests
# ---------------------------------------------------------------------------


class FakeDataHandler:
    """Minimal DataHandler stub that serves canned OHLCV data.

    Usage:
        handler = FakeDataHandler({"AAPL": make_ohlcv([100, 101, 102])})
        bars = handler.get_latest_bars("AAPL", 2)
    """

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_latest_bars(self, symbol: str, n: int) -> pd.DataFrame:
        df = self._data.get(symbol, pd.DataFrame())
        return df.tail(n)


# ---------------------------------------------------------------------------
# OHLCV data builders
# ---------------------------------------------------------------------------


def make_ohlcv(close_values: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from close prices.

    open/high/low are derived from close (±1).
    """
    return pd.DataFrame(
        {
            "open": close_values,
            "high": [v + 1 for v in close_values],
            "low": [v - 1 for v in close_values],
            "close": close_values,
            "volume": [1000] * len(close_values),
        }
    )


def make_ohlcv_trend(start: float, n: int, trend: float = 0.0) -> pd.DataFrame:
    """Build an OHLCV DataFrame with a linear trend."""
    closes = [start + i * trend for i in range(n)]
    return make_ohlcv(closes)


# ---------------------------------------------------------------------------
# Event factories
# ---------------------------------------------------------------------------


def make_signal(
    symbol: str = "AAPL",
    signal_type: SignalType = SignalType.LONG,
    strength: float = 1.0,
    timestamp: datetime | None = None,
) -> SignalEvent:
    return SignalEvent(
        timestamp=timestamp or datetime.now(),
        symbol=symbol,
        signal_type=signal_type,
        strength=strength,
    )


def make_order(
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    quantity: int = 100,
    order_type: OrderType = OrderType.MARKET,
    timestamp: datetime | None = None,
) -> OrderEvent:
    return OrderEvent(
        timestamp=timestamp or datetime.now(),
        symbol=symbol,
        order_type=order_type,
        side=side,
        quantity=quantity,
    )

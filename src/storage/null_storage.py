"""No-op storage backend for tests and backtests without a database."""

import uuid
from datetime import datetime
from typing import Any

import pandas as pd

from .backend import StorageBackend


class NullStorage(StorageBackend):
    """Storage backend that silently discards all writes.

    Reads return empty results. Used when no database is configured.
    """

    def create_session(
        self,
        session_type: str,
        config: dict[str, Any],
        initial_capital: float,
    ) -> str:
        return str(uuid.uuid4())

    def end_session(self, session_id: str, final_equity: float) -> None:
        pass

    def record_trade(self, session_id: str, trade: Any) -> None:
        pass

    def record_equity_snapshot(self, session_id: str, snapshot: dict[str, Any]) -> None:
        pass

    def record_order(self, session_id: str, order_data: dict[str, Any]) -> None:
        pass

    def save_engine_state(self, session_id: str, state: dict[str, Any]) -> None:
        pass

    def load_engine_state(self, session_id: str) -> dict[str, Any] | None:
        return None

    def get_trades(
        self,
        session_id: str,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        return []

    def get_equity_curve(self, session_id: str) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "equity",
                "cash",
                "positions_value",
                "num_positions",
            ]
        )

    def get_orders(
        self,
        session_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        return []

    def list_sessions(self) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        pass

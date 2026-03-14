"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd


class StorageBackend(ABC):
    """Interface for persisting trades, equity, orders, and engine state."""

    @abstractmethod
    def create_session(
        self,
        session_type: str,
        config: dict[str, Any],
        initial_capital: float,
    ) -> str:
        """Create a new trading session. Returns session ID."""

    @abstractmethod
    def end_session(self, session_id: str, final_equity: float) -> None:
        """Mark a session as ended with final equity."""

    @abstractmethod
    def record_trade(self, session_id: str, trade: Any) -> None:
        """Record a completed trade."""

    @abstractmethod
    def record_equity_snapshot(self, session_id: str, snapshot: dict[str, Any]) -> None:
        """Record a point-in-time equity snapshot."""

    @abstractmethod
    def record_order(self, session_id: str, order_data: dict[str, Any]) -> None:
        """Record an order."""

    @abstractmethod
    def save_engine_state(self, session_id: str, state: dict[str, Any]) -> None:
        """Save engine state for crash recovery."""

    @abstractmethod
    def load_engine_state(self, session_id: str) -> dict[str, Any] | None:
        """Load the latest engine state for a session. Returns None if none."""

    @abstractmethod
    def get_trades(
        self,
        session_id: str,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Query trades with optional filters."""

    @abstractmethod
    def get_equity_curve(self, session_id: str) -> pd.DataFrame:
        """Get equity curve as a DataFrame for a session."""

    @abstractmethod
    def get_orders(
        self,
        session_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query orders with optional status filter."""

    @abstractmethod
    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with summary info."""

    @abstractmethod
    def close(self) -> None:
        """Close the storage backend and release resources."""

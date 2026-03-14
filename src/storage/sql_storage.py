"""SQL implementation of the StorageBackend (SQLite, PostgreSQL, etc.)."""

import json
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, desc, select
from sqlalchemy.orm import Session

from .backend import StorageBackend
from .models import (
    Base,
    EngineStateRecord,
    EquitySnapshot,
    OrderRecord,
    SessionRecord,
    TradeRecord,
)


class SQLStorage(StorageBackend):
    """SQLAlchemy-backed storage supporting SQLite, PostgreSQL, and others.

    Args:
        db_url: SQLAlchemy database URL.
               SQLite:      ``sqlite:///trading.db``
               PostgreSQL:  ``postgresql://user:pass@host/dbname``
    """

    def __init__(self, db_url: str = "sqlite:///trading.db") -> None:
        kwargs: dict[str, Any] = {}
        if db_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self._engine = create_engine(db_url, **kwargs)
        Base.metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_type: str,
        config: dict[str, Any],
        initial_capital: float,
    ) -> str:
        session_id = str(uuid.uuid4())
        with Session(self._engine) as db:
            record = SessionRecord(
                id=session_id,
                session_type=session_type,
                started_at=datetime.now(),
                config_json=json.dumps(config, default=str),
                initial_capital=initial_capital,
            )
            db.add(record)
            db.commit()
        return session_id

    def end_session(self, session_id: str, final_equity: float) -> None:
        with Session(self._engine) as db:
            record = db.get(SessionRecord, session_id)
            if record is not None:
                record.ended_at = datetime.now()
                record.final_equity = final_equity
                db.commit()

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def record_trade(self, session_id: str, trade: Any) -> None:
        with Session(self._engine) as db:
            record = TradeRecord(
                session_id=session_id,
                timestamp=trade.timestamp,
                symbol=trade.symbol,
                side=trade.side,
                quantity=trade.quantity,
                price=trade.price,
                commission=trade.commission,
                pnl=trade.pnl,
            )
            db.add(record)
            db.commit()

    def get_trades(
        self,
        session_id: str,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        with Session(self._engine) as db:
            stmt = select(TradeRecord).where(TradeRecord.session_id == session_id)
            if symbol is not None:
                stmt = stmt.where(TradeRecord.symbol == symbol)
            if start is not None:
                stmt = stmt.where(TradeRecord.timestamp >= start)
            if end is not None:
                stmt = stmt.where(TradeRecord.timestamp <= end)
            stmt = stmt.order_by(TradeRecord.timestamp)

            rows = db.execute(stmt).scalars().all()
            return [
                {
                    "timestamp": r.timestamp,
                    "symbol": r.symbol,
                    "side": r.side,
                    "quantity": r.quantity,
                    "price": r.price,
                    "commission": r.commission,
                    "pnl": r.pnl,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Equity snapshots
    # ------------------------------------------------------------------

    def record_equity_snapshot(self, session_id: str, snapshot: dict[str, Any]) -> None:
        with Session(self._engine) as db:
            record = EquitySnapshot(
                session_id=session_id,
                timestamp=snapshot["timestamp"],
                equity=snapshot["equity"],
                cash=snapshot["cash"],
                positions_value=snapshot["positions_value"],
                num_positions=snapshot["num_positions"],
            )
            db.add(record)
            db.commit()

    def get_equity_curve(self, session_id: str) -> pd.DataFrame:
        with Session(self._engine) as db:
            stmt = (
                select(EquitySnapshot)
                .where(EquitySnapshot.session_id == session_id)
                .order_by(EquitySnapshot.timestamp)
            )
            rows = db.execute(stmt).scalars().all()

            if not rows:
                return pd.DataFrame(
                    columns=[
                        "timestamp",
                        "equity",
                        "cash",
                        "positions_value",
                        "num_positions",
                    ]
                )

            return pd.DataFrame(
                [
                    {
                        "timestamp": r.timestamp,
                        "equity": r.equity,
                        "cash": r.cash,
                        "positions_value": r.positions_value,
                        "num_positions": r.num_positions,
                    }
                    for r in rows
                ]
            )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def record_order(self, session_id: str, order_data: dict[str, Any]) -> None:
        with Session(self._engine) as db:
            record = OrderRecord(
                session_id=session_id,
                order_id=order_data["order_id"],
                timestamp=order_data["timestamp"],
                symbol=order_data["symbol"],
                side=order_data["side"],
                order_type=order_data["order_type"],
                quantity=order_data["quantity"],
                status=order_data["status"],
                filled_quantity=order_data.get("filled_quantity", 0),
                filled_avg_price=order_data.get("filled_avg_price", 0.0),
            )
            db.add(record)
            db.commit()

    def get_orders(
        self,
        session_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        with Session(self._engine) as db:
            stmt = select(OrderRecord).where(OrderRecord.session_id == session_id)
            if status is not None:
                stmt = stmt.where(OrderRecord.status == status)
            stmt = stmt.order_by(OrderRecord.timestamp)

            rows = db.execute(stmt).scalars().all()
            return [
                {
                    "order_id": r.order_id,
                    "timestamp": r.timestamp,
                    "symbol": r.symbol,
                    "side": r.side,
                    "order_type": r.order_type,
                    "quantity": r.quantity,
                    "status": r.status,
                    "filled_quantity": r.filled_quantity,
                    "filled_avg_price": r.filled_avg_price,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Engine state (crash recovery)
    # ------------------------------------------------------------------

    def save_engine_state(self, session_id: str, state: dict[str, Any]) -> None:
        stats = state.get("statistics", {})
        positions = stats.get("positions", {})

        with Session(self._engine) as db:
            record = EngineStateRecord(
                session_id=session_id,
                saved_at=datetime.now(),
                cash=state.get("cash", 0.0),
                positions_json=json.dumps(positions, default=str),
                bars_processed=stats.get("bars_processed", 0),
                events_processed=stats.get("events_processed", 0),
                orders_submitted=stats.get("orders_submitted", 0),
                orders_rejected=stats.get("orders_rejected", 0),
                fills_processed=stats.get("fills_processed", 0),
            )
            db.add(record)
            db.commit()

    def load_engine_state(self, session_id: str) -> dict[str, Any] | None:
        with Session(self._engine) as db:
            stmt = (
                select(EngineStateRecord)
                .where(EngineStateRecord.session_id == session_id)
                .order_by(desc(EngineStateRecord.saved_at))
                .limit(1)
            )
            record = db.execute(stmt).scalar_one_or_none()

            if record is None:
                return None

            positions = json.loads(record.positions_json)
            return {
                "cash": record.cash,
                "statistics": {
                    "positions": positions,
                    "bars_processed": record.bars_processed,
                    "events_processed": record.events_processed,
                    "orders_submitted": record.orders_submitted,
                    "orders_rejected": record.orders_rejected,
                    "fills_processed": record.fills_processed,
                },
            }

    # ------------------------------------------------------------------
    # Session queries
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict[str, Any]]:
        with Session(self._engine) as db:
            stmt = select(SessionRecord).order_by(desc(SessionRecord.started_at))
            rows = db.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "session_type": r.session_type,
                    "started_at": r.started_at,
                    "ended_at": r.ended_at,
                    "initial_capital": r.initial_capital,
                    "final_equity": r.final_equity,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Bulk operations (for backtest batching)
    # ------------------------------------------------------------------

    def bulk_record_trades(self, session_id: str, trades: list[Any]) -> None:
        """Insert many trades in a single transaction."""
        with Session(self._engine) as db:
            records = [
                TradeRecord(
                    session_id=session_id,
                    timestamp=t.timestamp,
                    symbol=t.symbol,
                    side=t.side,
                    quantity=t.quantity,
                    price=t.price,
                    commission=t.commission,
                    pnl=t.pnl,
                )
                for t in trades
            ]
            db.add_all(records)
            db.commit()

    def bulk_record_equity(
        self, session_id: str, snapshots: list[dict[str, Any]]
    ) -> None:
        """Insert many equity snapshots in a single transaction."""
        with Session(self._engine) as db:
            records = [
                EquitySnapshot(
                    session_id=session_id,
                    timestamp=s["timestamp"],
                    equity=s["equity"],
                    cash=s["cash"],
                    positions_value=s["positions_value"],
                    num_positions=s["num_positions"],
                )
                for s in snapshots
            ]
            db.add_all(records)
            db.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._engine.dispose()

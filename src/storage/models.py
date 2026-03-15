"""SQLAlchemy ORM models for the trading system persistence layer."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class SessionRecord(Base):
    """A trading session (backtest or paper trading run)."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_type: Mapped[str] = mapped_column(String(20))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    initial_capital: Mapped[float] = mapped_column(Float)
    final_equity: Mapped[float | None] = mapped_column(Float, nullable=True)


class TradeRecord(Base):
    """A completed trade (fill)."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)


class EquitySnapshot(Base):
    """A point-in-time equity snapshot."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    num_positions: Mapped[int] = mapped_column(Integer)


class OrderRecord(Base):
    """An order submitted to the broker."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    order_id: Mapped[str] = mapped_column(String(36), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    symbol: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(4))
    order_type: Mapped[str] = mapped_column(String(10))
    quantity: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    filled_avg_price: Mapped[float] = mapped_column(Float, default=0.0)


class EngineStateRecord(Base):
    """Periodic engine state snapshot for crash recovery."""

    __tablename__ = "engine_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    saved_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_json: Mapped[str] = mapped_column(Text, default="{}")
    bars_processed: Mapped[int] = mapped_column(Integer, default=0)
    events_processed: Mapped[int] = mapped_column(Integer, default=0)
    orders_submitted: Mapped[int] = mapped_column(Integer, default=0)
    orders_rejected: Mapped[int] = mapped_column(Integer, default=0)
    fills_processed: Mapped[int] = mapped_column(Integer, default=0)

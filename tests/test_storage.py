"""Tests for the SQL storage layer.

Covers SQLStorage (full CRUD), NullStorage (no-op), and
integration with Portfolio and BacktestEngine.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from src.storage.null_storage import NullStorage
from src.storage.sql_storage import SQLStorage

# ── Helpers ──────────────────────────────────────────────────────────


@dataclass
class FakeTrade:
    """Minimal trade-like object for testing record_trade."""

    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    pnl: float = 0.0


def _make_snapshot(
    ts: datetime,
    equity: float = 100_000.0,
    cash: float = 90_000.0,
) -> dict:
    return {
        "timestamp": ts,
        "equity": equity,
        "cash": cash,
        "positions_value": equity - cash,
        "num_positions": 1,
    }


# ── SQLStorage Tests ─────────────────────────────────────────────


class TestSessionLifecycle:
    def test_create_session(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {"symbols": ["AAPL"]}, 100_000.0)
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID length
        storage.close()

    def test_end_session(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("paper", {}, 100_000.0)
        storage.end_session(sid, 105_000.0)
        sessions = storage.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["final_equity"] == 105_000.0
        assert sessions[0]["ended_at"] is not None
        storage.close()

    def test_list_sessions(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        storage.create_session("backtest", {}, 100_000.0)
        storage.create_session("paper", {}, 50_000.0)
        sessions = storage.list_sessions()
        assert len(sessions) == 2
        storage.close()

    def test_list_sessions_empty(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        assert storage.list_sessions() == []
        storage.close()


class TestTrades:
    def test_record_and_get(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {}, 100_000.0)
        trade = FakeTrade(
            timestamp=datetime(2025, 1, 1, 10, 0),
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            commission=15.0,
            pnl=0.0,
        )
        storage.record_trade(sid, trade)
        trades = storage.get_trades(sid)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "AAPL"
        assert trades[0]["quantity"] == 100
        assert trades[0]["price"] == 150.0
        storage.close()

    def test_get_by_symbol(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {}, 100_000.0)
        for sym in ["AAPL", "MSFT", "AAPL"]:
            storage.record_trade(
                sid,
                FakeTrade(
                    timestamp=datetime(2025, 1, 1),
                    symbol=sym,
                    side="BUY",
                    quantity=10,
                    price=100.0,
                    commission=1.0,
                ),
            )
        aapl_trades = storage.get_trades(sid, symbol="AAPL")
        assert len(aapl_trades) == 2
        msft_trades = storage.get_trades(sid, symbol="MSFT")
        assert len(msft_trades) == 1
        storage.close()

    def test_get_by_date_range(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {}, 100_000.0)
        for day in range(1, 6):
            storage.record_trade(
                sid,
                FakeTrade(
                    timestamp=datetime(2025, 1, day),
                    symbol="AAPL",
                    side="BUY",
                    quantity=10,
                    price=100.0,
                    commission=1.0,
                ),
            )
        trades = storage.get_trades(
            sid, start=datetime(2025, 1, 2), end=datetime(2025, 1, 4)
        )
        assert len(trades) == 3
        storage.close()

    def test_get_empty(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {}, 100_000.0)
        assert storage.get_trades(sid) == []
        storage.close()

    def test_bulk_record_trades(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {}, 100_000.0)
        trades = [
            FakeTrade(
                timestamp=datetime(2025, 1, i),
                symbol="AAPL",
                side="BUY",
                quantity=10 * i,
                price=100.0 + i,
                commission=1.0,
            )
            for i in range(1, 11)
        ]
        storage.bulk_record_trades(sid, trades)
        loaded = storage.get_trades(sid)
        assert len(loaded) == 10
        storage.close()


class TestEquitySnapshots:
    def test_record_and_get_curve(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {}, 100_000.0)
        for i in range(5):
            storage.record_equity_snapshot(
                sid,
                _make_snapshot(
                    datetime(2025, 1, 1) + timedelta(days=i),
                    equity=100_000.0 + i * 1000,
                ),
            )
        curve = storage.get_equity_curve(sid)
        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == 5
        assert list(curve.columns) == [
            "timestamp",
            "equity",
            "cash",
            "positions_value",
            "num_positions",
        ]
        storage.close()

    def test_get_empty_curve(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {}, 100_000.0)
        curve = storage.get_equity_curve(sid)
        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == 0
        storage.close()

    def test_bulk_record_equity(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {}, 100_000.0)
        snapshots = [
            _make_snapshot(datetime(2025, 1, 1) + timedelta(days=i)) for i in range(20)
        ]
        storage.bulk_record_equity(sid, snapshots)
        curve = storage.get_equity_curve(sid)
        assert len(curve) == 20
        storage.close()


class TestOrders:
    def test_record_and_get(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("paper", {}, 100_000.0)
        storage.record_order(
            sid,
            {
                "order_id": "ord-001",
                "timestamp": datetime(2025, 1, 1),
                "symbol": "AAPL",
                "side": "BUY",
                "order_type": "MARKET",
                "quantity": 100,
                "status": "FILLED",
                "filled_quantity": 100,
                "filled_avg_price": 150.0,
            },
        )
        orders = storage.get_orders(sid)
        assert len(orders) == 1
        assert orders[0]["order_id"] == "ord-001"
        assert orders[0]["status"] == "FILLED"
        storage.close()

    def test_get_by_status(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("paper", {}, 100_000.0)
        for status in ["FILLED", "REJECTED", "FILLED"]:
            storage.record_order(
                sid,
                {
                    "order_id": f"ord-{status}",
                    "timestamp": datetime(2025, 1, 1),
                    "symbol": "AAPL",
                    "side": "BUY",
                    "order_type": "MARKET",
                    "quantity": 10,
                    "status": status,
                },
            )
        filled = storage.get_orders(sid, status="FILLED")
        assert len(filled) == 2
        rejected = storage.get_orders(sid, status="REJECTED")
        assert len(rejected) == 1
        storage.close()


class TestEngineState:
    def test_save_and_load(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("paper", {}, 100_000.0)
        state = {
            "cash": 95_000.0,
            "statistics": {
                "positions": {"AAPL": {"quantity": 50, "avg_cost": 100.0}},
                "bars_processed": 120,
                "events_processed": 200,
                "orders_submitted": 5,
                "orders_rejected": 1,
                "fills_processed": 4,
            },
        }
        storage.save_engine_state(sid, state)
        loaded = storage.load_engine_state(sid)
        assert loaded is not None
        assert loaded["cash"] == 95_000.0
        assert loaded["statistics"]["bars_processed"] == 120
        assert loaded["statistics"]["fills_processed"] == 4
        assert loaded["statistics"]["positions"]["AAPL"]["quantity"] == 50
        storage.close()

    def test_load_returns_none_when_empty(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("paper", {}, 100_000.0)
        assert storage.load_engine_state(sid) is None
        storage.close()

    def test_load_returns_latest(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("paper", {}, 100_000.0)
        for i in range(3):
            storage.save_engine_state(
                sid,
                {
                    "cash": float(i * 1000),
                    "statistics": {
                        "positions": {},
                        "bars_processed": i * 10,
                        "events_processed": 0,
                        "orders_submitted": 0,
                        "orders_rejected": 0,
                        "fills_processed": 0,
                    },
                },
            )
        loaded = storage.load_engine_state(sid)
        assert loaded is not None
        assert loaded["cash"] == 2000.0
        assert loaded["statistics"]["bars_processed"] == 20
        storage.close()

    def test_round_trip_positions_json(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("paper", {}, 100_000.0)
        positions = {
            "AAPL": {"quantity": 100, "avg_cost": 150.0, "market_value": 15500.0},
            "MSFT": {"quantity": 50, "avg_cost": 380.0, "market_value": 19500.0},
        }
        storage.save_engine_state(
            sid,
            {
                "cash": 70_000.0,
                "statistics": {
                    "positions": positions,
                    "bars_processed": 0,
                    "events_processed": 0,
                    "orders_submitted": 0,
                    "orders_rejected": 0,
                    "fills_processed": 0,
                },
            },
        )
        loaded = storage.load_engine_state(sid)
        assert loaded is not None
        assert loaded["statistics"]["positions"] == positions
        storage.close()


class TestCrossSessions:
    def test_trades_isolated_by_session(self, tmp_path):
        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        s1 = storage.create_session("backtest", {}, 100_000.0)
        s2 = storage.create_session("backtest", {}, 100_000.0)

        storage.record_trade(
            s1,
            FakeTrade(
                timestamp=datetime(2025, 1, 1),
                symbol="AAPL",
                side="BUY",
                quantity=10,
                price=100.0,
                commission=1.0,
            ),
        )
        storage.record_trade(
            s2,
            FakeTrade(
                timestamp=datetime(2025, 1, 1),
                symbol="MSFT",
                side="BUY",
                quantity=20,
                price=200.0,
                commission=2.0,
            ),
        )

        assert len(storage.get_trades(s1)) == 1
        assert storage.get_trades(s1)[0]["symbol"] == "AAPL"
        assert len(storage.get_trades(s2)) == 1
        assert storage.get_trades(s2)[0]["symbol"] == "MSFT"
        storage.close()


# ── NullStorage Tests ────────────────────────────────────────────────


class TestNullStorage:
    def test_create_session_returns_uuid(self):
        storage = NullStorage()
        sid = storage.create_session("backtest", {}, 100_000.0)
        assert isinstance(sid, str)
        assert len(sid) == 36

    def test_end_session_no_error(self):
        storage = NullStorage()
        storage.end_session("fake-id", 100_000.0)

    def test_record_trade_no_error(self):
        storage = NullStorage()
        storage.record_trade(
            "fake-id",
            FakeTrade(
                timestamp=datetime(2025, 1, 1),
                symbol="AAPL",
                side="BUY",
                quantity=10,
                price=100.0,
                commission=1.0,
            ),
        )

    def test_get_trades_empty(self):
        storage = NullStorage()
        assert storage.get_trades("fake-id") == []

    def test_get_equity_curve_empty(self):
        storage = NullStorage()
        curve = storage.get_equity_curve("fake-id")
        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == 0

    def test_get_orders_empty(self):
        storage = NullStorage()
        assert storage.get_orders("fake-id") == []

    def test_save_engine_state_no_error(self):
        storage = NullStorage()
        storage.save_engine_state("fake-id", {"cash": 100_000.0})

    def test_load_engine_state_none(self):
        storage = NullStorage()
        assert storage.load_engine_state("fake-id") is None

    def test_list_sessions_empty(self):
        storage = NullStorage()
        assert storage.list_sessions() == []

    def test_close_no_error(self):
        storage = NullStorage()
        storage.close()


# ── Integration: Portfolio with Storage ──────────────────────────────


class TestPortfolioStorageIntegration:
    def test_portfolio_records_trades_to_db(self, tmp_path):
        from src.events.event import FillEvent, OrderSide
        from src.portfolio.portfolio import Portfolio

        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {}, 100_000.0)

        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=["AAPL"],
            storage=storage,
            session_id=sid,
        )

        # Simulate a fill
        fill = FillEvent(
            timestamp=datetime(2025, 1, 1, 10, 0),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill(fill)

        trades = storage.get_trades(sid)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "AAPL"
        assert trades[0]["quantity"] == 100
        storage.close()

    def test_portfolio_records_equity_to_db(self, tmp_path):
        from src.portfolio.portfolio import Portfolio

        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        sid = storage.create_session("backtest", {}, 100_000.0)

        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=["AAPL"],
            storage=storage,
            session_id=sid,
        )

        portfolio.update_timeindex(datetime(2025, 1, 1))

        curve = storage.get_equity_curve(sid)
        assert len(curve) == 1
        assert curve["equity"].iloc[0] == 100_000.0
        storage.close()


# ── Integration: BacktestEngine with Storage ─────────────────────────


class TestBacktestStorageIntegration:
    def test_backtest_persists_session(self, tmp_path):
        """Full backtest writes session, trades, and equity to DB."""
        from unittest.mock import MagicMock

        from src.backtest.engine import BacktestEngine
        from src.data.data_handler import DataHandler

        storage = SQLStorage(db_url=f"sqlite:///{tmp_path / 'test.db'}")

        # Minimal mock data handler that produces 3 bars then stops
        data_handler = MagicMock(spec=DataHandler)
        call_count = 0

        def update_bars_side_effect():
            nonlocal call_count
            call_count += 1

        def continue_backtest_side_effect():
            return call_count < 3

        data_handler.update_bars = update_bars_side_effect
        type(data_handler).continue_backtest = property(
            lambda self: continue_backtest_side_effect()
        )
        data_handler.get_current_timestamp.return_value = datetime(2025, 1, 1)

        # Minimal strategy / portfolio / execution mocks
        strategy = MagicMock()
        portfolio = MagicMock()
        portfolio.get_equity.return_value = 100_000.0
        portfolio.get_positions.return_value = {}
        portfolio.get_trade_history.return_value = []
        portfolio.equity_history = []

        execution = MagicMock()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            storage=storage,
        )

        engine.run()

        sessions = storage.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_type"] == "backtest"
        assert sessions[0]["final_equity"] == 100_000.0
        storage.close()

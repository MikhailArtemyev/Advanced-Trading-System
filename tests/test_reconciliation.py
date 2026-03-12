"""Tests for PaperTradingEngine.reconcile_positions()."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.broker.order_manager import OrderManager
from src.engine.paper_engine import PaperTradingEngine
from src.portfolio.portfolio import Portfolio


def _make_engine(portfolio, broker_positions):
    """Create a PaperTradingEngine with mocked broker positions."""
    broker = AsyncMock()
    broker.get_positions = AsyncMock(return_value=broker_positions)

    order_manager = MagicMock(spec=OrderManager)
    order_manager._broker = broker
    order_manager.all_orders = {}

    strategy = MagicMock()
    strategy.set_event_queue = MagicMock()

    data_handler = MagicMock()

    engine = PaperTradingEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        order_manager=order_manager,
    )
    return engine


class TestReconcilePositions:
    @pytest.mark.asyncio
    async def test_matching_positions(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL"])
        portfolio.positions["AAPL"].quantity = 50
        portfolio.positions["AAPL"].avg_cost = 150.0

        broker_positions = {"AAPL": {"quantity": 50, "avg_cost": 150.0}}
        engine = _make_engine(portfolio, broker_positions)

        report = await engine.reconcile_positions()
        assert report["matched"] is True
        assert report["discrepancies"] == []
        assert report["symbols_checked"] == 1

    @pytest.mark.asyncio
    async def test_quantity_mismatch(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL"])
        portfolio.positions["AAPL"].quantity = 50

        broker_positions = {"AAPL": {"quantity": 40, "avg_cost": 150.0}}
        engine = _make_engine(portfolio, broker_positions)

        report = await engine.reconcile_positions()
        assert report["matched"] is False
        assert len(report["discrepancies"]) == 1
        d = report["discrepancies"][0]
        assert d["symbol"] == "AAPL"
        assert d["engine_quantity"] == 50
        assert d["broker_quantity"] == 40
        assert d["difference"] == 10

    @pytest.mark.asyncio
    async def test_symbol_in_engine_not_broker(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL", "MSFT"])
        portfolio.positions["AAPL"].quantity = 50
        portfolio.positions["MSFT"].quantity = 30

        # Broker only knows about AAPL
        broker_positions = {"AAPL": {"quantity": 50}}
        engine = _make_engine(portfolio, broker_positions)

        report = await engine.reconcile_positions()
        assert report["matched"] is False
        # MSFT is in engine (qty 30) but not in broker (qty 0)
        msft_disc = [d for d in report["discrepancies"] if d["symbol"] == "MSFT"]
        assert len(msft_disc) == 1
        assert msft_disc[0]["engine_quantity"] == 30
        assert msft_disc[0]["broker_quantity"] == 0

    @pytest.mark.asyncio
    async def test_symbol_in_broker_not_engine(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL"])

        broker_positions = {
            "AAPL": {"quantity": 0},
            "TSLA": {"quantity": 25},
        }
        engine = _make_engine(portfolio, broker_positions)

        report = await engine.reconcile_positions()
        assert report["matched"] is False
        tsla_disc = [d for d in report["discrepancies"] if d["symbol"] == "TSLA"]
        assert len(tsla_disc) == 1
        assert tsla_disc[0]["broker_quantity"] == 25
        assert tsla_disc[0]["engine_quantity"] == 0

    @pytest.mark.asyncio
    async def test_multiple_symbols_all_match(self):
        portfolio = Portfolio(
            initial_capital=100_000, symbols=["AAPL", "MSFT", "GOOGL"]
        )
        portfolio.positions["AAPL"].quantity = 50
        portfolio.positions["MSFT"].quantity = 30
        # GOOGL stays at 0

        broker_positions = {
            "AAPL": {"quantity": 50},
            "MSFT": {"quantity": 30},
            "GOOGL": {"quantity": 0},
        }
        engine = _make_engine(portfolio, broker_positions)

        report = await engine.reconcile_positions()
        assert report["matched"] is True
        assert report["symbols_checked"] == 3

    @pytest.mark.asyncio
    async def test_empty_positions_match(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL"])
        broker_positions = {"AAPL": {"quantity": 0}}
        engine = _make_engine(portfolio, broker_positions)

        report = await engine.reconcile_positions()
        assert report["matched"] is True

    @pytest.mark.asyncio
    async def test_report_has_timestamp(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL"])
        broker_positions = {"AAPL": {"quantity": 0}}
        engine = _make_engine(portfolio, broker_positions)

        report = await engine.reconcile_positions()
        assert "timestamp" in report
        # Should be parseable ISO timestamp
        datetime.fromisoformat(report["timestamp"])

    @pytest.mark.asyncio
    async def test_multiple_discrepancies(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL", "MSFT"])
        portfolio.positions["AAPL"].quantity = 50
        portfolio.positions["MSFT"].quantity = 30

        broker_positions = {
            "AAPL": {"quantity": 45},
            "MSFT": {"quantity": 20},
        }
        engine = _make_engine(portfolio, broker_positions)

        report = await engine.reconcile_positions()
        assert report["matched"] is False
        assert len(report["discrepancies"]) == 2

    @pytest.mark.asyncio
    async def test_no_broker_positions(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL"])
        portfolio.positions["AAPL"].quantity = 50

        broker_positions = {}
        engine = _make_engine(portfolio, broker_positions)

        report = await engine.reconcile_positions()
        assert report["matched"] is False
        assert report["discrepancies"][0]["broker_quantity"] == 0

    @pytest.mark.asyncio
    async def test_discrepancies_sorted_by_symbol(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["MSFT", "AAPL", "TSLA"])
        portfolio.positions["MSFT"].quantity = 10
        portfolio.positions["AAPL"].quantity = 20
        portfolio.positions["TSLA"].quantity = 30

        broker_positions = {
            "MSFT": {"quantity": 5},
            "AAPL": {"quantity": 15},
            "TSLA": {"quantity": 25},
        }
        engine = _make_engine(portfolio, broker_positions)

        report = await engine.reconcile_positions()
        symbols = [d["symbol"] for d in report["discrepancies"]]
        assert symbols == sorted(symbols)


class TestGetState:
    def test_returns_statistics(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL"])
        broker_positions = {}
        engine = _make_engine(portfolio, broker_positions)

        state = engine.get_state()
        assert "statistics" in state
        assert state["statistics"]["bars_processed"] == 0

    def test_returns_cash(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL"])
        engine = _make_engine(portfolio, {})
        state = engine.get_state()
        assert state["cash"] == 100_000.0

    def test_returns_trades_list(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL"])
        engine = _make_engine(portfolio, {})
        state = engine.get_state()
        assert state["trades"] == []

    def test_returns_orders_dict(self):
        portfolio = Portfolio(initial_capital=100_000, symbols=["AAPL"])
        engine = _make_engine(portfolio, {})
        state = engine.get_state()
        assert state["orders"] == {}

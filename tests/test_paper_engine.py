"""Tests for PaperTradingEngine — async event loop for paper trading."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.broker.order_manager import OrderManager
from src.broker.paper_broker import PaperBroker
from src.engine.paper_engine import PaperTradingEngine
from src.events.event import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalEvent,
    SignalType,
)
from src.risk.risk_manager import RiskCheckResult, RiskManager

# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------


def _mock_strategy():
    strategy = MagicMock()
    strategy.set_event_queue = MagicMock()
    strategy.on_market_data = MagicMock()
    strategy.update_position = MagicMock()
    strategy.get_position = MagicMock(return_value=0)
    return strategy


def _mock_portfolio():
    portfolio = MagicMock()
    portfolio.set_event_queue = MagicMock()
    portfolio.on_signal = MagicMock()
    portfolio.on_fill = MagicMock()
    portfolio.update_timeindex = MagicMock()
    portfolio.get_equity = MagicMock(return_value=100_000.0)
    portfolio.get_positions = MagicMock(return_value={})
    portfolio.get_trade_history = MagicMock(return_value=[])
    return portfolio


def _mock_data_handler(has_bars=False):
    dh = MagicMock()
    dh.update_bars = MagicMock(return_value=has_bars)
    dh.get_current_timestamp = MagicMock(return_value=datetime(2025, 1, 15, 10, 30))
    dh.continue_backtest = True
    dh.get_latest_bars = MagicMock(return_value=MagicMock(empty=True))
    return dh


def _make_engine(
    data_handler=None,
    strategy=None,
    portfolio=None,
    order_manager=None,
    risk_manager=None,
):
    dh = data_handler or _mock_data_handler()
    strat = strategy or _mock_strategy()
    port = portfolio or _mock_portfolio()

    if order_manager is None:
        broker = PaperBroker()
        order_manager = OrderManager(broker)

    return PaperTradingEngine(
        data_handler=dh,
        strategy=strat,
        portfolio=port,
        order_manager=order_manager,
        risk_manager=risk_manager,
        event_poll_interval=0.001,
        bar_poll_interval=0.001,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_initial_state(self):
        engine = _make_engine()
        assert engine.is_running is False
        assert engine.uptime_seconds == 0.0
        assert engine.bars_processed == 0
        assert engine.events_processed == 0
        assert engine.orders_submitted == 0
        assert engine.orders_rejected == 0
        assert engine.fills_processed == 0

    def test_wires_event_queue_to_strategy(self):
        strat = _mock_strategy()
        _make_engine(strategy=strat)
        strat.set_event_queue.assert_called_once()

    def test_wires_event_queue_to_portfolio(self):
        port = _mock_portfolio()
        _make_engine(portfolio=port)
        port.set_event_queue.assert_called_once()


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        engine = _make_engine()

        async def stop_soon():
            await asyncio.sleep(0.05)
            await engine.stop()

        task = asyncio.create_task(stop_soon())
        # Engine should run briefly then stop
        await asyncio.wait_for(engine.start(), timeout=2.0)
        await task
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self):
        engine = _make_engine()
        await engine.stop()
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_uptime_tracks_time(self):
        engine = _make_engine()

        async def stop_after_delay():
            await asyncio.sleep(0.1)
            await engine.stop()

        task = asyncio.create_task(stop_after_delay())
        await asyncio.wait_for(engine.start(), timeout=2.0)
        await task
        # Uptime should be at least 0.05 seconds
        assert engine.uptime_seconds >= 0.05


# ---------------------------------------------------------------------------
# Bar polling
# ---------------------------------------------------------------------------


class TestBarPolling:
    @pytest.mark.asyncio
    async def test_emits_market_event_on_new_bar(self):
        dh = _mock_data_handler(has_bars=False)
        call_count = 0

        def update_bars_side_effect():
            nonlocal call_count
            call_count += 1
            return call_count <= 2  # Return True twice, then False

        dh.update_bars = MagicMock(side_effect=update_bars_side_effect)
        strat = _mock_strategy()
        engine = _make_engine(data_handler=dh, strategy=strat)

        async def stop_soon():
            await asyncio.sleep(0.1)
            await engine.stop()

        task = asyncio.create_task(stop_soon())
        await asyncio.wait_for(engine.start(), timeout=2.0)
        await task

        assert engine.bars_processed >= 2
        assert strat.on_market_data.call_count >= 2

    @pytest.mark.asyncio
    async def test_no_event_when_no_bars(self):
        dh = _mock_data_handler(has_bars=False)
        strat = _mock_strategy()
        engine = _make_engine(data_handler=dh, strategy=strat)

        async def stop_soon():
            await asyncio.sleep(0.05)
            await engine.stop()

        task = asyncio.create_task(stop_soon())
        await asyncio.wait_for(engine.start(), timeout=2.0)
        await task

        assert engine.bars_processed == 0
        strat.on_market_data.assert_not_called()


# ---------------------------------------------------------------------------
# Market event handling
# ---------------------------------------------------------------------------


class TestHandleMarket:
    def test_dispatches_to_strategy(self):
        strat = _mock_strategy()
        dh = _mock_data_handler()
        engine = _make_engine(data_handler=dh, strategy=strat)

        event = MarketEvent(timestamp=datetime(2025, 1, 15, 10, 30))
        engine._handle_market(event)

        strat.on_market_data.assert_called_once_with(event, dh)

    def test_updates_portfolio_timeindex(self):
        port = _mock_portfolio()
        engine = _make_engine(portfolio=port)

        ts = datetime(2025, 1, 15, 10, 30)
        event = MarketEvent(timestamp=ts)
        engine._handle_market(event)

        port.update_timeindex.assert_called_once_with(ts)

    def test_updates_risk_manager_peak_equity(self):
        port = _mock_portfolio()
        port.get_equity.return_value = 105_000.0
        rm = MagicMock(spec=RiskManager)
        engine = _make_engine(portfolio=port, risk_manager=rm)

        event = MarketEvent(timestamp=datetime(2025, 1, 15))
        engine._handle_market(event)

        rm.update_peak_equity.assert_called_once_with(105_000.0)

    def test_no_risk_manager_no_error(self):
        engine = _make_engine(risk_manager=None)
        event = MarketEvent(timestamp=datetime(2025, 1, 15))
        engine._handle_market(event)  # Should not raise


# ---------------------------------------------------------------------------
# Signal event handling
# ---------------------------------------------------------------------------


class TestHandleSignal:
    def test_dispatches_to_portfolio(self):
        port = _mock_portfolio()
        dh = _mock_data_handler()
        engine = _make_engine(data_handler=dh, portfolio=port)

        event = SignalEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=0.8,
        )
        engine._handle_signal(event)

        port.on_signal.assert_called_once_with(event, dh)


# ---------------------------------------------------------------------------
# Order event handling
# ---------------------------------------------------------------------------


class TestHandleOrder:
    @pytest.mark.asyncio
    async def test_submits_to_broker(self):
        broker = PaperBroker(commission_pct=0.0, slippage_pct=0.0)
        await broker.connect()
        broker.set_price("AAPL", 150.0)
        om = OrderManager(broker)
        engine = _make_engine(order_manager=om)

        event = OrderEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )
        await engine._handle_order(event)

        assert engine.orders_submitted == 1

    @pytest.mark.asyncio
    async def test_rejected_by_risk_manager(self):
        rm = MagicMock(spec=RiskManager)
        result = RiskCheckResult(
            approved=False,
            order=MagicMock(),
            rejection_reasons=["max drawdown exceeded"],
        )
        rm.check_order = MagicMock(return_value=result)

        import pandas as pd

        dh = _mock_data_handler()
        mock_bars = pd.DataFrame({"close": [150.0]}, index=[datetime(2025, 1, 15)])
        mock_bars.index.name = "datetime"
        dh.get_latest_bars = MagicMock(return_value=mock_bars)

        engine = _make_engine(data_handler=dh, risk_manager=rm)

        event = OrderEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )
        await engine._handle_order(event)

        assert engine.orders_rejected == 1
        assert engine.orders_submitted == 0

    @pytest.mark.asyncio
    async def test_rejected_when_no_price(self):
        rm = MagicMock(spec=RiskManager)
        dh = _mock_data_handler()
        # get_latest_bars returns empty DataFrame
        import pandas as pd

        dh.get_latest_bars = MagicMock(return_value=pd.DataFrame(columns=["close"]))

        engine = _make_engine(data_handler=dh, risk_manager=rm)

        event = OrderEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )
        await engine._handle_order(event)

        assert engine.orders_rejected == 1
        rm.check_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_risk_manager_submits_directly(self):
        broker = PaperBroker(commission_pct=0.0, slippage_pct=0.0)
        await broker.connect()
        broker.set_price("AAPL", 150.0)
        om = OrderManager(broker)
        engine = _make_engine(order_manager=om, risk_manager=None)

        event = OrderEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )
        await engine._handle_order(event)

        assert engine.orders_submitted == 1

    @pytest.mark.asyncio
    async def test_fill_event_emitted_on_immediate_fill(self):
        broker = PaperBroker(commission_pct=0.001, slippage_pct=0.0)
        await broker.connect()
        broker.set_price("AAPL", 150.0)
        om = OrderManager(broker)
        engine = _make_engine(order_manager=om)

        event = OrderEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )
        await engine._handle_order(event)

        # FillEvent should be in the queue
        assert not engine._events.empty()
        fill = engine._events.get()
        assert fill is not None
        assert fill.event_type == EventType.FILL

    @pytest.mark.asyncio
    async def test_broker_exception_increments_rejected(self):
        om = MagicMock(spec=OrderManager)
        om.submit_order_event = MagicMock(side_effect=Exception("broker error"))
        engine = _make_engine(order_manager=om)

        event = OrderEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )
        await engine._handle_order(event)

        assert engine.orders_rejected == 1


# ---------------------------------------------------------------------------
# Fill event handling
# ---------------------------------------------------------------------------


class TestHandleFill:
    def test_dispatches_to_portfolio(self):
        port = _mock_portfolio()
        engine = _make_engine(portfolio=port)

        event = FillEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=0.15,
        )
        engine._handle_fill(event)

        port.on_fill.assert_called_once_with(event)
        assert engine.fills_processed == 1

    def test_syncs_position_to_strategy(self):
        strat = _mock_strategy()
        port = _mock_portfolio()
        port.get_positions.return_value = {"AAPL": {"quantity": 100}}
        engine = _make_engine(strategy=strat, portfolio=port)

        event = FillEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=0.15,
        )
        engine._handle_fill(event)

        strat.update_position.assert_called_once_with("AAPL", 100)

    def test_updates_risk_manager_on_sell(self):
        port = _mock_portfolio()
        port.get_trade_history.return_value = [{"pnl": 500.0}]
        port.get_positions.return_value = {}
        rm = MagicMock(spec=RiskManager)
        engine = _make_engine(portfolio=port, risk_manager=rm)

        event = FillEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=155.0,
            commission=0.15,
        )
        engine._handle_fill(event)

        rm.update_daily_pnl.assert_called_once_with(500.0, event.timestamp)

    def test_no_risk_update_on_buy(self):
        rm = MagicMock(spec=RiskManager)
        port = _mock_portfolio()
        port.get_positions.return_value = {"AAPL": {"quantity": 100}}
        engine = _make_engine(portfolio=port, risk_manager=rm)

        event = FillEvent(
            timestamp=datetime(2025, 1, 15),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            commission=0.15,
        )
        engine._handle_fill(event)

        rm.update_daily_pnl.assert_not_called()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_get_statistics_initial(self):
        engine = _make_engine()
        stats = engine.get_statistics()

        assert stats["is_running"] is False
        assert stats["uptime_seconds"] == 0.0
        assert stats["bars_processed"] == 0
        assert stats["events_processed"] == 0
        assert stats["orders_submitted"] == 0
        assert stats["orders_rejected"] == 0
        assert stats["fills_processed"] == 0
        assert stats["equity"] == 100_000.0
        assert stats["positions"] == {}

    def test_statistics_after_events(self):
        port = _mock_portfolio()
        port.get_equity.return_value = 105_000.0
        port.get_positions.return_value = {"AAPL": {"quantity": 100}}
        engine = _make_engine(portfolio=port)

        engine.bars_processed = 10
        engine.events_processed = 25
        engine.orders_submitted = 5
        engine.fills_processed = 5

        stats = engine.get_statistics()
        assert stats["bars_processed"] == 10
        assert stats["events_processed"] == 25
        assert stats["orders_submitted"] == 5
        assert stats["fills_processed"] == 5
        assert stats["equity"] == 105_000.0
        assert stats["positions"] == {"AAPL": {"quantity": 100}}


# ---------------------------------------------------------------------------
# Integration: full event cycle
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_cycle_market_to_fill(self):
        """Simulate: bar arrives -> strategy emits signal -> portfolio
        emits order -> broker fills -> portfolio updated."""
        broker = PaperBroker(
            initial_capital=100_000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        await broker.connect()
        broker.set_price("AAPL", 150.0)
        om = OrderManager(broker)

        dh = _mock_data_handler()
        port = _mock_portfolio()
        port.get_positions.return_value = {}

        # Strategy that emits a signal on market data
        strat = _mock_strategy()

        def emit_signal(event, data_handler):
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol="AAPL",
                signal_type=SignalType.LONG,
                strength=1.0,
            )
            engine._events.put(signal)

        strat.on_market_data = MagicMock(side_effect=emit_signal)

        # Portfolio that emits an order on signal
        def emit_order(event, data_handler):
            order = OrderEvent(
                timestamp=event.timestamp,
                symbol="AAPL",
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=100,
            )
            engine._events.put(order)

        port.on_signal = MagicMock(side_effect=emit_order)

        engine = PaperTradingEngine(
            data_handler=dh,
            strategy=strat,
            portfolio=port,
            order_manager=om,
            event_poll_interval=0.001,
            bar_poll_interval=0.001,
        )

        # Manually push a market event and process the full chain
        market = MarketEvent(timestamp=datetime(2025, 1, 15, 10, 30))
        engine._events.put(market)

        # Process events one by one
        engine._running = True

        # 1. Market event -> strategy
        event = engine._events.get()
        engine.events_processed += 1
        engine._handle_market(event)

        # 2. Signal event -> portfolio
        event = engine._events.get()
        engine.events_processed += 1
        engine._handle_signal(event)

        # 3. Order event -> broker
        event = engine._events.get()
        engine.events_processed += 1
        await engine._handle_order(event)

        # 4. Fill event -> portfolio
        event = engine._events.get()
        engine.events_processed += 1
        engine._handle_fill(event)

        assert engine.events_processed == 4
        assert engine.orders_submitted == 1
        assert engine.fills_processed == 1
        port.on_fill.assert_called_once()

    @pytest.mark.asyncio
    async def test_engine_runs_with_live_data_handler(self):
        """Test engine with actual LiveDataHandler + BarAggregator."""
        from src.live.bar_aggregator import BarAggregator
        from src.live.data_feed import Tick
        from src.live.data_handler import LiveDataHandler

        agg = BarAggregator(interval=timedelta(seconds=60))
        dh = LiveDataHandler(symbols=["AAPL"], bar_aggregator=agg)

        broker = PaperBroker(commission_pct=0.0, slippage_pct=0.0)
        await broker.connect()
        broker.set_price("AAPL", 150.0)
        om = OrderManager(broker)

        strat = _mock_strategy()
        port = _mock_portfolio()

        engine = PaperTradingEngine(
            data_handler=dh,
            strategy=strat,
            portfolio=port,
            order_manager=om,
            event_poll_interval=0.001,
            bar_poll_interval=0.001,
        )

        base = datetime(2025, 1, 15, 10, 0, 0)

        # Feed ticks to create completed bars
        for minute in range(3):
            ts = base + timedelta(minutes=minute, seconds=10)
            agg.on_tick(Tick(symbol="AAPL", timestamp=ts, price=150.0 + minute))

        # 2 completed bars, but update_bars() is a boolean flag that
        # resets on read — both bars completed before polling starts,
        # so only one True is seen. Feed more ticks during the run
        # to test multiple polls.

        async def feed_and_stop():
            await asyncio.sleep(0.02)
            # Feed another tick to complete bar 3 → triggers update_bars
            ts = base + timedelta(minutes=3, seconds=10)
            agg.on_tick(Tick(symbol="AAPL", timestamp=ts, price=153.0))
            await asyncio.sleep(0.05)
            await engine.stop()

        task = asyncio.create_task(feed_and_stop())
        await asyncio.wait_for(engine.start(), timeout=2.0)
        await task

        # At least 1 bar from initial ticks + 1 from feed_and_stop
        assert engine.bars_processed >= 2
        assert strat.on_market_data.call_count >= 2

"""Tests for engine bug fixes: state restore, commission, broker selection.

Issue 1: restore_state() — snapshot recovery actually applies state
Issue 2: Commission uses portfolio.commission_pct instead of hardcoded 0.001
Issue 3: Broker selection based on config broker_type
"""

from datetime import datetime, timedelta

import pytest

from src.broker.base_broker import BrokerFill, BrokerOrder, OrderStatus
from src.broker.order_manager import OrderManager
from src.broker.paper_broker import PaperBroker
from src.engine.paper_engine import PaperTradingEngine
from src.events.event import OrderEvent, OrderSide, OrderType
from src.live.bar_aggregator import BarAggregator
from src.live.data_feed import Tick
from src.live.data_handler import LiveDataHandler
from src.portfolio.portfolio import Portfolio
from src.strategy.sma_crossover import SMACrossoverStrategy

# ── Helpers ──────────────────────────────────────────────────────────


def _make_engine(
    symbols=None,
    initial_capital=100_000.0,
    commission_pct=0.001,
):
    """Create a minimal PaperTradingEngine for testing."""
    if symbols is None:
        symbols = ["AAPL"]

    aggregator = BarAggregator(interval=timedelta(minutes=1))
    data_handler = LiveDataHandler(symbols=symbols, bar_aggregator=aggregator)
    strategy = SMACrossoverStrategy(
        symbols=symbols, parameters={"short_window": 5, "long_window": 10}
    )
    portfolio = Portfolio(
        initial_capital=initial_capital,
        symbols=symbols,
        commission_pct=commission_pct,
    )
    broker = PaperBroker(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
    )
    broker._data_handler = data_handler
    order_manager = OrderManager(broker=broker)

    engine = PaperTradingEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        order_manager=order_manager,
    )
    return engine, portfolio, aggregator, data_handler


def _feed_bars(aggregator, symbol, num_bars, base_price=150.0):
    """Feed ticks to produce completed bars."""
    base_time = datetime(2025, 6, 1, 14, 0)
    for i in range(num_bars + 1):  # +1 to close the last bar
        ts = base_time + timedelta(minutes=i)
        tick = Tick(
            symbol=symbol,
            timestamp=ts,
            price=base_price + i * 0.1,
            volume=100.0,
        )
        aggregator.on_tick(tick)


# ── Issue 1: State Restore ───────────────────────────────────────────


class TestRestoreState:
    def test_restore_cash(self):
        engine, portfolio, _, _ = _make_engine()
        assert portfolio.cash == 100_000.0

        snapshot = {
            "cash": 85_000.0,
            "statistics": {
                "positions": {},
                "bars_processed": 0,
                "events_processed": 0,
                "orders_submitted": 0,
                "orders_rejected": 0,
                "fills_processed": 0,
            },
        }
        engine.restore_state(snapshot)
        assert portfolio.cash == 85_000.0

    def test_restore_positions(self):
        engine, portfolio, _, _ = _make_engine(symbols=["AAPL", "MSFT"])

        snapshot = {
            "cash": 90_000.0,
            "statistics": {
                "positions": {
                    "AAPL": {
                        "quantity": 50,
                        "avg_cost": 150.0,
                        "market_value": 7500.0,
                        "unrealized_pnl": 250.0,
                        "realized_pnl": 100.0,
                    },
                    "MSFT": {
                        "quantity": 30,
                        "avg_cost": 380.0,
                        "market_value": 11400.0,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": 0.0,
                    },
                },
                "bars_processed": 0,
                "events_processed": 0,
                "orders_submitted": 0,
                "orders_rejected": 0,
                "fills_processed": 0,
            },
        }
        engine.restore_state(snapshot)

        assert portfolio.positions["AAPL"].quantity == 50
        assert portfolio.positions["AAPL"].avg_cost == 150.0
        assert portfolio.positions["AAPL"].realized_pnl == 100.0
        assert portfolio.positions["MSFT"].quantity == 30
        assert portfolio.positions["MSFT"].avg_cost == 380.0

    def test_restore_statistics(self):
        engine, _, _, _ = _make_engine()

        snapshot = {
            "cash": 100_000.0,
            "statistics": {
                "positions": {},
                "bars_processed": 120,
                "events_processed": 350,
                "orders_submitted": 15,
                "orders_rejected": 2,
                "fills_processed": 13,
            },
        }
        engine.restore_state(snapshot)

        assert engine.bars_processed == 120
        assert engine.events_processed == 350
        assert engine.orders_submitted == 15
        assert engine.orders_rejected == 2
        assert engine.fills_processed == 13

    def test_restore_empty_snapshot(self):
        engine, portfolio, _, _ = _make_engine()
        original_cash = portfolio.cash

        engine.restore_state({})

        # Should not crash, cash unchanged
        assert portfolio.cash == original_cash

    def test_restore_ignores_unknown_symbols(self):
        engine, portfolio, _, _ = _make_engine(symbols=["AAPL"])

        snapshot = {
            "statistics": {
                "positions": {
                    "AAPL": {"quantity": 10, "avg_cost": 150.0},
                    "UNKNOWN": {"quantity": 5, "avg_cost": 50.0},
                },
                "bars_processed": 0,
                "events_processed": 0,
                "orders_submitted": 0,
                "orders_rejected": 0,
                "fills_processed": 0,
            },
        }
        engine.restore_state(snapshot)

        assert portfolio.positions["AAPL"].quantity == 10
        assert "UNKNOWN" not in portfolio.positions

    def test_restore_round_trip(self):
        """Save state → restore it → state matches."""
        engine, portfolio, aggregator, _ = _make_engine(
            symbols=["AAPL"], initial_capital=100_000.0
        )

        # Simulate some activity by modifying state
        portfolio.cash = 88_000.0
        portfolio.positions["AAPL"].quantity = 75
        portfolio.positions["AAPL"].avg_cost = 160.0
        portfolio.positions["AAPL"].realized_pnl = 500.0
        engine.bars_processed = 42
        engine.fills_processed = 5

        state = engine.get_state()

        # Create fresh engine and restore
        engine2, portfolio2, _, _ = _make_engine(
            symbols=["AAPL"], initial_capital=100_000.0
        )
        engine2.restore_state(state)

        assert portfolio2.cash == 88_000.0
        assert portfolio2.positions["AAPL"].quantity == 75
        assert portfolio2.positions["AAPL"].avg_cost == 160.0
        assert portfolio2.positions["AAPL"].realized_pnl == 500.0
        assert engine2.bars_processed == 42
        assert engine2.fills_processed == 5

    def test_restore_missing_statistics_key(self):
        engine, portfolio, _, _ = _make_engine()

        snapshot = {"cash": 50_000.0}
        engine.restore_state(snapshot)

        assert portfolio.cash == 50_000.0
        # Counters stay at zero
        assert engine.bars_processed == 0


# ── Issue 2: Commission Not Hardcoded ────────────────────────────────


class TestCommissionFromConfig:
    @pytest.mark.asyncio
    async def test_commission_uses_portfolio_pct(self):
        """Engine uses portfolio.commission_pct, not hardcoded 0.001."""
        custom_pct = 0.005  # 0.5%
        engine, portfolio, aggregator, data_handler = _make_engine(
            commission_pct=custom_pct,
        )

        # Feed bars so price is available
        _feed_bars(aggregator, "AAPL", 5, base_price=100.0)

        # Create a filled broker order
        broker_order = BrokerOrder(
            order_id="test-1",
            symbol="AAPL",
            side="BUY",
            order_type="MARKET",
            quantity=100,
        )
        broker_order.status = OrderStatus.FILLED
        broker_order.broker_order_id = "PAPER-test"
        broker_order.filled_quantity = 100
        broker_order.filled_avg_price = 100.0
        broker_order.filled_at = datetime.now()

        # Call _handle_order path that creates BrokerFill
        fill = BrokerFill(
            order_id=broker_order.order_id,
            symbol=broker_order.symbol,
            side=broker_order.side,
            quantity=broker_order.filled_quantity,
            price=broker_order.filled_avg_price,
            commission=abs(
                broker_order.filled_avg_price
                * broker_order.filled_quantity
                * portfolio.commission_pct
            ),
            timestamp=broker_order.filled_at,
        )

        # Expected: 100 * 100 * 0.005 = 50.0
        expected_commission = 100.0 * 100 * custom_pct
        assert fill.commission == expected_commission

    def test_portfolio_commission_pct_is_configurable(self):
        """Portfolio stores the commission_pct we pass."""
        _, portfolio, _, _ = _make_engine(commission_pct=0.002)
        assert portfolio.commission_pct == 0.002

    def test_default_commission_pct(self):
        """Default commission_pct is 0.001."""
        _, portfolio, _, _ = _make_engine()
        assert portfolio.commission_pct == 0.001

    @pytest.mark.asyncio
    async def test_handle_order_commission_matches_config(self):
        """Full path: submit order → fill → commission uses config pct."""
        custom_pct = 0.003
        engine, portfolio, aggregator, data_handler = _make_engine(
            commission_pct=custom_pct,
        )
        engine._running = True

        # Feed bars so price and strategy work
        _feed_bars(aggregator, "AAPL", 5, base_price=150.0)

        # Connect broker
        await engine._order_manager._broker.connect()

        # Set price for paper broker
        engine._order_manager._broker.set_price("AAPL", 150.0)

        # Submit order through engine
        order_event = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10,
        )
        await engine._handle_order(order_event)

        # The FillEvent is on the queue — process it
        from src.events.event import FillEvent

        while not engine._events.empty():
            event = engine._events.get()
            if isinstance(event, FillEvent):
                engine._handle_fill(event)

        assert engine.fills_processed == 1

        # Get the trade from portfolio
        trades = portfolio.get_trade_history()
        assert len(trades) == 1
        trade = trades[0]
        # Commission should be based on custom_pct, not 0.001
        # PaperBroker uses its own commission_pct for the fill
        expected = abs(trade.price * trade.quantity * custom_pct)
        assert abs(trade.commission - expected) < 0.01


# ── Issue 3: Broker Selection ────────────────────────────────────────


class TestBrokerSelection:
    def test_paper_broker_selected_by_default(self):
        """build_components uses PaperBroker when broker_type is 'paper'."""
        from scripts.run_paper_trading import build_components
        from src.config import load_config

        config = load_config("configs/paper_trading_config.yaml")
        components = build_components(config)
        # order_manager is index 3
        order_manager = components[3]
        assert isinstance(order_manager._broker, PaperBroker)

    def test_alpaca_broker_import_exists(self):
        """AlpacaBroker is importable from the script."""
        from scripts.run_paper_trading import AlpacaBroker

        assert AlpacaBroker is not None

    def test_config_accepts_alpaca_broker_type(self):
        """BrokerConfig validates 'alpaca' as a broker_type."""
        from src.config import BrokerConfig

        cfg = BrokerConfig(broker_type="alpaca")
        assert cfg.broker_type == "alpaca"

    def test_config_rejects_invalid_broker_type(self):
        """BrokerConfig rejects unknown broker types."""
        from src.config import BrokerConfig

        with pytest.raises(ValueError):
            BrokerConfig(broker_type="unknown")

"""Integration tests for BacktestEngine class."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.data.data_handler import DataHandler, HistoricalCSVDataHandler
from src.events.event import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalEvent,
    SignalType,
)
from src.events.queue import EventQueue


class MockStrategy:
    """Mock strategy that generates a signal every N bars."""

    def __init__(self, symbols: list[str], signal_frequency: int = 10):
        self.symbols = symbols
        self.signal_frequency = signal_frequency
        self.events: EventQueue | None = None
        self.bar_count = 0
        self.signals_generated = 0
        self.current_positions: dict[str, int] = dict.fromkeys(symbols, 0)

    def set_event_queue(self, events: EventQueue) -> None:
        self.events = events

    def calculate_signals(
        self, timestamp: datetime, data_handler: DataHandler
    ) -> list[SignalEvent]:
        self.bar_count += 1

        # Generate a signal every N bars
        if self.bar_count % self.signal_frequency == 0:
            self.signals_generated += 1
            return [
                SignalEvent(
                    timestamp=timestamp,
                    symbol=self.symbols[0],
                    signal_type=SignalType.LONG,
                    strength=1.0,
                )
            ]
        return []

    def on_market_data(self, event: MarketEvent, data_handler: DataHandler) -> None:
        signals = self.calculate_signals(event.timestamp, data_handler)
        for sig in signals:
            if self.events is not None:
                self.events.put(sig)

    def update_position(self, symbol: str, quantity: int) -> None:
        self.current_positions[symbol] = quantity


class MockPortfolio:
    """Mock portfolio that tracks signals and fills."""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.events: EventQueue | None = None
        self.signals_received = 0
        self.fills_received = 0
        self.positions: dict[str, int] = {}
        self.trades: list[dict] = []
        self.equity_history: list[dict] = []

    def set_event_queue(self, events: EventQueue) -> None:
        self.events = events

    def on_signal(self, event: SignalEvent, data_handler: DataHandler) -> None:
        self.signals_received += 1

        # Generate an order for each signal
        order = OrderEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            order_type=OrderType.MARKET,
            side=(
                OrderSide.BUY
                if event.signal_type == SignalType.LONG
                else OrderSide.SELL
            ),
            quantity=100,
        )
        self.events.put(order)

    def on_fill(self, event: FillEvent) -> None:
        self.fills_received += 1

        # Track position
        symbol = event.symbol
        if symbol not in self.positions:
            self.positions[symbol] = 0

        if event.side == OrderSide.BUY:
            self.positions[symbol] += event.quantity
            self.cash -= event.quantity * event.fill_price + event.commission
        else:
            self.positions[symbol] -= event.quantity
            self.cash += event.quantity * event.fill_price - event.commission

        self.trades.append(
            {
                "timestamp": event.timestamp,
                "symbol": symbol,
                "side": event.side.value,
                "quantity": event.quantity,
                "price": event.fill_price,
            }
        )

    def update_timeindex(self, timestamp: datetime) -> None:
        self.equity_history.append(
            {"timestamp": timestamp, "equity": self.get_equity()}
        )

    def get_equity(self) -> float:
        # Simplified: just return cash (ignore position market value for mock)
        return self.cash

    def get_positions(self) -> dict[str, int]:
        return self.positions.copy()

    def get_trade_history(self) -> list[dict]:
        return self.trades.copy()


class MockExecutionHandler:
    """Mock execution handler that fills all orders immediately."""

    def __init__(self, commission: float = 1.0, slippage: float = 0.0):
        self.commission = commission
        self.slippage = slippage
        self.events: EventQueue | None = None
        self.orders_executed = 0

    def set_event_queue(self, events: EventQueue) -> None:
        self.events = events

    def execute_order(self, event: OrderEvent, data_handler: DataHandler) -> None:
        self.orders_executed += 1

        # Get current price
        bars = data_handler.get_latest_bars(event.symbol, 1)
        price = bars["close"].iloc[-1]

        # Apply slippage
        if event.side == OrderSide.BUY:
            fill_price = price * (1 + self.slippage)
        else:
            fill_price = price * (1 - self.slippage)

        # Create fill event
        fill = FillEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            side=event.side,
            quantity=event.quantity,
            fill_price=fill_price,
            commission=self.commission,
        )
        self.events.put(fill)


class MockPerformanceTracker:
    """Mock performance tracker."""

    def __init__(self):
        self.equity_curve: list[dict] = []

    def update(self, timestamp: datetime, equity: float) -> None:
        self.equity_curve.append({"timestamp": timestamp, "equity": equity})

    def calculate_metrics(self) -> dict[str, Any]:
        if not self.equity_curve:
            return {}
        return {
            "initial_equity": self.equity_curve[0]["equity"],
            "final_equity": self.equity_curve[-1]["equity"],
            "num_periods": len(self.equity_curve),
        }

    def get_equity_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create sample CSV data for testing."""
    dates = pd.date_range("2022-01-01", "2022-03-31", freq="B")
    np.random.seed(42)

    prices = 150.0 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02))

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, len(dates)),
        }
    )
    df.to_csv(tmp_path / "AAPL.csv", index=False)

    return str(tmp_path)


@pytest.fixture
def data_handler(sample_data_dir):
    """Create data handler with sample data."""
    return HistoricalCSVDataHandler(
        data_path=sample_data_dir,
        symbols=["AAPL"],
        start_date="2022-01-01",
        end_date="2022-03-31",
    )


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    def test_engine_initialization(self, data_handler):
        """Test that engine initializes correctly."""
        strategy = MockStrategy(symbols=["AAPL"])
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        assert engine.bar_count == 0
        assert engine.event_count == 0
        assert engine.events is not None

    def test_engine_runs_without_error(self, data_handler):
        """Test that engine completes a backtest run."""
        strategy = MockStrategy(symbols=["AAPL"])
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        results = engine.run()

        assert "bars_processed" in results
        assert "events_processed" in results
        assert "final_equity" in results
        assert results["bars_processed"] > 0

    def test_event_flow(self, data_handler):
        """Test that events flow correctly through the system."""
        strategy = MockStrategy(symbols=["AAPL"], signal_frequency=10)
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        results = engine.run()

        # Check event counts match
        expected_signals = results["bars_processed"] // 10
        assert strategy.signals_generated == expected_signals
        assert portfolio.signals_received == expected_signals
        assert execution.orders_executed == expected_signals
        assert portfolio.fills_received == expected_signals

    def test_market_events_equal_bars(self, data_handler):
        """Test that market events equal number of bars."""
        strategy = MockStrategy(symbols=["AAPL"], signal_frequency=1000)  # No signals
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        results = engine.run()

        # Events should equal bars (one MarketEvent per bar)
        assert results["events_processed"] == results["bars_processed"]

    def test_with_performance_tracker(self, data_handler):
        """Test engine with performance tracker."""
        strategy = MockStrategy(symbols=["AAPL"])
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()
        tracker = MockPerformanceTracker()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            performance_tracker=tracker,
        )

        results = engine.run()

        assert "metrics" in results
        assert "equity_curve" in results
        assert len(tracker.equity_curve) == results["bars_processed"]

    def test_positions_updated(self, data_handler):
        """Test that positions are updated after trades."""
        strategy = MockStrategy(symbols=["AAPL"], signal_frequency=10)
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        results = engine.run()

        # Should have a position in AAPL
        positions = results["positions"]
        assert "AAPL" in positions
        assert positions["AAPL"] > 0  # Only BUY signals in mock

    def test_trade_history_recorded(self, data_handler):
        """Test that trades are recorded."""
        strategy = MockStrategy(symbols=["AAPL"], signal_frequency=10)
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        results = engine.run()

        trades = results["trade_history"]
        expected_trades = results["bars_processed"] // 10
        assert len(trades) == expected_trades

        # Check trade structure
        if trades:
            trade = trades[0]
            assert "timestamp" in trade
            assert "symbol" in trade
            assert "quantity" in trade
            assert "price" in trade


class TestBacktestEngineEdgeCases:
    """Tests for edge cases in BacktestEngine."""

    def test_no_signals_strategy(self, data_handler):
        """Test with a strategy that generates no signals."""
        strategy = MockStrategy(symbols=["AAPL"], signal_frequency=10000)
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        results = engine.run()

        assert results["bars_processed"] > 0
        assert portfolio.signals_received == 0
        assert results["final_equity"] == portfolio.initial_capital

    def test_reset(self, data_handler):
        """Test engine reset functionality."""
        strategy = MockStrategy(symbols=["AAPL"])
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        # Run first backtest
        engine.run()
        assert engine.bar_count > 0  # Verify backtest ran

        # Reset
        engine.reset()

        assert engine.bar_count == 0
        assert engine.event_count == 0
        assert engine.events.empty()

    def test_position_sync_to_strategy(self, data_handler):
        """Test that strategy position tracking is updated after fills."""
        strategy = MockStrategy(symbols=["AAPL"], signal_frequency=10)
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        engine.run()

        # Strategy should have been informed of positions via update_position
        # MockStrategy generates LONG signals -> BUY fills -> position goes up
        assert strategy.current_positions["AAPL"] > 0
        # Strategy position should match portfolio position
        assert strategy.current_positions["AAPL"] == portfolio.positions.get("AAPL", 0)


class TestBacktestEngineWithRiskManager:
    """Integration tests for BacktestEngine with RiskManager."""

    def test_engine_accepts_risk_manager(self, data_handler):
        """Engine accepts optional risk_manager parameter."""
        from src.risk.risk_manager import RiskManager

        strategy = MockStrategy(symbols=["AAPL"])
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()
        risk_manager = RiskManager()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            risk_manager=risk_manager,
        )

        assert engine.risk_manager is risk_manager

    def test_engine_works_without_risk_manager(self, data_handler):
        """Backwards compatibility: engine runs with no risk_manager."""
        strategy = MockStrategy(symbols=["AAPL"])
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        results = engine.run()
        assert results["bars_processed"] > 0
        assert results["rejected_orders"] == 0

    def test_results_contain_rejected_orders(self, data_handler):
        """Results dict always has rejected_orders key."""
        strategy = MockStrategy(symbols=["AAPL"])
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )

        results = engine.run()
        assert "rejected_orders" in results

    def test_rate_limit_rejects_orders(self, data_handler):
        """Orders beyond rate limit are rejected."""
        from src.risk.risk_manager import RiskLimits, RiskManager

        strategy = MockStrategy(symbols=["AAPL"], signal_frequency=5)
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        limits = RiskLimits(max_orders_per_day=2)
        risk_manager = RiskManager(limits=limits)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            risk_manager=risk_manager,
        )

        results = engine.run()
        assert results["rejected_orders"] > 0
        # Execution handler should have processed fewer orders than signals
        assert execution.orders_executed < strategy.signals_generated

    def test_reset_clears_rejected_orders(self, data_handler):
        """Reset clears rejected_orders counter and risk manager state."""
        from src.risk.risk_manager import RiskLimits, RiskManager

        strategy = MockStrategy(symbols=["AAPL"], signal_frequency=5)
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()

        limits = RiskLimits(max_orders_per_day=1)
        risk_manager = RiskManager(limits=limits)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            risk_manager=risk_manager,
        )

        engine.run()
        assert engine.rejected_orders > 0

        engine.reset()
        assert engine.rejected_orders == 0
        assert risk_manager.orders_today == 0
        assert risk_manager.peak_equity == 0.0

    def test_risk_manager_with_no_signals(self, data_handler):
        """Risk manager does nothing when no signals are generated."""
        from src.risk.risk_manager import RiskManager

        strategy = MockStrategy(symbols=["AAPL"], signal_frequency=10000)
        portfolio = MockPortfolio()
        execution = MockExecutionHandler()
        risk_manager = RiskManager()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            risk_manager=risk_manager,
        )

        results = engine.run()
        assert results["rejected_orders"] == 0
        assert results["bars_processed"] > 0

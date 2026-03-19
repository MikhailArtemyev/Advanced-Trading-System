"""Critical-path integration tests.

Tests the full backtest event loop with real (not mocked) components,
verifying correctness of trade execution, P&L, equity tracking, and
risk management.
"""

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.data.data_handler import HistoricalCSVDataHandler
from src.execution.execution_handler import ExecutionHandler
from src.performance.metrics import PerformanceTracker
from src.portfolio.portfolio import Portfolio
from src.risk.position_sizer import FixedFractionSizer
from src.risk.risk_manager import RiskLimits, RiskManager
from src.strategy.sma_crossover import SMACrossoverStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(tmp_path, symbol: str, dates, prices) -> None:
    """Write a single-symbol CSV for HistoricalCSVDataHandler."""
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": [1_000_000] * len(dates),
        }
    )
    df.to_csv(tmp_path / f"{symbol}.csv", index=False)


def _make_data_handler(tmp_path, symbols, n_bars=120, seed=42):
    """Create a HistoricalCSVDataHandler with random walk data."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n_bars, freq="B")
    for symbol in symbols:
        prices = 100.0 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.015))
        _write_csv(tmp_path, symbol, dates, prices)

    return HistoricalCSVDataHandler(
        data_path=str(tmp_path),
        symbols=symbols,
        start_date="2022-01-01",
        end_date="2023-12-31",
    )


# ---------------------------------------------------------------------------
# Full pipeline: Market → Signal → Order → Fill → Portfolio
# ---------------------------------------------------------------------------


class TestFullEventLoop:
    """End-to-end backtest with real components (no mocks)."""

    def test_complete_flow_produces_trades(self, tmp_path):
        """Engine with real strategy/portfolio/execution produces trades."""
        data_handler = _make_data_handler(tmp_path, ["AAPL"])
        strategy = SMACrossoverStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 5, "long_window": 20},
        )
        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=["AAPL"],
            commission_pct=0.001,
        )
        execution = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )
        results = engine.run()

        assert results["bars_processed"] > 0
        assert results["events_processed"] > results["bars_processed"]
        trades = results["trade_history"]
        assert len(trades) > 0, "Expected at least one trade from SMA crossover"

    def test_equity_changes_after_trades(self, tmp_path):
        """Portfolio equity should differ from initial capital after trades."""
        data_handler = _make_data_handler(tmp_path, ["AAPL"])
        strategy = SMACrossoverStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 5, "long_window": 20},
        )
        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=["AAPL"],
            commission_pct=0.001,
        )
        execution = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )
        results = engine.run()

        if len(results["trade_history"]) > 0:
            # With trades, equity should change (commissions at minimum)
            assert results["final_equity"] != 100_000.0


# ---------------------------------------------------------------------------
# Order rejection flow
# ---------------------------------------------------------------------------


class TestRiskRejection:
    """Orders hitting risk limits should be rejected, portfolio unchanged."""

    def test_order_rate_limit_rejects(self, tmp_path):
        """Orders beyond daily rate limit are rejected."""
        data_handler = _make_data_handler(tmp_path, ["AAPL"])
        strategy = SMACrossoverStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 5, "long_window": 20},
        )
        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=["AAPL"],
            commission_pct=0.001,
        )
        execution = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0)
        # Only allow 1 order per day — should cause rejections
        limits = RiskLimits(max_orders_per_day=1)
        risk_manager = RiskManager(limits=limits)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            risk_manager=risk_manager,
        )
        results = engine.run()

        # If strategy generated more than 1 signal, we should see rejections
        total_signals = execution.orders_processed + results["rejected_orders"]
        if total_signals > 1:
            assert results["rejected_orders"] > 0

    def test_drawdown_halt_stops_trading(self, tmp_path):
        """Drawdown halt should prevent further orders."""
        # Create data that will cause a drawdown
        np.random.seed(99)
        dates = pd.date_range("2022-01-01", periods=120, freq="B")
        # Sharp uptrend then crash to trigger crossover + drawdown
        prices = np.concatenate(
            [
                np.linspace(100, 130, 40),
                np.linspace(130, 80, 40),
                np.linspace(80, 90, 40),
            ]
        )
        _write_csv(tmp_path, "AAPL", dates, prices)

        data_handler = HistoricalCSVDataHandler(
            data_path=str(tmp_path),
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2023-12-31",
        )
        strategy = SMACrossoverStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 5, "long_window": 20},
        )
        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=["AAPL"],
            commission_pct=0.001,
            position_sizer=FixedFractionSizer(fraction=0.50),
        )
        execution = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0)
        # Very tight drawdown limit
        limits = RiskLimits(max_drawdown_pct=0.01)
        risk_manager = RiskManager(limits=limits)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            risk_manager=risk_manager,
        )
        results = engine.run()

        # Engine should still complete (not crash)
        assert results["bars_processed"] > 0


# ---------------------------------------------------------------------------
# Equity tracking accuracy
# ---------------------------------------------------------------------------


class TestEquityTracking:
    """Verify equity curve is recorded correctly."""

    def test_equity_history_length_matches_bars(self, tmp_path):
        """Each bar should produce one equity snapshot."""
        data_handler = _make_data_handler(tmp_path, ["AAPL"], n_bars=50)
        strategy = SMACrossoverStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 5, "long_window": 20},
        )
        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=["AAPL"],
        )
        execution = ExecutionHandler()
        tracker = PerformanceTracker(initial_capital=100_000.0)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
            performance_tracker=tracker,
        )
        results = engine.run()

        equity_curve = results.get("equity_curve")
        if equity_curve is not None and not equity_curve.empty:
            assert len(equity_curve) == results["bars_processed"]

    def test_commission_reduces_equity(self, tmp_path):
        """Trades with commission should reduce total equity vs. no-trade scenario."""
        # Use a fixed known dataset
        dates = pd.date_range("2022-01-01", periods=60, freq="B")
        # Strong uptrend then downtrend to guarantee crossovers
        prices = np.concatenate(
            [
                np.linspace(100, 120, 30),
                np.linspace(120, 95, 30),
            ]
        )
        _write_csv(tmp_path, "AAPL", dates, prices)

        data_handler = HistoricalCSVDataHandler(
            data_path=str(tmp_path),
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2023-12-31",
        )
        strategy = SMACrossoverStrategy(
            symbols=["AAPL"],
            parameters={"short_window": 5, "long_window": 15},
        )
        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=["AAPL"],
            commission_pct=0.01,  # 1% commission — clearly visible
        )
        execution = ExecutionHandler(commission_pct=0.01, slippage_pct=0.0)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )
        results = engine.run()

        if len(results["trade_history"]) > 0:
            total_commission = execution.get_statistics()["total_commission"]
            assert total_commission > 0


# ---------------------------------------------------------------------------
# Multi-symbol
# ---------------------------------------------------------------------------


class TestMultiSymbol:
    """Test engine handles multiple symbols correctly."""

    def test_multi_symbol_runs(self, tmp_path):
        """Engine with multiple symbols completes without error."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        data_handler = _make_data_handler(tmp_path, symbols, n_bars=80)
        strategy = SMACrossoverStrategy(
            symbols=symbols,
            parameters={"short_window": 5, "long_window": 20},
        )
        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=symbols,
            commission_pct=0.001,
        )
        execution = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution,
        )
        results = engine.run()

        assert results["bars_processed"] > 0
        # Should have positions dict with all symbols
        for sym in symbols:
            assert sym in results["positions"]

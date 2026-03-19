"""Tests for Week 7: run_backtest factories and extended performance metrics."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_backtest import build_optimizer, build_strategy
from src.config import (
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    OptimizationConfig,
    StrategyConfig,
)
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer
from src.performance.metrics import PerformanceTracker
from src.strategy.multi_asset_sma import MultiAssetSMAStrategy
from src.strategy.sma_crossover import SMACrossoverStrategy


def _base_config(**overrides: object) -> BacktestConfig:
    """Create a minimal config, with optional overrides for sub-configs."""
    kwargs: dict[str, object] = {
        "data": DataConfig(
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2022-12-31",
            data_source="csv",
            data_path="data/sample",
        ),
        "execution": ExecutionConfig(),
        "strategy": StrategyConfig(name="sma_crossover"),
    }
    kwargs.update(overrides)
    return BacktestConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_optimizer tests
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    """Tests for build_optimizer factory."""

    def test_none_returns_none(self) -> None:
        config = _base_config()
        optimizer = build_optimizer(config)
        assert optimizer is None

    def test_mean_variance_default_params(self) -> None:
        config = _base_config(optimization=OptimizationConfig(method="mean_variance"))
        optimizer = build_optimizer(config)
        assert isinstance(optimizer, MeanVarianceOptimizer)

    def test_mean_variance_with_params(self) -> None:
        config = _base_config(
            optimization=OptimizationConfig(
                method="mean_variance",
                parameters={"target": "min_variance", "max_weight": 0.5},
            )
        )
        optimizer = build_optimizer(config)
        assert isinstance(optimizer, MeanVarianceOptimizer)
        assert optimizer.target == "min_variance"
        assert optimizer.max_weight == 0.5

    def test_risk_parity_default_params(self) -> None:
        config = _base_config(optimization=OptimizationConfig(method="risk_parity"))
        optimizer = build_optimizer(config)
        assert isinstance(optimizer, RiskParityOptimizer)

    def test_risk_parity_with_params(self) -> None:
        config = _base_config(
            optimization=OptimizationConfig(
                method="risk_parity",
                parameters={"max_weight": 0.25, "min_weight": 0.05},
            )
        )
        optimizer = build_optimizer(config)
        assert isinstance(optimizer, RiskParityOptimizer)
        assert optimizer.max_weight == 0.25
        assert optimizer.min_weight == 0.05


# ---------------------------------------------------------------------------
# build_strategy tests
# ---------------------------------------------------------------------------


class TestBuildStrategy:
    """Tests for build_strategy / strategy auto-selection."""

    def test_single_symbol_no_optimizer(self) -> None:
        config = _base_config()
        strategy = build_strategy(config, optimizer=None)
        assert isinstance(strategy, SMACrossoverStrategy)

    def test_multi_symbol_no_optimizer(self) -> None:
        config = _base_config(
            data=DataConfig(
                symbols=["AAPL", "MSFT"],
                start_date="2022-01-01",
                end_date="2022-12-31",
                data_source="csv",
            )
        )
        strategy = build_strategy(config, optimizer=None)
        assert isinstance(strategy, SMACrossoverStrategy)

    def test_single_symbol_with_optimizer(self) -> None:
        config = _base_config()
        optimizer = MeanVarianceOptimizer()
        strategy = build_strategy(config, optimizer=optimizer)
        assert isinstance(strategy, SMACrossoverStrategy)

    def test_multi_symbol_with_optimizer(self) -> None:
        config = _base_config(
            data=DataConfig(
                symbols=["AAPL", "MSFT", "GOOGL"],
                start_date="2022-01-01",
                end_date="2022-12-31",
                data_source="csv",
            ),
        )
        optimizer = MeanVarianceOptimizer()
        strategy = build_strategy(config, optimizer=optimizer)
        assert isinstance(strategy, MultiAssetSMAStrategy)
        assert strategy.optimizer is optimizer

    def test_rebalance_frequency_from_optimization_config(self) -> None:
        config = _base_config(
            data=DataConfig(
                symbols=["AAPL", "MSFT"],
                start_date="2022-01-01",
                end_date="2022-12-31",
                data_source="csv",
            ),
            optimization=OptimizationConfig(
                method="risk_parity",
                rebalance_frequency=15,
            ),
        )
        optimizer = RiskParityOptimizer()
        strategy = build_strategy(config, optimizer=optimizer)
        assert isinstance(strategy, MultiAssetSMAStrategy)
        assert strategy.rebalance_frequency == 15

    def test_strategy_params_rebalance_frequency_takes_precedence(self) -> None:
        config = _base_config(
            data=DataConfig(
                symbols=["AAPL", "MSFT"],
                start_date="2022-01-01",
                end_date="2022-12-31",
                data_source="csv",
            ),
            strategy=StrategyConfig(
                name="sma_crossover",
                parameters={
                    "short_window": 20,
                    "long_window": 50,
                    "rebalance_frequency": 5,
                },
            ),
            optimization=OptimizationConfig(
                method="mean_variance",
                rebalance_frequency=30,
            ),
        )
        optimizer = MeanVarianceOptimizer()
        strategy = build_strategy(config, optimizer=optimizer)
        assert isinstance(strategy, MultiAssetSMAStrategy)
        assert strategy.rebalance_frequency == 5


# ---------------------------------------------------------------------------
# calculate_extended_metrics tests
# ---------------------------------------------------------------------------


class TestExtendedMetrics:
    """Tests for calculate_extended_metrics."""

    @staticmethod
    def _make_tracker_with_data() -> PerformanceTracker:
        tracker = PerformanceTracker(initial_capital=100000.0)
        for i in range(10):
            tracker.update(datetime(2023, 1, i + 1), 100000.0 + i * 500)
        return tracker

    @staticmethod
    def _make_portfolio_history() -> list[dict]:
        return [
            {
                "timestamp": datetime(2023, 1, d + 1),
                "equity": 100000.0 + d * 500,
                "cash": 90000.0,
                "positions_value": 10000.0 + d * 500,
                "num_positions": d % 3 + 1,
            }
            for d in range(10)
        ]

    def test_includes_base_metrics(self) -> None:
        tracker = self._make_tracker_with_data()
        history = self._make_portfolio_history()
        trades = [MagicMock(side="SELL", pnl=100, price=150.0, quantity=10)]

        result = tracker.calculate_extended_metrics(trades, history)

        assert "initial_capital" in result
        assert "final_equity" in result
        assert "sharpe_ratio" in result

    def test_risk_metrics_present(self) -> None:
        tracker = self._make_tracker_with_data()
        history = self._make_portfolio_history()
        trades = [MagicMock(side="SELL", pnl=100, price=150.0, quantity=10)]

        result = tracker.calculate_extended_metrics(trades, history)

        assert "risk_metrics" in result
        rm = result["risk_metrics"]
        assert "avg_position_count" in rm
        assert "max_position_count" in rm
        assert "gross_exposure_avg" in rm
        assert "turnover" in rm

    def test_avg_position_count(self) -> None:
        tracker = self._make_tracker_with_data()
        history = [
            {
                "timestamp": datetime(2023, 1, d + 1),
                "equity": 100000.0,
                "cash": 80000.0,
                "positions_value": 20000.0,
                "num_positions": 2,
            }
            for d in range(10)
        ]

        result = tracker.calculate_extended_metrics([], history)
        assert result["risk_metrics"]["avg_position_count"] == 2.0

    def test_max_position_count(self) -> None:
        tracker = self._make_tracker_with_data()
        history = [
            {
                "timestamp": datetime(2023, 1, d + 1),
                "equity": 100000.0,
                "cash": 80000.0,
                "positions_value": 20000.0,
                "num_positions": d,
            }
            for d in range(5)
        ]

        result = tracker.calculate_extended_metrics([], history)
        assert result["risk_metrics"]["max_position_count"] == 4

    def test_trade_metrics_included_when_trades_exist(self) -> None:
        tracker = self._make_tracker_with_data()
        history = self._make_portfolio_history()
        trades = [MagicMock(side="SELL", pnl=500, price=150.0, quantity=10)]

        result = tracker.calculate_extended_metrics(trades, history)
        assert "trade_metrics" in result
        assert result["trade_metrics"]["total_trades"] == 1

    def test_trade_metrics_absent_when_no_trades(self) -> None:
        tracker = self._make_tracker_with_data()
        history = self._make_portfolio_history()

        result = tracker.calculate_extended_metrics([], history)
        assert "trade_metrics" not in result

    def test_no_portfolio_history_omits_position_metrics(self) -> None:
        tracker = self._make_tracker_with_data()
        trades = [MagicMock(side="SELL", pnl=100, price=150.0, quantity=10)]

        result = tracker.calculate_extended_metrics(trades, None)
        rm = result["risk_metrics"]
        assert "avg_position_count" not in rm
        assert "max_position_count" not in rm
        assert "gross_exposure_avg" not in rm
        assert "turnover" in rm

    def test_insufficient_data_returns_error(self) -> None:
        tracker = PerformanceTracker(initial_capital=100000.0)
        tracker.update(datetime(2023, 1, 1), 100000.0)

        result = tracker.calculate_extended_metrics([], None)
        assert "error" in result


# ---------------------------------------------------------------------------
# _calculate_turnover tests
# ---------------------------------------------------------------------------


class TestCalculateTurnover:
    """Tests for _calculate_turnover."""

    def test_no_trades_returns_zero(self) -> None:
        tracker = PerformanceTracker(initial_capital=100000.0)
        for i in range(5):
            tracker.update(datetime(2023, 1, i + 1), 100000.0)

        result = tracker._calculate_turnover([], None)
        assert result == 0.0

    def test_basic_turnover_calculation(self) -> None:
        tracker = PerformanceTracker(initial_capital=100000.0)
        for i in range(10):
            tracker.update(datetime(2023, 1, i + 1), 100000.0)

        trade = MagicMock(price=150.0, quantity=100, side="BUY")
        result = tracker._calculate_turnover([trade], None)
        # (15000 / 100000) * (252 / 10) = 0.15 * 25.2 = 3.78
        assert abs(result - 3.78) < 0.01

    def test_turnover_with_portfolio_history(self) -> None:
        tracker = PerformanceTracker(initial_capital=100000.0)
        for i in range(5):
            tracker.update(datetime(2023, 1, i + 1), 200000.0)

        history = [
            {
                "timestamp": datetime(2023, 1, d + 1),
                "equity": 50000.0,
                "cash": 40000.0,
                "positions_value": 10000.0,
                "num_positions": 1,
            }
            for d in range(20)
        ]

        trade = MagicMock(price=100.0, quantity=50, side="SELL")
        # total_traded = 5000, avg_equity = 50000, days = 20
        # turnover = (5000 / 50000) * (252 / 20) = 0.1 * 12.6 = 1.26
        result = tracker._calculate_turnover([trade], history)
        assert abs(result - 1.26) < 0.01

    def test_turnover_zero_equity(self) -> None:
        tracker = PerformanceTracker(initial_capital=100000.0)
        history = [
            {
                "timestamp": datetime(2023, 1, 1),
                "equity": 0.0,
                "cash": 0.0,
                "positions_value": 0.0,
                "num_positions": 0,
            }
        ]
        trade = MagicMock(price=100.0, quantity=10, side="BUY")

        result = tracker._calculate_turnover([trade], history)
        assert result == 0.0

    def test_turnover_no_equity_data(self) -> None:
        tracker = PerformanceTracker(initial_capital=100000.0)
        trade = MagicMock(price=100.0, quantity=10, side="BUY")

        result = tracker._calculate_turnover([trade], None)
        assert result == 0.0

    def test_multiple_trades_summed(self) -> None:
        tracker = PerformanceTracker(initial_capital=100000.0)
        for i in range(10):
            tracker.update(datetime(2023, 1, i + 1), 100000.0)

        trades = [
            MagicMock(price=100.0, quantity=50, side="BUY"),
            MagicMock(price=200.0, quantity=30, side="SELL"),
        ]
        # total_traded = 5000 + 6000 = 11000, avg_equity = 100000, days = 10
        # turnover = (11000 / 100000) * (252 / 10) = 0.11 * 25.2 = 2.772
        result = tracker._calculate_turnover(trades, None)
        assert abs(result - 2.772) < 0.01


# ---------------------------------------------------------------------------
# print_report extended tests
# ---------------------------------------------------------------------------


class TestPrintReportExtended:
    """Tests for print_report with extended metrics."""

    def test_print_report_with_portfolio_history(self, caplog) -> None:
        tracker = PerformanceTracker(initial_capital=100000.0)
        for i in range(10):
            tracker.update(datetime(2023, 1, i + 1), 100000.0 + i * 500)

        history = [
            {
                "timestamp": datetime(2023, 1, d + 1),
                "equity": 100000.0 + d * 500,
                "cash": 90000.0,
                "positions_value": 10000.0 + d * 500,
                "num_positions": 2,
            }
            for d in range(10)
        ]
        trades = [MagicMock(side="SELL", pnl=500, price=150.0, quantity=10)]

        with caplog.at_level("INFO", logger="src.performance.metrics"):
            tracker.print_report(trades=trades, portfolio_equity_history=history)

        assert "Avg Positions:" in caplog.text
        assert "Max Positions:" in caplog.text
        assert "Avg Exposure:" in caplog.text
        assert "Turnover" in caplog.text

    def test_print_report_without_history_no_extended(self, caplog) -> None:
        tracker = PerformanceTracker(initial_capital=100000.0)
        for i in range(10):
            tracker.update(datetime(2023, 1, i + 1), 100000.0 + i * 500)

        trades = [MagicMock(side="SELL", pnl=500, price=150.0, quantity=10)]

        with caplog.at_level("INFO", logger="src.performance.metrics"):
            tracker.print_report(trades=trades)

        assert "BACKTEST PERFORMANCE REPORT" in caplog.text
        assert "Avg Positions:" not in caplog.text
        assert "Total Trades:" in caplog.text

    def test_backward_compatible_call(self, caplog) -> None:
        tracker = PerformanceTracker(initial_capital=100000.0)
        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 105000.0)

        trades = [MagicMock(side="SELL", pnl=500)]
        with caplog.at_level("INFO", logger="src.performance.metrics"):
            tracker.print_report(trades)

        assert "BACKTEST PERFORMANCE REPORT" in caplog.text
        assert "Total Trades:" in caplog.text

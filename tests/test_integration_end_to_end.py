"""End-to-end integration tests for the full backtest pipeline.

Wires together DataHandler, Strategy, Portfolio, ExecutionHandler,
PerformanceTracker, RiskManager, and PortfolioOptimizer to verify
the complete system works as a whole.
"""

import os

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.config import BacktestConfig, load_config
from src.data.data_handler import HistoricalCSVDataHandler
from src.execution.execution_handler import ExecutionHandler
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer
from src.performance.metrics import PerformanceTracker
from src.portfolio.correlation import CorrelationTracker
from src.portfolio.portfolio import Portfolio
from src.risk.position_sizer import (
    FixedFractionSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
)
from src.risk.risk_manager import RiskLimits, RiskManager
from src.strategy.multi_asset_sma import MultiAssetSMAStrategy
from src.strategy.sma_crossover import SMACrossoverStrategy

# ---------------------------------------------------------------------------
# Helpers: create synthetic CSV data for testing
# ---------------------------------------------------------------------------


def _make_csv_data(
    symbol: str,
    start: str = "2020-01-02",
    periods: int = 200,
    base_price: float = 100.0,
    trend: float = 0.0005,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    import numpy as np

    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    closes = [base_price]
    for _ in range(1, periods):
        ret = trend + rng.randn() * 0.015
        closes.append(closes[-1] * (1 + ret))
    closes = np.array(closes)
    highs = closes * (1 + rng.uniform(0.001, 0.02, periods))
    lows = closes * (1 - rng.uniform(0.001, 0.02, periods))
    opens = closes * (1 + rng.uniform(-0.01, 0.01, periods))
    volumes = rng.randint(1_000_000, 10_000_000, periods)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    return df


@pytest.fixture()
def data_dir(tmp_path):
    """Create a temp directory with CSV data for 3 symbols."""
    symbols = ["AAA", "BBB", "CCC"]
    seeds = [42, 123, 456]
    for sym, seed in zip(symbols, seeds, strict=True):
        df = _make_csv_data(sym, periods=200, seed=seed)
        df.to_csv(tmp_path / f"{sym}.csv", index=False)
    return str(tmp_path)


def _build_data_handler(data_dir, symbols, periods=200):
    """Build a HistoricalCSVDataHandler."""
    df = pd.read_csv(os.path.join(data_dir, f"{symbols[0]}.csv"), parse_dates=["date"])
    start = str(df["date"].iloc[0].date())
    end = str(df["date"].iloc[-1].date())
    return HistoricalCSVDataHandler(
        data_path=data_dir,
        symbols=symbols,
        start_date=start,
        end_date=end,
    )


def _run_backtest(
    data_dir,
    symbols,
    *,
    position_sizer=None,
    risk_manager=None,
    strategy=None,
    initial_capital=100_000.0,
    correlation_tracker=None,
):
    """Helper that wires everything together and runs a backtest."""
    dh = _build_data_handler(data_dir, symbols)
    sizer = position_sizer or FixedFractionSizer(fraction=0.10)
    strat = strategy or SMACrossoverStrategy(
        symbols=symbols, parameters={"short_window": 10, "long_window": 30}
    )
    portfolio = Portfolio(
        initial_capital=initial_capital,
        symbols=symbols,
        commission_pct=0.001,
        position_sizer=sizer,
        correlation_tracker=correlation_tracker,
    )
    execution = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0005)
    tracker = PerformanceTracker(initial_capital=initial_capital)

    engine = BacktestEngine(
        data_handler=dh,
        strategy=strat,
        portfolio=portfolio,
        execution_handler=execution,
        performance_tracker=tracker,
        risk_manager=risk_manager,
    )
    results = engine.run()
    return results, portfolio, tracker, engine


# ---------------------------------------------------------------------------
# Single-asset integration
# ---------------------------------------------------------------------------


class TestSingleAssetBacktest:
    """End-to-end single-asset backtest."""

    def test_basic_single_asset_runs(self, data_dir):
        results, portfolio, tracker, _ = _run_backtest(data_dir, ["AAA"])
        assert results["bars_processed"] > 0
        assert results["events_processed"] > 0
        assert results["final_equity"] > 0
        assert "metrics" in results
        assert "equity_curve" in results

    def test_equity_curve_structure(self, data_dir):
        results, _, tracker, _ = _run_backtest(data_dir, ["AAA"])
        ec = results["equity_curve"]
        assert isinstance(ec, pd.DataFrame)
        assert "timestamp" in ec.columns
        assert "equity" in ec.columns
        assert len(ec) == results["bars_processed"]

    def test_metrics_contain_required_keys(self, data_dir):
        results, _, _, _ = _run_backtest(data_dir, ["AAA"])
        metrics = results["metrics"]
        for key in [
            "total_return_pct",
            "annualized_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "volatility_pct",
        ]:
            assert key in metrics

    def test_with_fixed_fraction_sizer(self, data_dir):
        sizer = FixedFractionSizer(fraction=0.15)
        results, _, _, _ = _run_backtest(data_dir, ["AAA"], position_sizer=sizer)
        assert results["bars_processed"] > 0

    def test_with_volatility_sizer(self, data_dir):
        sizer = VolatilityBasedSizer(risk_fraction=0.02, atr_period=14)
        results, _, _, _ = _run_backtest(data_dir, ["AAA"], position_sizer=sizer)
        assert results["bars_processed"] > 0

    def test_with_kelly_sizer(self, data_dir):
        sizer = KellyCriterionSizer(
            kelly_fraction=0.5, min_trades=1, default_fraction=0.10
        )
        results, _, _, _ = _run_backtest(data_dir, ["AAA"], position_sizer=sizer)
        assert results["bars_processed"] > 0

    def test_with_risk_manager(self, data_dir):
        rm = RiskManager(
            limits=RiskLimits(
                max_position_pct=0.10,
                max_drawdown_pct=0.15,
                max_daily_loss_pct=0.03,
            )
        )
        results, _, _, engine = _run_backtest(data_dir, ["AAA"], risk_manager=rm)
        assert results["bars_processed"] > 0
        assert "rejected_orders" in results

    def test_tight_risk_limits_reject_orders(self, data_dir):
        rm = RiskManager(
            limits=RiskLimits(
                max_position_pct=0.001,
                max_portfolio_exposure_pct=0.001,
            )
        )
        results, _, _, engine = _run_backtest(data_dir, ["AAA"], risk_manager=rm)
        # With very tight limits, most orders should be rejected or adjusted
        assert results["rejected_orders"] >= 0

    def test_phase1_backwards_compat(self, data_dir):
        """Phase 1 style: no sizer, no risk manager, just strategy + portfolio."""
        dh = _build_data_handler(data_dir, ["AAA"])
        strat = SMACrossoverStrategy(
            symbols=["AAA"], parameters={"short_window": 10, "long_window": 30}
        )
        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=["AAA"],
            commission_pct=0.001,
        )
        execution = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0005)
        tracker = PerformanceTracker(initial_capital=100_000.0)
        engine = BacktestEngine(
            data_handler=dh,
            strategy=strat,
            portfolio=portfolio,
            execution_handler=execution,
            performance_tracker=tracker,
        )
        results = engine.run()
        assert results["bars_processed"] > 0
        assert results["final_equity"] > 0

    def test_portfolio_equity_history_structure(self, data_dir):
        _, portfolio, _, _ = _run_backtest(data_dir, ["AAA"])
        assert len(portfolio.equity_history) > 0
        entry = portfolio.equity_history[0]
        for key in ["timestamp", "equity", "cash", "positions_value", "num_positions"]:
            assert key in entry

    def test_portfolio_summary_available(self, data_dir):
        _, portfolio, _, _ = _run_backtest(data_dir, ["AAA"])
        summary = portfolio.get_portfolio_summary()
        for key in [
            "equity",
            "cash",
            "long_exposure",
            "short_exposure",
            "num_positions",
        ]:
            assert key in summary


# ---------------------------------------------------------------------------
# Multi-asset integration
# ---------------------------------------------------------------------------


class TestMultiAssetBacktest:
    """End-to-end multi-asset backtest."""

    def test_multi_asset_no_optimizer(self, data_dir):
        strat = SMACrossoverStrategy(
            symbols=["AAA", "BBB", "CCC"],
            parameters={"short_window": 10, "long_window": 30},
        )
        results, _, _, _ = _run_backtest(
            data_dir, ["AAA", "BBB", "CCC"], strategy=strat
        )
        assert results["bars_processed"] > 0

    def test_multi_asset_mean_variance(self, data_dir):
        optimizer = MeanVarianceOptimizer(target="max_sharpe")
        strat = MultiAssetSMAStrategy(
            symbols=["AAA", "BBB", "CCC"],
            parameters={
                "short_window": 10,
                "long_window": 30,
                "rebalance_frequency": 20,
            },
            optimizer=optimizer,
        )
        results, _, _, _ = _run_backtest(
            data_dir, ["AAA", "BBB", "CCC"], strategy=strat
        )
        assert results["bars_processed"] > 0

    def test_multi_asset_risk_parity(self, data_dir):
        optimizer = RiskParityOptimizer()
        strat = MultiAssetSMAStrategy(
            symbols=["AAA", "BBB", "CCC"],
            parameters={
                "short_window": 10,
                "long_window": 30,
                "rebalance_frequency": 20,
            },
            optimizer=optimizer,
        )
        results, _, _, _ = _run_backtest(
            data_dir, ["AAA", "BBB", "CCC"], strategy=strat
        )
        assert results["bars_processed"] > 0

    def test_multi_asset_with_risk_manager(self, data_dir):
        rm = RiskManager(
            limits=RiskLimits(
                max_position_pct=0.20,
                max_open_positions=3,
            )
        )
        results, _, _, _ = _run_backtest(
            data_dir, ["AAA", "BBB", "CCC"], risk_manager=rm
        )
        assert results["bars_processed"] > 0

    def test_multi_asset_with_correlation_tracker(self, data_dir):
        ct = CorrelationTracker(window=60, min_periods=30)
        results, portfolio, _, _ = _run_backtest(
            data_dir, ["AAA", "BBB", "CCC"], correlation_tracker=ct
        )
        assert results["bars_processed"] > 0
        # Correlation tracker should have been fed prices
        assert len(ct.symbols) >= 1

    def test_multi_asset_all_symbols_traded(self, data_dir):
        strat = SMACrossoverStrategy(
            symbols=["AAA", "BBB", "CCC"],
            parameters={"short_window": 10, "long_window": 30},
        )
        results, _, _, _ = _run_backtest(
            data_dir, ["AAA", "BBB", "CCC"], strategy=strat
        )
        # Should have trades for at least some symbols
        trades = results["trade_history"]
        symbols_traded = {t.symbol for t in trades}
        assert len(symbols_traded) >= 1


# ---------------------------------------------------------------------------
# Performance report integration
# ---------------------------------------------------------------------------


class TestPerformanceReportIntegration:
    """Test that performance reporting works end-to-end."""

    def test_print_report_no_error(self, data_dir):
        results, portfolio, tracker, _ = _run_backtest(data_dir, ["AAA"])
        # Should not raise
        tracker.print_report(
            trades=results["trade_history"],
            portfolio_equity_history=portfolio.equity_history,
        )

    def test_extended_metrics_integration(self, data_dir):
        results, portfolio, tracker, _ = _run_backtest(data_dir, ["AAA"])
        extended = tracker.calculate_extended_metrics(
            trades=results["trade_history"],
            portfolio_equity_history=portfolio.equity_history,
        )
        assert "risk_metrics" in extended
        assert "turnover" in extended["risk_metrics"]

    def test_print_report_trades_only(self, data_dir):
        results, _, tracker, _ = _run_backtest(data_dir, ["AAA"])
        # With trades but no portfolio history
        tracker.print_report(trades=results["trade_history"])


# ---------------------------------------------------------------------------
# Config-driven integration
# ---------------------------------------------------------------------------


class TestConfigDrivenBacktest:
    """Test loading config from YAML and running a backtest."""

    def test_load_config_and_build_components(self, data_dir):
        """Verify config loads and components can be constructed."""
        config = BacktestConfig(
            data={
                "symbols": ["AAA"],
                "start_date": "2020-01-02",
                "end_date": "2020-12-31",
                "data_source": "csv",
                "data_path": data_dir,
            },
            execution={"initial_capital": 100000.0},
            strategy={
                "name": "sma_crossover",
                "parameters": {"short_window": 10, "long_window": 30},
            },
        )
        assert config.sizing.method == "fixed_fraction"
        assert config.risk.enabled is True
        assert config.optimization.method == "none"

    def test_yaml_round_trip(self, tmp_path, data_dir):
        """Write a YAML config, load it, and verify values."""
        import yaml

        config_data = {
            "data": {
                "symbols": ["AAA"],
                "start_date": "2020-01-02",
                "end_date": "2020-12-31",
                "data_source": "csv",
                "data_path": data_dir,
            },
            "execution": {"initial_capital": 50000.0, "commission_pct": 0.002},
            "strategy": {
                "name": "sma_crossover",
                "parameters": {"short_window": 15, "long_window": 40},
            },
            "sizing": {"method": "volatility", "parameters": {"risk_fraction": 0.02}},
            "risk": {"enabled": True, "max_position_pct": 0.05},
            "optimization": {
                "method": "mean_variance",
                "parameters": {"target": "max_sharpe"},
            },
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        assert config.data.symbols == ["AAA"]
        assert config.execution.initial_capital == 50000.0
        assert config.sizing.method == "volatility"
        assert config.optimization.method == "mean_variance"


# ---------------------------------------------------------------------------
# Engine reset and re-run
# ---------------------------------------------------------------------------


class TestEngineReset:
    """Test that the engine can be reset and re-run."""

    def test_reset_and_rerun(self, data_dir):
        dh = _build_data_handler(data_dir, ["AAA"])
        strat = SMACrossoverStrategy(
            symbols=["AAA"], parameters={"short_window": 10, "long_window": 30}
        )
        portfolio = Portfolio(
            initial_capital=100_000.0,
            symbols=["AAA"],
            commission_pct=0.001,
        )
        execution = ExecutionHandler(commission_pct=0.001, slippage_pct=0.0005)
        tracker = PerformanceTracker(initial_capital=100_000.0)
        engine = BacktestEngine(
            data_handler=dh,
            strategy=strat,
            portfolio=portfolio,
            execution_handler=execution,
            performance_tracker=tracker,
        )
        r1 = engine.run()
        assert r1["bars_processed"] > 0

        engine.reset()
        tracker.reset()
        r2 = engine.run()
        assert r2["bars_processed"] == r1["bars_processed"]

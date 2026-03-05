"""Configuration model validation tests.

Tests Pydantic config models, field validators, defaults, and
backwards compatibility with minimal Phase 1 configs.
"""

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    OptimizationConfig,
    RiskConfig,
    SizingConfig,
    load_config,
)

# ---------------------------------------------------------------------------
# Individual config model tests
# ---------------------------------------------------------------------------


class TestDataConfig:
    def test_required_fields(self):
        dc = DataConfig(
            symbols=["AAPL"], start_date="2020-01-01", end_date="2020-12-31"
        )
        assert dc.symbols == ["AAPL"]
        assert dc.data_source == "csv"

    def test_custom_data_source(self):
        dc = DataConfig(
            symbols=["X"],
            start_date="2020-01-01",
            end_date="2020-12-31",
            data_source="yfinance",
            data_path="/tmp/cache",
        )
        assert dc.data_source == "yfinance"
        assert dc.data_path == "/tmp/cache"


class TestExecutionConfig:
    def test_defaults(self):
        ec = ExecutionConfig()
        assert ec.initial_capital == 100000.0
        assert ec.commission_pct == 0.001
        assert ec.slippage_pct == 0.0005

    def test_custom_values(self):
        ec = ExecutionConfig(initial_capital=50000.0, commission_pct=0.002)
        assert ec.initial_capital == 50000.0

    def test_invalid_capital(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(initial_capital=-1000.0)

    def test_invalid_commission(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(commission_pct=-0.01)


class TestSizingConfig:
    def test_defaults(self):
        sc = SizingConfig()
        assert sc.method == "fixed_fraction"
        assert sc.parameters == {}

    def test_valid_methods(self):
        for method in ["fixed_fraction", "volatility", "kelly"]:
            sc = SizingConfig(method=method)
            assert sc.method == method

    def test_invalid_method(self):
        with pytest.raises(ValidationError, match="sizing method"):
            SizingConfig(method="invalid_method")

    def test_with_parameters(self):
        sc = SizingConfig(method="kelly", parameters={"kelly_fraction": 0.5})
        assert sc.parameters["kelly_fraction"] == 0.5


class TestRiskConfig:
    def test_defaults(self):
        rc = RiskConfig()
        assert rc.enabled is True
        assert rc.max_position_pct == 0.10
        assert rc.max_drawdown_pct == 0.15
        assert rc.max_open_positions == 20

    def test_disabled(self):
        rc = RiskConfig(enabled=False)
        assert rc.enabled is False

    def test_invalid_pct_range(self):
        with pytest.raises(ValidationError):
            RiskConfig(max_position_pct=1.5)

    def test_invalid_max_positions(self):
        with pytest.raises(ValidationError):
            RiskConfig(max_open_positions=0)


class TestOptimizationConfig:
    def test_defaults(self):
        oc = OptimizationConfig()
        assert oc.method == "none"
        assert oc.rebalance_frequency == 20

    def test_valid_methods(self):
        for method in ["none", "mean_variance", "risk_parity"]:
            oc = OptimizationConfig(method=method)
            assert oc.method == method

    def test_invalid_method(self):
        with pytest.raises(ValidationError, match="optimization method"):
            OptimizationConfig(method="genetic_algorithm")

    def test_with_parameters(self):
        oc = OptimizationConfig(
            method="mean_variance",
            parameters={"target": "max_sharpe", "max_weight": 0.4},
        )
        assert oc.parameters["target"] == "max_sharpe"


# ---------------------------------------------------------------------------
# BacktestConfig tests
# ---------------------------------------------------------------------------

_MINIMAL_DATA = {
    "data": {
        "symbols": ["AAPL"],
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
    },
    "execution": {},
    "strategy": {"name": "sma_crossover"},
}


class TestBacktestConfig:
    def test_defaults_from_minimal(self):
        config = BacktestConfig(**_MINIMAL_DATA)
        assert config.sizing.method == "fixed_fraction"
        assert config.risk.enabled is True
        assert config.optimization.method == "none"

    def test_phase1_backwards_compat(self):
        """Phase 1 config: only data, execution, strategy."""
        config = BacktestConfig(
            data={
                "symbols": ["AAPL"],
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
            },
            execution={"initial_capital": 100000.0},
            strategy={
                "name": "sma_crossover",
                "parameters": {"short_window": 20, "long_window": 50},
            },
        )
        # Sizing, risk, optimization should have defaults
        assert config.sizing.method == "fixed_fraction"
        assert config.risk.max_drawdown_pct == 0.15
        assert config.optimization.method == "none"

    def test_full_config(self):
        config = BacktestConfig(
            data={
                "symbols": ["AAPL", "MSFT"],
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
            },
            execution={"initial_capital": 50000.0},
            strategy={"name": "sma_crossover"},
            sizing={"method": "kelly", "parameters": {"kelly_fraction": 0.25}},
            risk={"enabled": True, "max_position_pct": 0.05},
            optimization={"method": "risk_parity", "rebalance_frequency": 30},
        )
        assert config.sizing.method == "kelly"
        assert config.optimization.rebalance_frequency == 30


# ---------------------------------------------------------------------------
# YAML round-trip tests
# ---------------------------------------------------------------------------


class TestYAMLRoundTrip:
    def test_write_and_load(self, tmp_path):
        config_data = {
            "data": {
                "symbols": ["X", "Y"],
                "start_date": "2021-01-01",
                "end_date": "2021-12-31",
                "data_source": "csv",
            },
            "execution": {"initial_capital": 75000.0, "commission_pct": 0.002},
            "strategy": {
                "name": "sma_crossover",
                "parameters": {"short_window": 10, "long_window": 50},
            },
            "sizing": {"method": "volatility", "parameters": {"risk_fraction": 0.03}},
            "risk": {"enabled": True, "max_drawdown_pct": 0.10},
            "optimization": {"method": "mean_variance", "rebalance_frequency": 25},
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config_data, f)

        loaded = load_config(str(path))
        assert loaded.data.symbols == ["X", "Y"]
        assert loaded.execution.initial_capital == 75000.0
        assert loaded.sizing.method == "volatility"
        assert loaded.optimization.method == "mean_variance"
        assert loaded.optimization.rebalance_frequency == 25
        assert loaded.risk.max_drawdown_pct == 0.10

    def test_missing_optional_sections(self, tmp_path):
        """YAML without sizing/risk/optimization sections should use defaults."""
        config_data = {
            "data": {
                "symbols": ["SPY"],
                "start_date": "2022-01-01",
                "end_date": "2022-06-30",
            },
            "execution": {},
            "strategy": {"name": "sma_crossover"},
        }
        path = tmp_path / "minimal.yaml"
        with open(path, "w") as f:
            yaml.dump(config_data, f)

        loaded = load_config(str(path))
        assert loaded.sizing.method == "fixed_fraction"
        assert loaded.risk.enabled is True
        assert loaded.optimization.method == "none"

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

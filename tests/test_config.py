"""Unit tests for configuration system."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    BacktestConfig,
    BrokerConfig,
    DataConfig,
    ExecutionConfig,
    LiveConfig,
    LiveDataConfig,
    OptimizationConfig,
    PersistenceConfig,
    RiskConfig,
    SizingConfig,
    StrategyConfig,
    load_config,
)


class TestDataConfig:
    """Tests for DataConfig model."""

    def test_valid_config(self):
        """Test creating a valid DataConfig."""
        config = DataConfig(
            symbols=["AAPL", "MSFT"],
            start_date="2020-01-01",
            end_date="2023-12-31",
        )
        assert config.symbols == ["AAPL", "MSFT"]
        assert config.start_date == "2020-01-01"
        assert config.end_date == "2023-12-31"
        assert config.data_source == "csv"  # default value
        assert config.data_path is None  # default value

    def test_with_all_fields(self):
        """Test DataConfig with all fields specified."""
        config = DataConfig(
            symbols=["AAPL"],
            start_date="2020-01-01",
            end_date="2023-12-31",
            data_source="yfinance",
            data_path="/path/to/data",
        )
        assert config.data_source == "yfinance"
        assert config.data_path == "/path/to/data"

    def test_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            DataConfig(symbols=["AAPL"], start_date="2020-01-01")  # missing end_date


class TestExecutionConfig:
    """Tests for ExecutionConfig model."""

    def test_default_values(self):
        """Test ExecutionConfig with default values."""
        config = ExecutionConfig()
        assert config.initial_capital == 100000.0
        assert config.commission_pct == 0.001
        assert config.slippage_pct == 0.0005

    def test_custom_values(self):
        """Test ExecutionConfig with custom values."""
        config = ExecutionConfig(
            initial_capital=50000.0,
            commission_pct=0.002,
            slippage_pct=0.001,
        )
        assert config.initial_capital == 50000.0
        assert config.commission_pct == 0.002
        assert config.slippage_pct == 0.001

    def test_invalid_capital(self):
        """Test that negative capital raises ValidationError."""
        with pytest.raises(ValidationError):
            ExecutionConfig(initial_capital=-1000.0)

    def test_invalid_commission(self):
        """Test that negative commission raises ValidationError."""
        with pytest.raises(ValidationError):
            ExecutionConfig(commission_pct=-0.001)


class TestStrategyConfig:
    """Tests for StrategyConfig model."""

    def test_minimal_config(self):
        """Test StrategyConfig with only required fields."""
        config = StrategyConfig(name="sma_crossover")
        assert config.name == "sma_crossover"
        assert config.parameters == {}

    def test_with_parameters(self):
        """Test StrategyConfig with parameters."""
        config = StrategyConfig(
            name="sma_crossover",
            parameters={"short_window": 20, "long_window": 50},
        )
        assert config.parameters["short_window"] == 20
        assert config.parameters["long_window"] == 50


class TestBacktestConfig:
    """Tests for BacktestConfig model."""

    def test_full_config(self):
        """Test creating a complete BacktestConfig."""
        config = BacktestConfig(
            data=DataConfig(
                symbols=["AAPL"],
                start_date="2020-01-01",
                end_date="2023-12-31",
            ),
            execution=ExecutionConfig(initial_capital=100000.0),
            strategy=StrategyConfig(
                name="sma_crossover",
                parameters={"short_window": 20, "long_window": 50},
            ),
        )
        assert config.data.symbols == ["AAPL"]
        assert config.execution.initial_capital == 100000.0
        assert config.strategy.name == "sma_crossover"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid YAML config file."""
        config_data = {
            "data": {
                "symbols": ["AAPL", "MSFT"],
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "data_source": "csv",
                "data_path": "data/sample",
            },
            "execution": {
                "initial_capital": 100000.0,
                "commission_pct": 0.001,
                "slippage_pct": 0.0005,
            },
            "strategy": {
                "name": "sma_crossover",
                "parameters": {"short_window": 20, "long_window": 50},
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))

        assert config.data.symbols == ["AAPL", "MSFT"]
        assert config.data.start_date == "2020-01-01"
        assert config.execution.initial_capital == 100000.0
        assert config.strategy.name == "sma_crossover"
        assert config.strategy.parameters["short_window"] == 20

    def test_load_missing_file(self):
        """Test that loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/non/existent/path/config.yaml")

    def test_load_invalid_config(self, tmp_path):
        """Test that loading an invalid config raises ValidationError."""
        config_data = {
            "data": {
                "symbols": ["AAPL"],
                # missing required fields
            },
            "execution": {},
            "strategy": {"name": "test"},
        }

        config_file = tmp_path / "invalid_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValidationError):
            load_config(str(config_file))

    def test_load_actual_config_file(self):
        """Test loading the actual config file in configs/."""
        config_path = Path(__file__).parent.parent / "configs" / "backtest_config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
            assert config.data.symbols == ["AAPL", "MSFT", "GOOGL"]
            assert config.strategy.name == "sma_crossover"


class TestSizingConfig:
    """Tests for SizingConfig model."""

    def test_default_values(self):
        config = SizingConfig()
        assert config.method == "fixed_fraction"
        assert config.parameters == {}

    def test_valid_methods(self):
        for method in ["fixed_fraction", "volatility", "kelly"]:
            config = SizingConfig(method=method)
            assert config.method == method

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            SizingConfig(method="invalid")

    def test_with_parameters(self):
        config = SizingConfig(
            method="volatility",
            parameters={"risk_fraction": 0.03, "atr_period": 20},
        )
        assert config.parameters["risk_fraction"] == 0.03
        assert config.parameters["atr_period"] == 20


class TestRiskConfig:
    """Tests for RiskConfig model."""

    def test_default_values(self):
        config = RiskConfig()
        assert config.enabled is True
        assert config.max_position_pct == 0.10
        assert config.max_portfolio_exposure_pct == 1.0
        assert config.max_daily_loss_pct == 0.03
        assert config.max_drawdown_pct == 0.15
        assert config.max_open_positions == 20
        assert config.max_orders_per_day == 100

    def test_defaults_match_risk_limits(self):
        """Verify defaults match RiskLimits dataclass."""
        from src.risk.risk_manager import RiskLimits

        config = RiskConfig()
        limits = RiskLimits()
        assert config.max_position_pct == limits.max_position_pct
        assert config.max_portfolio_exposure_pct == limits.max_portfolio_exposure_pct
        assert config.max_daily_loss_pct == limits.max_daily_loss_pct
        assert config.max_drawdown_pct == limits.max_drawdown_pct
        assert config.max_open_positions == limits.max_open_positions
        assert config.max_orders_per_day == limits.max_orders_per_day

    def test_disabled(self):
        config = RiskConfig(enabled=False)
        assert config.enabled is False

    def test_invalid_pct_zero(self):
        with pytest.raises(ValidationError):
            RiskConfig(max_position_pct=0.0)

    def test_invalid_pct_above_one(self):
        with pytest.raises(ValidationError):
            RiskConfig(max_position_pct=1.5)

    def test_invalid_pct_negative(self):
        with pytest.raises(ValidationError):
            RiskConfig(max_daily_loss_pct=-0.01)

    def test_valid_boundary_values(self):
        config = RiskConfig(max_position_pct=1.0, max_daily_loss_pct=0.001)
        assert config.max_position_pct == 1.0
        assert config.max_daily_loss_pct == 0.001

    def test_invalid_max_open_positions_zero(self):
        with pytest.raises(ValidationError):
            RiskConfig(max_open_positions=0)


class TestOptimizationConfig:
    """Tests for OptimizationConfig model."""

    def test_default_values(self):
        config = OptimizationConfig()
        assert config.method == "none"
        assert config.rebalance_frequency == 20
        assert config.parameters == {}

    def test_valid_methods(self):
        for method in ["none", "mean_variance", "risk_parity"]:
            config = OptimizationConfig(method=method)
            assert config.method == method

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            OptimizationConfig(method="genetic")

    def test_invalid_rebalance_frequency(self):
        with pytest.raises(ValidationError):
            OptimizationConfig(rebalance_frequency=0)


class TestBacktestConfigExtended:
    """Tests for BacktestConfig with new optional sections."""

    def _base_data(self) -> DataConfig:
        return DataConfig(
            symbols=["AAPL"],
            start_date="2020-01-01",
            end_date="2023-12-31",
        )

    def test_backwards_compatible_no_new_sections(self):
        """Existing config without sizing/risk/optimization still works."""
        config = BacktestConfig(
            data=self._base_data(),
            execution=ExecutionConfig(),
            strategy=StrategyConfig(name="sma_crossover"),
        )
        assert config.sizing.method == "fixed_fraction"
        assert config.risk.enabled is True
        assert config.optimization.method == "none"

    def test_with_all_sections(self):
        config = BacktestConfig(
            data=self._base_data(),
            execution=ExecutionConfig(),
            strategy=StrategyConfig(name="sma_crossover"),
            sizing=SizingConfig(method="volatility"),
            risk=RiskConfig(enabled=False),
            optimization=OptimizationConfig(method="mean_variance"),
        )
        assert config.sizing.method == "volatility"
        assert config.risk.enabled is False
        assert config.optimization.method == "mean_variance"

    def test_load_yaml_without_new_sections(self, tmp_path):
        """YAML without sizing/risk/optimization loads with defaults."""
        config_data = {
            "data": {
                "symbols": ["AAPL"],
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
            },
            "execution": {},
            "strategy": {"name": "sma_crossover"},
        }
        config_file = tmp_path / "old_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        assert config.sizing.method == "fixed_fraction"
        assert config.risk.enabled is True
        assert config.optimization.method == "none"

    def test_load_yaml_with_new_sections(self, tmp_path):
        """YAML with sizing/risk/optimization loads correctly."""
        config_data = {
            "data": {
                "symbols": ["AAPL"],
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
            },
            "execution": {"initial_capital": 50000.0},
            "strategy": {"name": "sma_crossover", "parameters": {}},
            "sizing": {
                "method": "volatility",
                "parameters": {"risk_fraction": 0.03},
            },
            "risk": {
                "enabled": True,
                "max_drawdown_pct": 0.10,
            },
            "optimization": {
                "method": "none",
                "rebalance_frequency": 10,
            },
        }
        config_file = tmp_path / "extended_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        assert config.sizing.method == "volatility"
        assert config.sizing.parameters["risk_fraction"] == 0.03
        assert config.risk.max_drawdown_pct == 0.10
        assert config.optimization.rebalance_frequency == 10


# ---------------------------------------------------------------------------
# Phase 4: Live config models
# ---------------------------------------------------------------------------


class TestLiveDataConfig:
    def test_defaults(self):
        config = LiveDataConfig()
        assert config.feed_type == "websocket"
        assert config.url == ""
        assert config.symbols == []
        assert config.bar_interval_seconds == 60
        assert config.max_history == 5000
        assert config.reconnect_attempts == 10
        assert config.reconnect_delay == 1.0

    def test_custom_values(self):
        config = LiveDataConfig(
            feed_type="polling",
            url="ws://example.com",
            symbols=["AAPL", "MSFT"],
            bar_interval_seconds=300,
            max_history=1000,
            reconnect_attempts=5,
            reconnect_delay=2.0,
        )
        assert config.feed_type == "polling"
        assert config.url == "ws://example.com"
        assert config.symbols == ["AAPL", "MSFT"]
        assert config.bar_interval_seconds == 300

    def test_valid_feed_types(self):
        for ft in ["websocket", "polling"]:
            config = LiveDataConfig(feed_type=ft)
            assert config.feed_type == ft

    def test_invalid_feed_type(self):
        with pytest.raises(ValidationError):
            LiveDataConfig(feed_type="grpc")

    def test_invalid_bar_interval(self):
        with pytest.raises(ValidationError):
            LiveDataConfig(bar_interval_seconds=0)

    def test_invalid_max_history(self):
        with pytest.raises(ValidationError):
            LiveDataConfig(max_history=0)


class TestBrokerConfig:
    def test_defaults(self):
        config = BrokerConfig()
        assert config.broker_type == "paper"
        assert config.api_key == ""
        assert config.api_secret == ""
        assert config.base_url == ""
        assert config.paper_mode is True
        assert config.fill_delay_ms == 100
        assert config.slippage_model == "fixed"

    def test_valid_broker_types(self):
        for bt in ["paper", "alpaca", "ibkr"]:
            config = BrokerConfig(broker_type=bt)
            assert config.broker_type == bt

    def test_invalid_broker_type(self):
        with pytest.raises(ValidationError):
            BrokerConfig(broker_type="robinhood")

    def test_valid_slippage_models(self):
        for sm in ["fixed", "proportional", "volume"]:
            config = BrokerConfig(slippage_model=sm)
            assert config.slippage_model == sm

    def test_invalid_slippage_model(self):
        with pytest.raises(ValidationError):
            BrokerConfig(slippage_model="random")

    def test_invalid_fill_delay(self):
        with pytest.raises(ValidationError):
            BrokerConfig(fill_delay_ms=-1)

    def test_custom_values(self):
        config = BrokerConfig(
            broker_type="alpaca",
            api_key="key123",
            api_secret="secret456",
            base_url="https://api.alpaca.markets",
            paper_mode=False,
            fill_delay_ms=50,
            slippage_model="proportional",
        )
        assert config.api_key == "key123"
        assert config.paper_mode is False


class TestPersistenceConfig:
    def test_defaults(self):
        config = PersistenceConfig()
        assert config.enabled is True
        assert config.state_dir == "state/"
        assert config.save_interval_seconds == 300
        assert config.max_snapshots == 10

    def test_custom_values(self):
        config = PersistenceConfig(
            enabled=False,
            state_dir="/tmp/state",
            save_interval_seconds=60,
            max_snapshots=5,
        )
        assert config.enabled is False
        assert config.state_dir == "/tmp/state"

    def test_invalid_save_interval(self):
        with pytest.raises(ValidationError):
            PersistenceConfig(save_interval_seconds=0)

    def test_invalid_max_snapshots(self):
        with pytest.raises(ValidationError):
            PersistenceConfig(max_snapshots=0)


class TestLiveConfig:
    def test_defaults(self):
        config = LiveConfig()
        assert config.data.feed_type == "websocket"
        assert config.broker.broker_type == "paper"
        assert config.persistence.enabled is True

    def test_nested_override(self):
        config = LiveConfig(
            data=LiveDataConfig(url="ws://test", bar_interval_seconds=300),
            broker=BrokerConfig(broker_type="alpaca"),
        )
        assert config.data.url == "ws://test"
        assert config.data.bar_interval_seconds == 300
        assert config.broker.broker_type == "alpaca"


class TestBacktestConfigWithLive:
    def _base_data(self) -> DataConfig:
        return DataConfig(
            symbols=["AAPL"],
            start_date="2020-01-01",
            end_date="2023-12-31",
        )

    def test_backwards_compatible_no_live(self):
        """Existing configs without live section still work."""
        config = BacktestConfig(
            data=self._base_data(),
            execution=ExecutionConfig(),
            strategy=StrategyConfig(name="sma_crossover"),
        )
        assert config.live.data.feed_type == "websocket"
        assert config.live.broker.broker_type == "paper"

    def test_with_live_section(self):
        config = BacktestConfig(
            data=self._base_data(),
            execution=ExecutionConfig(),
            strategy=StrategyConfig(name="sma_crossover"),
            live=LiveConfig(
                data=LiveDataConfig(
                    url="ws://stream.example.com",
                    symbols=["AAPL"],
                    bar_interval_seconds=300,
                ),
            ),
        )
        assert config.live.data.url == "ws://stream.example.com"
        assert config.live.data.bar_interval_seconds == 300

    def test_load_yaml_without_live(self, tmp_path):
        config_data = {
            "data": {
                "symbols": ["AAPL"],
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
            },
            "execution": {},
            "strategy": {"name": "sma_crossover"},
        }
        config_file = tmp_path / "no_live.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        assert config.live.data.feed_type == "websocket"
        assert config.live.broker.broker_type == "paper"

    def test_load_yaml_with_live(self, tmp_path):
        config_data = {
            "data": {
                "symbols": ["AAPL"],
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
            },
            "execution": {},
            "strategy": {"name": "sma_crossover"},
            "live": {
                "data": {
                    "feed_type": "websocket",
                    "url": "ws://stream.example.com",
                    "bar_interval_seconds": 300,
                },
                "broker": {
                    "broker_type": "paper",
                    "fill_delay_ms": 50,
                },
            },
        }
        config_file = tmp_path / "with_live.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        assert config.live.data.url == "ws://stream.example.com"
        assert config.live.data.bar_interval_seconds == 300
        assert config.live.broker.fill_delay_ms == 50

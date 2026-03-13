"""Configuration system using Pydantic for validation."""

from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data sources and date ranges."""

    symbols: list[str]
    start_date: str
    end_date: str
    data_source: str = "csv"  # csv, yfinance, database
    data_path: str | None = None


class ExecutionConfig(BaseModel):
    """Configuration for execution parameters."""

    initial_capital: float = Field(default=100000.0, gt=0)
    commission_pct: float = Field(default=0.001, ge=0)  # 0.1%
    slippage_pct: float = Field(default=0.0005, ge=0)  # 0.05%


class StrategyConfig(BaseModel):
    """Configuration for trading strategy."""

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class SizingConfig(BaseModel):
    """Configuration for position sizing."""

    method: str = "fixed_fraction"
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"fixed_fraction", "volatility", "kelly"}
        if v not in allowed:
            msg = f"sizing method must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v


class RiskConfig(BaseModel):
    """Configuration for risk management limits."""

    enabled: bool = True
    max_position_pct: float = Field(default=0.10, gt=0, le=1.0)
    max_portfolio_exposure_pct: float = Field(default=1.0, gt=0, le=1.0)
    max_daily_loss_pct: float = Field(default=0.03, gt=0, le=1.0)
    max_drawdown_pct: float = Field(default=0.15, gt=0, le=1.0)
    max_open_positions: int = Field(default=20, ge=1)
    max_orders_per_day: int = Field(default=100, ge=1)


class OptimizationConfig(BaseModel):
    """Configuration for portfolio optimization (placeholder for Week 5)."""

    method: str = "none"
    rebalance_frequency: int = Field(default=20, ge=1)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"none", "mean_variance", "risk_parity"}
        if v not in allowed:
            msg = f"optimization method must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v


class FeatureConfig(BaseModel):
    """Configuration for feature engineering pipeline."""

    technical: list[dict[str, Any]] = Field(default_factory=list)
    statistical: list[dict[str, Any]] = Field(default_factory=list)
    target: dict[str, Any] = Field(
        default_factory=lambda: {"horizon": 5, "type": "direction"}
    )


class MLConfig(BaseModel):
    """Configuration for ML model."""

    model: str = "xgboost"
    mode: str = "classification"
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed = {"xgboost", "lightgbm"}
        if v not in allowed:
            msg = f"ml model must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"classification", "regression"}
        if v not in allowed:
            msg = f"ml mode must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v


class ValidationConfig(BaseModel):
    """Configuration for CPCV validation."""

    method: str = "cpcv"
    n_splits: int = Field(default=6, ge=3)
    n_test_splits: int = Field(default=2, ge=1)
    purge_window: int = Field(default=5, ge=0)
    embargo_pct: float = Field(default=0.01, ge=0.0, le=1.0)


class RegimeConfig(BaseModel):
    """Configuration for HMM regime detection."""

    enabled: bool = False
    n_regimes: int = Field(default=3, ge=2)
    vol_window: int = Field(default=20, ge=2)


class TrackingConfig(BaseModel):
    """Configuration for MLflow experiment tracking."""

    enabled: bool = False
    experiment_name: str = "trading_system"


class LiveDataConfig(BaseModel):
    """Configuration for live data feeds."""

    feed_type: str = "websocket"
    url: str = ""
    symbols: list[str] = Field(default_factory=list)
    bar_interval_seconds: int = Field(default=60, gt=0)
    max_history: int = Field(default=5000, gt=0)
    reconnect_attempts: int = Field(default=10, ge=0)
    reconnect_delay: float = Field(default=1.0, gt=0)

    @field_validator("feed_type")
    @classmethod
    def validate_feed_type(cls, v: str) -> str:
        allowed = {"websocket", "polling", "alpaca"}
        if v not in allowed:
            msg = f"feed_type must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v


class BrokerConfig(BaseModel):
    """Configuration for broker connection."""

    broker_type: str = "paper"
    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    paper_mode: bool = True
    fill_delay_ms: int = Field(default=100, ge=0)
    slippage_model: str = "fixed"

    @field_validator("broker_type")
    @classmethod
    def validate_broker_type(cls, v: str) -> str:
        allowed = {"paper", "alpaca", "ibkr"}
        if v not in allowed:
            msg = f"broker_type must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("slippage_model")
    @classmethod
    def validate_slippage_model(cls, v: str) -> str:
        allowed = {"fixed", "proportional", "volume"}
        if v not in allowed:
            msg = f"slippage_model must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v


class PersistenceConfig(BaseModel):
    """Configuration for state persistence."""

    enabled: bool = True
    state_dir: str = "state/"
    save_interval_seconds: int = Field(default=300, gt=0)
    max_snapshots: int = Field(default=10, ge=1)


class LiveConfig(BaseModel):
    """Top-level live/paper trading configuration."""

    data: LiveDataConfig = Field(default_factory=LiveDataConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)


class BacktestConfig(BaseModel):
    """Main configuration container for backtesting."""

    data: DataConfig
    execution: ExecutionConfig
    strategy: StrategyConfig
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)


def load_config(path: str) -> BacktestConfig:
    """
    Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Validated BacktestConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config values are invalid
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return BacktestConfig(**data)

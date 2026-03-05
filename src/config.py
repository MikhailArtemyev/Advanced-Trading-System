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


class BacktestConfig(BaseModel):
    """Main configuration container for backtesting."""

    data: DataConfig
    execution: ExecutionConfig
    strategy: StrategyConfig
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


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

"""Configuration system using Pydantic for validation."""

import os
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

    def model_post_init(self, __context: Any) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("ALPACA_API_KEY", "")
        if not self.api_secret:
            self.api_secret = os.environ.get("ALPACA_API_SECRET", "")


class DatabaseConfig(BaseModel):
    """Configuration for SQLite persistence."""

    enabled: bool = True
    db_url: str = "sqlite:///trading.db"
    save_interval_seconds: int = Field(default=300, gt=0)

    def model_post_init(self, __context: Any) -> None:
        env_url = os.environ.get("DATABASE_URL", "")
        if env_url and self.db_url == "sqlite:///trading.db":
            self.db_url = env_url


class AlertConfig(BaseModel):
    """Configuration for alerting and notifications."""

    enabled: bool = False
    channels: list[dict[str, Any]] = Field(default_factory=list)
    min_level: str = "warning"
    cooldown_seconds: float = Field(default=300.0, ge=0)

    @field_validator("min_level")
    @classmethod
    def validate_min_level(cls, v: str) -> str:
        allowed = {"info", "warning", "critical"}
        if v not in allowed:
            msg = f"min_level must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v

    def model_post_init(self, __context: Any) -> None:
        for channel in self.channels:
            ch_type = channel.get("type", "")
            if ch_type == "slack" and not channel.get("webhook_url"):
                channel["webhook_url"] = os.environ.get("SLACK_WEBHOOK_URL", "")
            elif ch_type == "email" and not channel.get("password"):
                channel["password"] = os.environ.get("SMTP_PASSWORD", "")


class LiveConfig(BaseModel):
    """Top-level live/paper trading configuration."""

    data: LiveDataConfig = Field(default_factory=LiveDataConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)


class BacktestConfig(BaseModel):
    """Main configuration container for backtesting."""

    data: DataConfig
    execution: ExecutionConfig
    strategy: StrategyConfig
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)

    def validate_for_live(self) -> list[str]:
        """Validate configuration for live/paper trading.

        Returns a list of issue strings prefixed with ``ERROR:`` or
        ``WARNING:``. An empty list means all checks passed.
        """
        issues: list[str] = []

        # API credentials for non-paper brokers
        broker = self.live.broker
        if broker.broker_type != "paper":
            if not broker.api_key:
                issues.append(
                    f"ERROR: {broker.broker_type} broker requires api_key "
                    "(set in YAML or ALPACA_API_KEY env var)"
                )
            if not broker.api_secret:
                issues.append(
                    f"ERROR: {broker.broker_type} broker requires api_secret "
                    "(set in YAML or ALPACA_API_SECRET env var)"
                )

        # Symbols
        if not self.data.symbols:
            issues.append("ERROR: No symbols configured in data.symbols")
        for symbol in self.data.symbols:
            if not symbol or not symbol.strip():
                issues.append("ERROR: Empty symbol found in data.symbols")

        # Alert channels
        alerts = self.live.alerts
        if alerts.enabled and not alerts.channels:
            issues.append("WARNING: Alerts are enabled but no channels configured")
        if alerts.enabled:
            for ch in alerts.channels:
                ch_type = ch.get("type", "")
                if ch_type == "slack" and not ch.get("webhook_url"):
                    issues.append(
                        "WARNING: Slack alert channel has no webhook_url "
                        "(set in YAML or SLACK_WEBHOOK_URL env var)"
                    )
                elif ch_type == "email" and not ch.get("password"):
                    issues.append(
                        "WARNING: Email alert channel has no password "
                        "(set in YAML or SMTP_PASSWORD env var)"
                    )

        return issues


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

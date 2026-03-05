"""Tests for run_backtest.py factory functions."""

import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_backtest import (
    build_data_handler,
    build_position_sizer,
    build_risk_manager,
)
from src.config import (
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    RiskConfig,
    SizingConfig,
    StrategyConfig,
)
from src.risk.position_sizer import (
    FixedFractionSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
)


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


class TestBuildPositionSizer:
    """Tests for build_position_sizer factory."""

    def test_default_fixed_fraction(self) -> None:
        config = _base_config()
        sizer = build_position_sizer(config)
        assert isinstance(sizer, FixedFractionSizer)

    def test_fixed_fraction_with_params(self) -> None:
        config = _base_config(
            sizing=SizingConfig(method="fixed_fraction", parameters={"fraction": 0.20})
        )
        sizer = build_position_sizer(config)
        assert isinstance(sizer, FixedFractionSizer)
        assert sizer.fraction == 0.20

    def test_volatility_sizer(self) -> None:
        config = _base_config(
            sizing=SizingConfig(
                method="volatility",
                parameters={"risk_fraction": 0.03},
            )
        )
        sizer = build_position_sizer(config)
        assert isinstance(sizer, VolatilityBasedSizer)
        assert sizer.risk_fraction == 0.03

    def test_kelly_sizer(self) -> None:
        config = _base_config(sizing=SizingConfig(method="kelly"))
        sizer = build_position_sizer(config)
        assert isinstance(sizer, KellyCriterionSizer)


class TestBuildRiskManager:
    """Tests for build_risk_manager factory."""

    def test_default_enabled(self) -> None:
        config = _base_config()
        rm = build_risk_manager(config)
        assert rm is not None
        assert rm.limits.max_position_pct == 0.10

    def test_disabled_returns_none(self) -> None:
        config = _base_config(risk=RiskConfig(enabled=False))
        rm = build_risk_manager(config)
        assert rm is None

    def test_custom_limits(self) -> None:
        config = _base_config(
            risk=RiskConfig(max_drawdown_pct=0.05, max_open_positions=5)
        )
        rm = build_risk_manager(config)
        assert rm is not None
        assert rm.limits.max_drawdown_pct == 0.05
        assert rm.limits.max_open_positions == 5


class TestBuildDataHandler:
    """Tests for build_data_handler factory."""

    def test_unknown_source_raises(self) -> None:
        config = _base_config(
            data=DataConfig(
                symbols=["AAPL"],
                start_date="2022-01-01",
                end_date="2022-12-31",
                data_source="database",
            )
        )
        with pytest.raises(ValueError, match="Unknown data_source"):
            build_data_handler(config)

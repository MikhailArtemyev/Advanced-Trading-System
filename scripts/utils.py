"""Shared utilities for trading scripts.

Builder functions for constructing components from config, synthetic
data generation, and the strategy registry.
"""

import math
from datetime import datetime, timedelta

from src.config import BacktestConfig
from src.data.alpaca_handler import AlpacaBarDataHandler
from src.data.data_handler import DataHandler, HistoricalCSVDataHandler
from src.data.yfinance_handler import YFinanceDataHandler
from src.live.bar_aggregator import Tick
from src.optimization.base_optimizer import PortfolioOptimizer
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer
from src.risk.position_sizer import (
    FixedFractionSizer,
    KellyCriterionSizer,
    PositionSizer,
    VolatilityBasedSizer,
)
from src.risk.risk_manager import RiskLimits, RiskManager
from src.strategy.base_strategy import Strategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.ml_filtered import MLFilteredStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.multi_asset_sma import MultiAssetSMAStrategy
from src.strategy.pairs_trading import PairsTradingStrategy
from src.strategy.sma_crossover import SMACrossoverStrategy

STRATEGY_MAP: dict[str, type[Strategy]] = {
    "sma_crossover": SMACrossoverStrategy,
    "multi_asset_sma": MultiAssetSMAStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "pairs_trading": PairsTradingStrategy,
    "ml_filtered": MLFilteredStrategy,
}


def build_data_handler(config: BacktestConfig) -> DataHandler:
    """Build data handler based on config.data.data_source."""
    if config.data.data_source == "yfinance":
        return YFinanceDataHandler(
            symbols=config.data.symbols,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            cache_dir=config.data.data_path or "data/cache",
        )
    elif config.data.data_source == "alpaca":
        bar_secs = config.live.data.bar_interval_seconds
        tf_map = {60: "1Min", 300: "5Min", 900: "15Min", 3600: "1Hour", 86400: "1Day"}
        timeframe = tf_map.get(bar_secs, "5Min")
        return AlpacaBarDataHandler(
            symbols=config.data.symbols,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            timeframe=timeframe,
            api_key=config.live.broker.api_key,
            api_secret=config.live.broker.api_secret,
            feed="iex",
            cache_dir=config.data.data_path or "data/alpaca_cache",
        )
    elif config.data.data_source == "csv":
        return HistoricalCSVDataHandler(
            data_path=config.data.data_path or "data/sample",
            symbols=config.data.symbols,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
        )
    else:
        msg = f"Unknown data_source: {config.data.data_source}"
        raise ValueError(msg)


def build_position_sizer(config: BacktestConfig) -> PositionSizer:
    """Build position sizer based on config.sizing."""
    method = config.sizing.method
    params = config.sizing.parameters

    if method == "fixed_fraction":
        return FixedFractionSizer(**params)
    elif method == "volatility":
        return VolatilityBasedSizer(**params)
    elif method == "kelly":
        return KellyCriterionSizer(**params)
    else:
        msg = f"Unknown sizing method: {method}"
        raise ValueError(msg)


def build_risk_manager(config: BacktestConfig) -> RiskManager | None:
    """Build risk manager based on config.risk. Returns None if disabled."""
    if not config.risk.enabled:
        return None

    limits = RiskLimits(
        max_position_pct=config.risk.max_position_pct,
        max_portfolio_exposure_pct=config.risk.max_portfolio_exposure_pct,
        max_daily_loss_pct=config.risk.max_daily_loss_pct,
        max_drawdown_pct=config.risk.max_drawdown_pct,
        max_open_positions=config.risk.max_open_positions,
        max_orders_per_day=config.risk.max_orders_per_day,
    )
    return RiskManager(limits=limits)


def build_optimizer(config: BacktestConfig) -> PortfolioOptimizer | None:
    """Build portfolio optimizer based on config.optimization.method.

    Returns None if method is 'none'.
    """
    method = config.optimization.method
    params = config.optimization.parameters

    if method == "none":
        return None
    elif method == "mean_variance":
        return MeanVarianceOptimizer(**params)
    elif method == "risk_parity":
        return RiskParityOptimizer(**params)
    else:
        msg = f"Unknown optimization method: {method}"
        raise ValueError(msg)


def build_strategy(
    config: BacktestConfig,
    optimizer: PortfolioOptimizer | None = None,
    data_handler: DataHandler | None = None,
) -> Strategy:
    """Build strategy based on config and optimizer.

    Auto-selects MultiAssetSMAStrategy when multiple symbols AND an
    optimizer are configured (for sma_crossover).
    """
    symbols = config.data.symbols
    params = dict(config.strategy.parameters)
    name = config.strategy.name

    if name == "momentum":
        return MomentumStrategy(symbols=symbols, parameters=params)

    if name == "mean_reversion":
        return MeanReversionStrategy(symbols=symbols, parameters=params)

    if name == "pairs_trading":
        return PairsTradingStrategy(symbols=symbols, parameters=params)

    if name == "ml_filtered":
        return MLFilteredStrategy(symbols=symbols, parameters=params)

    # Default: sma_crossover (with multi-asset variant when optimizer present)
    if len(symbols) > 1 and optimizer is not None:
        params.setdefault(
            "rebalance_frequency", config.optimization.rebalance_frequency
        )
        return MultiAssetSMAStrategy(
            symbols=symbols,
            parameters=params,
            optimizer=optimizer,
        )

    return SMACrossoverStrategy(
        symbols=symbols,
        parameters=params,
    )


def generate_synthetic_ticks(
    symbols: list[str],
    duration_minutes: int = 30,
    bar_interval_seconds: int = 60,
    ticks_per_second: float = 2.0,
    base_prices: dict[str, float] | None = None,
) -> list[Tick]:
    """Generate synthetic ticks for demo paper trading.

    Creates realistic-looking price movements with a sine wave trend
    so SMA crossover strategies can generate signals.

    Args:
        symbols: List of symbols to generate ticks for
        duration_minutes: Total duration of synthetic data
        bar_interval_seconds: Interval between bars (unused, kept for API compat)
        ticks_per_second: Number of ticks per second per symbol
        base_prices: Optional symbol -> base price mapping. Symbols not in
            this dict default to 100.0.
    """
    known_prices = {"AAPL": 175.0, "MSFT": 380.0, "GOOGL": 140.0}
    if base_prices is not None:
        known_prices.update(base_prices)

    ticks: list[Tick] = []
    total_seconds = duration_minutes * 60
    dt = 1.0 / ticks_per_second
    start = datetime.now()

    for i in range(int(total_seconds * ticks_per_second)):
        t = i * dt
        timestamp = start + timedelta(seconds=t)
        for symbol in symbols:
            base = known_prices.get(symbol, 100.0)
            # Sine wave for crossover + small noise
            trend = base * 0.03 * math.sin(2 * math.pi * t / (5 * 60))
            noise = base * 0.001 * math.sin(t * 17.3 + hash(symbol) % 100)
            price = base + trend + noise
            ticks.append(
                Tick(symbol=symbol, price=price, timestamp=timestamp, volume=100)
            )

    return ticks

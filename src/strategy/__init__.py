"""Strategy module for the trading system.

Contains the base Strategy class and concrete strategy implementations.
"""

from .base_strategy import Strategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .multi_asset_sma import MultiAssetSMAStrategy
from .pairs_trading import PairsTradingStrategy
from .sma_crossover import SMACrossoverStrategy

__all__ = [
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MultiAssetSMAStrategy",
    "PairsTradingStrategy",
    "SMACrossoverStrategy",
    "Strategy",
]

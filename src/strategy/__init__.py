"""Strategy module for the trading system.

Contains the base Strategy class and concrete strategy implementations.
"""

from .base_strategy import Strategy
from .multi_asset_sma import MultiAssetSMAStrategy
from .sma_crossover import SMACrossoverStrategy

__all__ = [
    "MultiAssetSMAStrategy",
    "SMACrossoverStrategy",
    "Strategy",
]

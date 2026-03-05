"""Abstract base classes for feature engineering.

Defines the FeatureGenerator contract that all feature implementations must
satisfy, and the FeatureResult dataclass they return.

Design principles:
- Generators are stateless: no instance state is mutated during compute()
- NaN rows from lookback periods are retained — the pipeline handles trimming
- OHLCV contract: columns open, high, low, close, volume; DatetimeIndex
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureResult:
    """Result of feature computation for one symbol.

    Attributes:
        features: DataFrame with feature columns, indexed by date.
            Rows with NaN (from lookback) are retained.
        feature_names: List of column names produced by this generator.
        method: Short identifier for the feature computation method.
    """

    features: pd.DataFrame
    feature_names: list[str]
    method: str


class FeatureGenerator(ABC):
    """Abstract base class for feature generators.

    Each generator produces one or more feature columns from OHLCV data.
    Generators must be stateless — all state lives in the input data.

    Subclasses must implement:
        compute(ohlcv) -> FeatureResult
        lookback (property) -> int
    """

    @abstractmethod
    def compute(self, ohlcv: pd.DataFrame) -> FeatureResult:
        """Compute features from OHLCV data.

        Args:
            ohlcv: DataFrame with columns: open, high, low, close, volume.
                   Index must be DatetimeIndex.

        Returns:
            FeatureResult with computed feature columns.
            Rows with NaN (from lookback requirements) are retained —
            the pipeline handles trimming.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def lookback(self) -> int:
        """Minimum number of bars required before the first valid output."""
        raise NotImplementedError

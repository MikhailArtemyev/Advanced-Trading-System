"""Base classes for portfolio optimization.

Defines the AllocationResult dataclass and PortfolioOptimizer abstract base
class that all concrete optimizers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AllocationResult:
    """Result of a portfolio optimization.

    Attributes:
        weights: Symbol → target weight mapping, sums to 1.0
        method: Name of the optimization method used
        expected_return: Annualized expected portfolio return
        expected_volatility: Annualized expected portfolio volatility
        sharpe_ratio: Expected Sharpe ratio
        notes: Additional information about the optimization
    """

    weights: dict[str, float] = field(default_factory=dict)
    method: str = ""
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    notes: str = ""


class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers.

    Concrete implementations compute target allocation weights across
    multiple assets given historical return data.
    """

    @abstractmethod
    def optimize(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
        current_weights: dict[str, float],
    ) -> AllocationResult:
        """Compute optimal portfolio weights.

        Args:
            symbols: Assets to allocate across
            returns_df: Historical daily returns (columns = symbols)
            current_weights: Current portfolio weights (symbol → weight)

        Returns:
            AllocationResult with target weights and portfolio metrics
        """

    def _equal_weight(self, symbols: list[str], reason: str) -> AllocationResult:
        """Fallback to equal-weight allocation.

        Args:
            symbols: Assets to allocate across
            reason: Why the optimizer fell back to equal weight

        Returns:
            AllocationResult with equal weights
        """
        n = len(symbols)
        w = 1.0 / n if n > 0 else 0.0
        return AllocationResult(
            weights=dict.fromkeys(symbols, w),
            method=f"{self._method_name}_equal_weight",
            notes=f"equal weight fallback: {reason}",
        )

    def _single_asset(self, symbol: str) -> AllocationResult:
        """Allocation for a single asset (100% weight).

        Args:
            symbol: The single asset

        Returns:
            AllocationResult with 100% weight on the asset
        """
        return AllocationResult(
            weights={symbol: 1.0},
            method=self._method_name,
            notes="single asset",
        )

    @property
    def _method_name(self) -> str:
        """Return the optimizer's method name for AllocationResult."""
        raise NotImplementedError

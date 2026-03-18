"""Rolling correlation tracker for multi-asset portfolios.

Tracks rolling-window correlations between asset price series.
Uses only data fed via update() to avoid look-ahead bias.
"""

import numpy as np
import pandas as pd


class CorrelationTracker:
    """Tracks rolling correlations between asset price series.

    Maintains a rolling window of prices for each symbol and computes
    pairwise Pearson correlations on demand.

    Attributes:
        window: Rolling window size for correlation calculation
        min_periods: Minimum observations required for valid correlation
    """

    def __init__(
        self,
        window: int = 60,
        min_periods: int = 30,
    ) -> None:
        """Initialize the correlation tracker.

        Args:
            window: Rolling window size in bars (default 60)
            min_periods: Minimum observations for valid correlation (default 30)
        """
        self.window = window
        self.min_periods = min_periods
        self._prices: dict[str, list[float]] = {}

    def update(self, symbol: str, price: float) -> None:
        """Record a price observation for a symbol.

        Maintains only the last ``window`` prices for memory efficiency.

        Args:
            symbol: The instrument symbol
            price: The current price
        """
        if symbol not in self._prices:
            self._prices[symbol] = []
        self._prices[symbol].append(price)
        # Trim to window size
        if len(self._prices[symbol]) > self.window:
            self._prices[symbol] = self._prices[symbol][-self.window :]

    def get_correlation(self, symbol_a: str, symbol_b: str) -> float | None:
        """Return the rolling Pearson correlation between two symbols.

        Computes correlation on the returns (percentage changes) of the
        price series, not the raw prices.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol

        Returns:
            Correlation coefficient, or None if insufficient data.
            Returns 1.0 if symbol_a == symbol_b.
            Returns 0.0 if either series has zero variance.
        """
        if symbol_a == symbol_b:
            return 1.0

        if symbol_a not in self._prices or symbol_b not in self._prices:
            return None

        prices_a = self._prices[symbol_a]
        prices_b = self._prices[symbol_b]

        # Need at least min_periods prices to compute returns
        if len(prices_a) < self.min_periods or len(prices_b) < self.min_periods:
            return None

        # Use the overlapping tail
        n = min(len(prices_a), len(prices_b))
        arr_a = np.array(prices_a[-n:])
        arr_b = np.array(prices_b[-n:])

        # Compute returns
        returns_a = np.diff(arr_a) / arr_a[:-1]
        returns_b = np.diff(arr_b) / arr_b[:-1]

        if len(returns_a) < 2:
            return None

        # Check for zero variance
        if np.std(returns_a) == 0.0 or np.std(returns_b) == 0.0:
            return 0.0

        corr_matrix = np.corrcoef(returns_a, returns_b)
        return float(corr_matrix[0, 1])

    def calculate_matrix(self) -> pd.DataFrame:
        """Compute the full pairwise correlation matrix.

        Returns a symmetric DataFrame with symbols as both index and columns.
        Cells are NaN where either symbol has fewer than min_periods observations.

        Returns:
            Correlation matrix as a pandas DataFrame
        """
        syms = sorted(self._prices.keys())
        n = len(syms)

        if n == 0:
            return pd.DataFrame()

        matrix = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(i, n):
                corr = self.get_correlation(syms[i], syms[j])
                if corr is not None:
                    matrix[i, j] = corr
                    matrix[j, i] = corr

        return pd.DataFrame(matrix, index=syms, columns=syms)

    @property
    def symbols(self) -> list[str]:
        """Return list of symbols currently being tracked."""
        return sorted(self._prices.keys())

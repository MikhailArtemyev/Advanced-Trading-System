"""Unit tests for the CorrelationTracker class."""

import math

import numpy as np
import pandas as pd

from src.portfolio.correlation import CorrelationTracker


class TestCorrelationTrackerInit:
    """Tests for initialization."""

    def test_default_parameters(self):
        """Default window=60, min_periods=30."""
        tracker = CorrelationTracker()
        assert tracker.window == 60
        assert tracker.min_periods == 30

    def test_custom_parameters(self):
        """Custom window and min_periods."""
        tracker = CorrelationTracker(window=20, min_periods=10)
        assert tracker.window == 20
        assert tracker.min_periods == 10

    def test_empty_state(self):
        """New tracker has no symbols and empty matrix."""
        tracker = CorrelationTracker()
        assert tracker.symbols == []
        matrix = tracker.calculate_matrix()
        assert matrix.empty


class TestCorrelationTrackerUpdate:
    """Tests for price updates."""

    def test_single_update(self):
        """Single update registers the symbol."""
        tracker = CorrelationTracker()
        tracker.update("AAPL", 150.0)
        assert "AAPL" in tracker.symbols

    def test_multiple_updates_same_symbol(self):
        """Multiple updates accumulate prices."""
        tracker = CorrelationTracker(window=100, min_periods=3)
        for price in [150.0, 151.0, 152.0, 153.0]:
            tracker.update("AAPL", price)
        assert "AAPL" in tracker.symbols

    def test_multiple_symbols(self):
        """Tracks multiple symbols independently."""
        tracker = CorrelationTracker()
        tracker.update("AAPL", 150.0)
        tracker.update("MSFT", 300.0)
        tracker.update("GOOGL", 100.0)
        assert sorted(tracker.symbols) == ["AAPL", "GOOGL", "MSFT"]

    def test_window_trimming(self):
        """Only the last `window` prices are retained."""
        tracker = CorrelationTracker(window=5, min_periods=2)
        for i in range(20):
            tracker.update("AAPL", 100.0 + i)
        # Internal list should be trimmed to 5
        assert len(tracker._prices["AAPL"]) == 5
        assert tracker._prices["AAPL"][0] == 115.0  # 20 - 5 = 15th price

    def test_symbols_property(self):
        """symbols property returns sorted list."""
        tracker = CorrelationTracker()
        tracker.update("MSFT", 300.0)
        tracker.update("AAPL", 150.0)
        assert tracker.symbols == ["AAPL", "MSFT"]


class TestCorrelationTrackerGetCorrelation:
    """Tests for pairwise correlation."""

    def test_perfect_positive_correlation(self):
        """Two symbols with proportional price moves -> correlation near 1.0."""
        tracker = CorrelationTracker(window=50, min_periods=10)
        np.random.seed(42)
        base = 100.0
        for _ in range(40):
            change = np.random.randn() * 2
            base += change
            tracker.update("A", base)
            tracker.update("B", base * 2)  # Perfectly proportional

        corr = tracker.get_correlation("A", "B")
        assert corr is not None
        assert corr > 0.99

    def test_perfect_negative_correlation(self):
        """Two symbols with opposite price moves -> correlation near -1.0."""
        tracker = CorrelationTracker(window=50, min_periods=10)
        np.random.seed(42)
        base = 100.0
        for _ in range(40):
            change = np.random.randn() * 2
            base += change
            tracker.update("A", base)
            tracker.update("B", 200.0 - base)  # Inverse

        corr = tracker.get_correlation("A", "B")
        assert corr is not None
        assert corr < -0.95

    def test_low_correlation(self):
        """Independent random series have low correlation."""
        tracker = CorrelationTracker(window=100, min_periods=30)
        np.random.seed(42)
        price_a = 100.0
        price_b = 100.0
        for _ in range(80):
            price_a += np.random.randn()
            price_b += np.random.randn()
            tracker.update("A", price_a)
            tracker.update("B", price_b)

        corr = tracker.get_correlation("A", "B")
        assert corr is not None
        assert abs(corr) < 0.5  # Not perfectly correlated

    def test_insufficient_data_returns_none(self):
        """Fewer than min_periods observations returns None."""
        tracker = CorrelationTracker(window=60, min_periods=30)
        for i in range(10):
            tracker.update("A", 100.0 + i)
            tracker.update("B", 200.0 + i)

        assert tracker.get_correlation("A", "B") is None

    def test_same_symbol_returns_one(self):
        """Correlation of a symbol with itself is 1.0."""
        tracker = CorrelationTracker()
        tracker.update("A", 100.0)
        assert tracker.get_correlation("A", "A") == 1.0

    def test_unknown_symbol_returns_none(self):
        """Unknown symbol returns None."""
        tracker = CorrelationTracker()
        tracker.update("A", 100.0)
        assert tracker.get_correlation("A", "B") is None
        assert tracker.get_correlation("B", "A") is None

    def test_constant_prices_returns_zero(self):
        """Constant price series (zero variance) returns 0.0."""
        tracker = CorrelationTracker(window=50, min_periods=10)
        for _ in range(30):
            tracker.update("A", 100.0)  # Constant
            tracker.update("B", 200.0 + np.random.randn())

        corr = tracker.get_correlation("A", "B")
        assert corr == 0.0


class TestCorrelationTrackerMatrix:
    """Tests for full correlation matrix."""

    def _build_tracker(self, n_obs: int = 40) -> CorrelationTracker:
        """Helper: build tracker with correlated data."""
        tracker = CorrelationTracker(window=60, min_periods=10)
        np.random.seed(42)
        base = 100.0
        for _ in range(n_obs):
            change = np.random.randn()
            base += change
            tracker.update("AAPL", base)
            tracker.update("MSFT", base * 1.5 + np.random.randn() * 0.1)
            tracker.update("GOOGL", 200.0 + np.random.randn() * 5)
        return tracker

    def test_matrix_shape(self):
        """Matrix should be N x N where N = number of symbols."""
        tracker = self._build_tracker()
        matrix = tracker.calculate_matrix()
        assert matrix.shape == (3, 3)

    def test_matrix_symmetry(self):
        """Correlation matrix is symmetric."""
        tracker = self._build_tracker()
        matrix = tracker.calculate_matrix()
        pd.testing.assert_frame_equal(matrix, matrix.T)

    def test_diagonal_is_one(self):
        """Diagonal elements are 1.0."""
        tracker = self._build_tracker()
        matrix = tracker.calculate_matrix()
        for sym in matrix.columns:
            assert matrix.loc[sym, sym] == 1.0

    def test_matrix_with_insufficient_data(self):
        """Cells with insufficient data are NaN."""
        tracker = CorrelationTracker(window=60, min_periods=30)
        for i in range(5):  # Only 5 data points
            tracker.update("A", 100.0 + i)
            tracker.update("B", 200.0 + i)

        matrix = tracker.calculate_matrix()
        # Diagonal should still be 1.0 (same symbol)
        assert matrix.loc["A", "A"] == 1.0
        # Off-diagonal should be NaN (insufficient data)
        assert math.isnan(matrix.loc["A", "B"])

    def test_get_matrix_alias(self):
        """get_matrix() and calculate_matrix() return same result."""
        tracker = self._build_tracker()
        m1 = tracker.calculate_matrix()
        m2 = tracker.get_matrix()
        pd.testing.assert_frame_equal(m1, m2)


class TestCorrelationTrackerNoLookAhead:
    """Verify no look-ahead bias."""

    def test_correlation_only_uses_data_fed_so_far(self):
        """Correlation changes as new data is fed, reflecting only past data."""
        tracker = CorrelationTracker(window=50, min_periods=10)
        np.random.seed(42)

        # Feed 15 observations of highly correlated data
        base = 100.0
        for _ in range(15):
            change = np.random.randn()
            base += change
            tracker.update("A", base)
            tracker.update("B", base * 2)

        corr_early = tracker.get_correlation("A", "B")
        assert corr_early is not None

        # Feed 15 more observations with anti-correlation
        for _ in range(15):
            change = np.random.randn()
            base += change
            tracker.update("A", base)
            tracker.update("B", 500.0 - base)

        corr_late = tracker.get_correlation("A", "B")
        assert corr_late is not None
        # Correlation should have moved lower as anti-correlated data was added
        assert corr_late < corr_early

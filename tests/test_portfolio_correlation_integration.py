"""Integration tests for Portfolio with CorrelationTracker."""

from datetime import datetime

import numpy as np

from src.portfolio.correlation import CorrelationTracker
from src.portfolio.portfolio import Portfolio


class TestPortfolioCorrelationIntegration:
    """Tests for correlation tracking through Portfolio.update_timeindex()."""

    def test_portfolio_without_correlation_tracker(self):
        """Default Portfolio (no tracker) works as before."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        portfolio.update_timeindex(datetime(2023, 1, 1))

        assert portfolio.correlation_tracker is None
        assert len(portfolio.equity_history) == 1

    def test_portfolio_with_correlation_tracker(self):
        """Portfolio with tracker updates correlations on update_timeindex()."""
        tracker = CorrelationTracker(window=60, min_periods=5)
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["AAPL", "MSFT"],
            correlation_tracker=tracker,
        )

        # Feed prices and update timeindex multiple times
        for i in range(10):
            portfolio.current_prices["AAPL"] = 150.0 + i
            portfolio.current_prices["MSFT"] = 300.0 + i * 2
            portfolio.update_timeindex(datetime(2023, 1, 1 + i))

        # Tracker should have received updates
        assert "AAPL" in tracker.symbols
        assert "MSFT" in tracker.symbols

    def test_correlation_matrix_after_multiple_bars(self):
        """After enough bars, correlation matrix is populated."""
        tracker = CorrelationTracker(window=60, min_periods=10)
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["AAPL", "MSFT"],
            correlation_tracker=tracker,
        )

        np.random.seed(42)
        base = 100.0
        for i in range(20):
            change = np.random.randn()
            base += change
            portfolio.current_prices["AAPL"] = base
            portfolio.current_prices["MSFT"] = base * 2 + np.random.randn() * 0.01
            portfolio.update_timeindex(datetime(2023, 1, 1, i))

        matrix = tracker.calculate_matrix()
        assert not matrix.empty
        assert "AAPL" in matrix.columns
        assert "MSFT" in matrix.columns

    def test_correlation_tracker_receives_all_priced_symbols(self):
        """Tracker receives updates for all symbols with current prices."""
        tracker = CorrelationTracker(window=60, min_periods=5)
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["AAPL", "MSFT", "GOOGL"],
            correlation_tracker=tracker,
        )

        portfolio.current_prices["AAPL"] = 150.0
        portfolio.current_prices["MSFT"] = 300.0
        # GOOGL has no price yet
        portfolio.update_timeindex(datetime(2023, 1, 1))

        assert "AAPL" in tracker.symbols
        assert "MSFT" in tracker.symbols
        assert "GOOGL" not in tracker.symbols  # No price was set

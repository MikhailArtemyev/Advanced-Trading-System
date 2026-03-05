"""Unit tests for Portfolio.get_portfolio_summary() and enhanced equity curve."""

from datetime import datetime

import pytest

from src.portfolio.portfolio import Portfolio


class TestGetPortfolioSummary:
    """Tests for the get_portfolio_summary() method."""

    def test_empty_portfolio(self):
        """No positions — all zeros except cash and equity."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL", "MSFT"])
        summary = portfolio.get_portfolio_summary()

        assert summary["long_exposure"] == 0.0
        assert summary["short_exposure"] == 0.0
        assert summary["net_exposure"] == 0.0
        assert summary["gross_exposure"] == 0.0
        assert summary["long_count"] == 0
        assert summary["short_count"] == 0
        assert summary["num_positions"] == 0
        assert summary["cash"] == 100000.0
        assert summary["equity"] == 100000.0

    def test_long_only(self):
        """Single long position: long_exposure > 0, short_exposure == 0."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        portfolio.positions["AAPL"].quantity = 100
        portfolio.positions["AAPL"].avg_cost = 150.0
        portfolio.current_prices["AAPL"] = 155.0

        summary = portfolio.get_portfolio_summary()

        assert summary["long_exposure"] == 100 * 155.0
        assert summary["short_exposure"] == 0.0
        assert summary["long_count"] == 1
        assert summary["short_count"] == 0
        assert summary["num_positions"] == 1

    def test_short_only(self):
        """Single short position: short_exposure > 0, long_exposure == 0."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        portfolio.positions["AAPL"].quantity = -50
        portfolio.positions["AAPL"].avg_cost = 150.0
        portfolio.current_prices["AAPL"] = 145.0

        summary = portfolio.get_portfolio_summary()

        assert summary["long_exposure"] == 0.0
        assert summary["short_exposure"] == 50 * 145.0
        assert summary["long_count"] == 0
        assert summary["short_count"] == 1
        assert summary["num_positions"] == 1

    def test_long_and_short(self):
        """Mixed portfolio: both exposures > 0, net_exposure = long - short."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL", "MSFT"])
        portfolio.positions["AAPL"].quantity = 100
        portfolio.positions["AAPL"].avg_cost = 150.0
        portfolio.current_prices["AAPL"] = 160.0

        portfolio.positions["MSFT"].quantity = -50
        portfolio.positions["MSFT"].avg_cost = 300.0
        portfolio.current_prices["MSFT"] = 290.0

        summary = portfolio.get_portfolio_summary()

        assert summary["long_exposure"] == 100 * 160.0
        assert summary["short_exposure"] == 50 * 290.0
        assert summary["net_exposure"] == 100 * 160.0 - 50 * 290.0
        assert summary["long_count"] == 1
        assert summary["short_count"] == 1
        assert summary["num_positions"] == 2

    def test_gross_exposure(self):
        """gross_exposure == long_exposure + short_exposure."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL", "MSFT"])
        portfolio.positions["AAPL"].quantity = 100
        portfolio.current_prices["AAPL"] = 150.0
        portfolio.positions["MSFT"].quantity = -50
        portfolio.current_prices["MSFT"] = 300.0

        summary = portfolio.get_portfolio_summary()

        expected_gross = 100 * 150.0 + 50 * 300.0
        assert summary["gross_exposure"] == expected_gross

    def test_includes_equity_and_cash(self):
        """Summary includes equity and cash values."""
        portfolio = Portfolio(initial_capital=50000.0, symbols=["AAPL"])
        summary = portfolio.get_portfolio_summary()

        assert summary["cash"] == 50000.0
        assert summary["equity"] == 50000.0


class TestEnhancedEquityCurve:
    """Tests for new equity_history fields."""

    def test_equity_history_contains_positions_value(self):
        """Each equity_history entry has 'positions_value' key."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        portfolio.update_timeindex(datetime(2023, 1, 1))

        assert "positions_value" in portfolio.equity_history[0]

    def test_equity_history_contains_num_positions(self):
        """Each equity_history entry has 'num_positions' key."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        portfolio.update_timeindex(datetime(2023, 1, 1))

        assert "num_positions" in portfolio.equity_history[0]

    def test_positions_value_matches_calculation(self):
        """positions_value == equity - cash."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        portfolio.positions["AAPL"].quantity = 100
        portfolio.positions["AAPL"].avg_cost = 150.0
        portfolio.current_prices["AAPL"] = 155.0
        portfolio.update_timeindex(datetime(2023, 1, 1))

        entry = portfolio.equity_history[0]
        expected_pos_value = 100 * 155.0
        assert entry["positions_value"] == pytest.approx(expected_pos_value)

    def test_num_positions_counts_nonzero(self):
        """num_positions counts only positions where quantity != 0."""
        portfolio = Portfolio(
            initial_capital=100000.0, symbols=["AAPL", "MSFT", "GOOGL"]
        )
        portfolio.positions["AAPL"].quantity = 100
        portfolio.positions["MSFT"].quantity = -50
        # GOOGL stays at 0
        portfolio.current_prices["AAPL"] = 150.0
        portfolio.current_prices["MSFT"] = 300.0
        portfolio.update_timeindex(datetime(2023, 1, 1))

        assert portfolio.equity_history[0]["num_positions"] == 2

    def test_equity_curve_dataframe_has_new_columns(self):
        """get_equity_curve() DataFrame includes positions_value and num_positions."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        portfolio.update_timeindex(datetime(2023, 1, 1))

        df = portfolio.get_equity_curve()
        assert "positions_value" in df.columns
        assert "num_positions" in df.columns

    def test_backwards_compatible_columns(self):
        """DataFrame still has timestamp, equity, cash columns."""
        portfolio = Portfolio(initial_capital=100000.0, symbols=["AAPL"])
        portfolio.update_timeindex(datetime(2023, 1, 1))

        df = portfolio.get_equity_curve()
        assert "timestamp" in df.columns
        assert "equity" in df.columns
        assert "cash" in df.columns

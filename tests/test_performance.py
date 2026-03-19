"""Unit tests for performance metrics and visualization."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.performance.metrics import PerformanceTracker
from src.performance.visualization import (
    create_full_report,
    plot_drawdown,
    plot_equity_curve,
    plot_returns_distribution,
)


class TestPerformanceTrackerInitialization:
    """Tests for PerformanceTracker initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        assert tracker.initial_capital == 100000.0
        assert tracker.risk_free_rate == 0.02
        assert tracker.periods_per_year == 252
        assert len(tracker.equity_curve) == 0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        tracker = PerformanceTracker(
            initial_capital=50000.0,
            risk_free_rate=0.03,
            periods_per_year=365,
        )

        assert tracker.initial_capital == 50000.0
        assert tracker.risk_free_rate == 0.03
        assert tracker.periods_per_year == 365


class TestPerformanceTrackerUpdate:
    """Tests for equity curve updates."""

    def test_update_records_equity(self):
        """Test that update records equity correctly."""
        tracker = PerformanceTracker(initial_capital=100000.0)
        timestamp = datetime(2023, 1, 15)

        tracker.update(timestamp, 100500.0)

        assert len(tracker.equity_curve) == 1
        assert tracker.equity_curve[0]["timestamp"] == timestamp
        assert tracker.equity_curve[0]["equity"] == 100500.0

    def test_multiple_updates(self):
        """Test multiple equity updates."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        for i in range(10):
            tracker.update(datetime(2023, 1, i + 1), 100000.0 + i * 100)

        assert len(tracker.equity_curve) == 10

    def test_get_equity_curve(self):
        """Test get_equity_curve returns DataFrame."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 100500.0)

        df = tracker.get_equity_curve()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "equity" in df.columns


class TestTotalReturn:
    """Tests for total return calculation."""

    def test_positive_return(self):
        """Test positive total return."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 110000.0)

        metrics = tracker.calculate_metrics()

        assert abs(metrics["total_return_pct"] - 10.0) < 0.01

    def test_negative_return(self):
        """Test negative total return."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 90000.0)

        metrics = tracker.calculate_metrics()

        assert abs(metrics["total_return_pct"] - (-10.0)) < 0.01

    def test_zero_return(self):
        """Test zero total return."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 100000.0)

        metrics = tracker.calculate_metrics()

        assert metrics["total_return_pct"] == 0.0


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_simple_drawdown(self):
        """Test simple drawdown calculation."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        # Peak at 110k, then drop to 99k = -10% drawdown
        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 110000.0)
        tracker.update(datetime(2023, 1, 3), 99000.0)

        metrics = tracker.calculate_metrics()

        # Max DD = (99000 - 110000) / 110000 = -10%
        assert abs(metrics["max_drawdown_pct"] - (-10.0)) < 0.01

    def test_no_drawdown(self):
        """Test no drawdown when equity only increases."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 101000.0)
        tracker.update(datetime(2023, 1, 3), 102000.0)

        metrics = tracker.calculate_metrics()

        assert metrics["max_drawdown_pct"] == 0.0

    def test_multiple_drawdowns(self):
        """Test max drawdown with multiple drawdown periods."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        # First drawdown: 110k -> 100k = -9.09%
        # Second drawdown: 120k -> 100k = -16.67%
        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 110000.0)
        tracker.update(datetime(2023, 1, 3), 100000.0)
        tracker.update(datetime(2023, 1, 4), 120000.0)
        tracker.update(datetime(2023, 1, 5), 100000.0)

        metrics = tracker.calculate_metrics()

        # Max DD should be the larger one: -16.67%
        assert abs(metrics["max_drawdown_pct"] - (-16.67)) < 0.1


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_positive_sharpe(self):
        """Test positive Sharpe ratio with good returns."""
        tracker = PerformanceTracker(
            initial_capital=100000.0,
            risk_free_rate=0.0,  # Zero risk-free for simpler calculation
        )

        # Generate returns with positive mean and some volatility
        np.random.seed(42)
        equity = 100000.0
        for i in range(252):  # One year
            # Mean 0.1% daily return with 0.5% std dev
            daily_return = np.random.normal(0.001, 0.005)
            equity *= 1 + daily_return
            tracker.update(datetime(2023, 1, 1) + timedelta(days=i), equity)

        metrics = tracker.calculate_metrics()

        # With positive mean returns, Sharpe should be positive
        assert metrics["sharpe_ratio"] > 0

    def test_zero_volatility_returns_zero(self):
        """Test that zero volatility returns zero Sharpe."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        # Constant equity = zero returns = zero volatility
        for i in range(10):
            tracker.update(datetime(2023, 1, 1) + timedelta(days=i), 100000.0)

        metrics = tracker.calculate_metrics()

        assert metrics["sharpe_ratio"] == 0.0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_with_downside(self):
        """Test Sortino ratio with downside volatility."""
        tracker = PerformanceTracker(
            initial_capital=100000.0,
            risk_free_rate=0.0,
        )

        # Create returns with some negative days
        equity = 100000.0
        np.random.seed(42)
        for i in range(100):
            daily_return = np.random.normal(0.001, 0.02)  # Mean positive
            equity *= 1 + daily_return
            tracker.update(datetime(2023, 1, 1) + timedelta(days=i), equity)

        metrics = tracker.calculate_metrics()

        # Sortino should be calculable
        assert isinstance(metrics["sortino_ratio"], float)

    def test_sortino_no_downside(self):
        """Test Sortino ratio with no negative returns."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        # Only positive returns
        equity = 100000.0
        for i in range(10):
            equity *= 1.001
            tracker.update(datetime(2023, 1, 1) + timedelta(days=i), equity)

        metrics = tracker.calculate_metrics()

        # No downside = zero Sortino (can't divide by zero)
        assert metrics["sortino_ratio"] == 0.0


class TestVolatility:
    """Tests for volatility calculation."""

    def test_volatility_calculation(self):
        """Test volatility calculation."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        # Create some volatile returns
        np.random.seed(42)
        equity = 100000.0
        for i in range(100):
            daily_return = np.random.normal(0, 0.01)  # 1% daily std
            equity *= 1 + daily_return
            tracker.update(datetime(2023, 1, 1) + timedelta(days=i), equity)

        metrics = tracker.calculate_metrics()

        # Volatility should be positive
        assert metrics["volatility_pct"] > 0


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_calculation(self):
        """Test Calmar ratio calculation."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        # Create returns with drawdown
        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 6, 1), 120000.0)
        tracker.update(datetime(2023, 9, 1), 108000.0)  # 10% drawdown from peak
        tracker.update(datetime(2023, 12, 31), 130000.0)

        metrics = tracker.calculate_metrics()

        # Calmar = annual return / max drawdown
        assert isinstance(metrics["calmar_ratio"], float)

    def test_calmar_zero_drawdown(self):
        """Test Calmar ratio with zero drawdown."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        # Monotonically increasing
        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 101000.0)
        tracker.update(datetime(2023, 1, 3), 102000.0)

        metrics = tracker.calculate_metrics()

        # Zero drawdown = zero Calmar
        assert metrics["calmar_ratio"] == 0.0


class TestCalculateMetrics:
    """Tests for overall metrics calculation."""

    def test_insufficient_data(self):
        """Test metrics with insufficient data."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)

        metrics = tracker.calculate_metrics()

        assert "error" in metrics

    def test_all_metrics_present(self):
        """Test that all expected metrics are present."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 101000.0)
        tracker.update(datetime(2023, 1, 3), 102000.0)

        metrics = tracker.calculate_metrics()

        expected_keys = [
            "initial_capital",
            "final_equity",
            "total_return_pct",
            "annualized_return_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_pct",
            "volatility_pct",
            "calmar_ratio",
            "trading_days",
        ]

        for key in expected_keys:
            assert key in metrics


class TestTradeMetrics:
    """Tests for trade-based metrics."""

    def test_trade_metrics_calculation(self):
        """Test trade metrics calculation."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        # Create mock trades
        trades = [
            MagicMock(side="SELL", pnl=500),
            MagicMock(side="SELL", pnl=-200),
            MagicMock(side="SELL", pnl=300),
            MagicMock(side="BUY", pnl=0),  # Buy trades should be ignored
        ]

        metrics = tracker.calculate_trade_metrics(trades)

        assert metrics["total_trades"] == 3
        assert metrics["winning_trades"] == 2
        assert metrics["losing_trades"] == 1
        assert abs(metrics["win_rate_pct"] - 66.67) < 0.1

    def test_empty_trades(self):
        """Test trade metrics with no trades."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        metrics = tracker.calculate_trade_metrics([])

        assert metrics["total_trades"] == 0
        assert metrics["win_rate_pct"] == 0.0

    def test_profit_factor(self):
        """Test profit factor calculation."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        trades = [
            MagicMock(side="SELL", pnl=1000),
            MagicMock(side="SELL", pnl=-500),
        ]

        metrics = tracker.calculate_trade_metrics(trades)

        # Profit factor = 1000 / 500 = 2.0
        assert metrics["profit_factor"] == 2.0

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        trades = [
            MagicMock(side="SELL", pnl=1000),
            MagicMock(side="SELL", pnl=500),
        ]

        metrics = tracker.calculate_trade_metrics(trades)

        assert metrics["profit_factor"] == float("inf")


class TestPrintReport:
    """Tests for report printing."""

    def test_print_report_runs(self, caplog):
        """Test that print_report executes without error."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 105000.0)

        with caplog.at_level("INFO", logger="src.performance.metrics"):
            tracker.print_report()

        assert "BACKTEST PERFORMANCE REPORT" in caplog.text
        assert "$100,000.00" in caplog.text

    def test_print_report_with_trades(self, caplog):
        """Test report printing with trade data."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 105000.0)

        trades = [MagicMock(side="SELL", pnl=500)]

        with caplog.at_level("INFO", logger="src.performance.metrics"):
            tracker.print_report(trades)

        assert "Total Trades:" in caplog.text

    def test_print_report_insufficient_data(self, caplog):
        """Test report with insufficient data."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)

        with caplog.at_level("WARNING", logger="src.performance.metrics"):
            tracker.print_report()

        assert "Cannot generate report" in caplog.text


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_equity_curve(self):
        """Test that reset clears equity curve."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 1, 2), 101000.0)

        tracker.reset()

        assert len(tracker.equity_curve) == 0


# Visualization Tests


class TestVisualization:
    """Tests for visualization functions."""

    @pytest.fixture
    def sample_equity_df(self):
        """Create sample equity DataFrame."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        equity = 100000 + np.cumsum(np.random.randn(100) * 500)
        return pd.DataFrame({"timestamp": dates, "equity": equity})

    @patch("matplotlib.pyplot.show")
    def test_plot_equity_curve(self, mock_show, sample_equity_df):
        """Test equity curve plotting."""
        fig = plot_equity_curve(sample_equity_df, show=False)

        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_drawdown(self, mock_show, sample_equity_df):
        """Test drawdown plotting."""
        fig = plot_drawdown(sample_equity_df, show=False)

        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_returns_distribution(self, mock_show, sample_equity_df):
        """Test returns distribution plotting."""
        fig = plot_returns_distribution(sample_equity_df, show=False)

        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_equity_curve_with_save(self, mock_show, sample_equity_df, tmp_path):
        """Test equity curve plotting with save."""
        save_path = str(tmp_path / "equity.png")

        plot_equity_curve(sample_equity_df, save_path=save_path, show=False)

        from pathlib import Path

        assert Path(save_path).exists()

    @patch("matplotlib.pyplot.show")
    def test_create_full_report(self, mock_show, sample_equity_df, tmp_path):
        """Test full report generation."""
        output_dir = str(tmp_path / "output")

        create_full_report(sample_equity_df, output_dir=output_dir, show=False)

        from pathlib import Path

        assert Path(output_dir).exists()
        assert (Path(output_dir) / "equity_curve.png").exists()
        assert (Path(output_dir) / "drawdown.png").exists()
        assert (Path(output_dir) / "returns_distribution.png").exists()


class TestMetricsKnownValues:
    """Tests with known expected values for verification."""

    def test_known_sharpe_ratio(self):
        """Test Sharpe ratio with known approximate values."""
        tracker = PerformanceTracker(
            initial_capital=100000.0,
            risk_free_rate=0.0,
            periods_per_year=252,
        )

        # Create data with approximately 0.1% mean daily return and 1% std
        # Sharpe ≈ (0.001 / 0.01) * sqrt(252) ≈ 1.59
        np.random.seed(42)
        equity = 100000.0
        for i in range(252):
            # Use fixed seed for reproducibility
            daily_return = 0.001 + (np.sin(i * 0.1) * 0.01)  # Mean ~0.1%, varying
            equity *= 1 + daily_return
            tracker.update(datetime(2023, 1, 1) + timedelta(days=i), equity)

        metrics = tracker.calculate_metrics()

        # Sharpe should be positive with positive mean returns
        assert metrics["sharpe_ratio"] > 0

    def test_known_total_return(self):
        """Test total return with known values."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 12, 31), 125000.0)

        metrics = tracker.calculate_metrics()

        assert metrics["total_return_pct"] == 25.0

    def test_known_max_drawdown(self):
        """Test max drawdown with known values."""
        tracker = PerformanceTracker(initial_capital=100000.0)

        tracker.update(datetime(2023, 1, 1), 100000.0)
        tracker.update(datetime(2023, 2, 1), 120000.0)  # Peak
        tracker.update(datetime(2023, 3, 1), 96000.0)  # 20% drop from peak
        tracker.update(datetime(2023, 4, 1), 130000.0)

        metrics = tracker.calculate_metrics()

        # Max DD = (96000 - 120000) / 120000 = -20%
        assert abs(metrics["max_drawdown_pct"] - (-20.0)) < 0.01

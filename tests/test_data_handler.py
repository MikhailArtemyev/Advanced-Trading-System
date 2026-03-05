"""Unit tests for data handler classes."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.data_handler import HistoricalCSVDataHandler


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create sample CSV data for testing."""
    # Create sample AAPL data
    dates = pd.date_range("2022-01-01", "2022-12-31", freq="B")  # Business days
    np.random.seed(42)

    # Generate realistic price data
    initial_price = 150.0
    returns = np.random.randn(len(dates)) * 0.02  # 2% daily volatility
    prices = initial_price * np.exp(np.cumsum(returns))

    aapl_df = pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 - np.random.uniform(0, 0.01, len(dates))),
            "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            "low": prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, len(dates)),
        }
    )
    aapl_df.to_csv(tmp_path / "AAPL.csv", index=False)

    # Create sample MSFT data with same dates
    initial_price_msft = 250.0
    returns_msft = np.random.randn(len(dates)) * 0.018
    prices_msft = initial_price_msft * np.exp(np.cumsum(returns_msft))

    msft_df = pd.DataFrame(
        {
            "date": dates,
            "open": prices_msft * (1 - np.random.uniform(0, 0.01, len(dates))),
            "high": prices_msft * (1 + np.random.uniform(0, 0.02, len(dates))),
            "low": prices_msft * (1 - np.random.uniform(0, 0.02, len(dates))),
            "close": prices_msft,
            "volume": np.random.randint(2000000, 8000000, len(dates)),
        }
    )
    msft_df.to_csv(tmp_path / "MSFT.csv", index=False)

    return str(tmp_path)


@pytest.fixture
def data_handler(sample_data_dir):
    """Create a data handler with sample data."""
    return HistoricalCSVDataHandler(
        data_path=sample_data_dir,
        symbols=["AAPL", "MSFT"],
        start_date="2022-01-01",
        end_date="2022-12-31",
    )


class TestHistoricalCSVDataHandler:
    """Tests for HistoricalCSVDataHandler class."""

    def test_load_data(self, data_handler):
        """Test that data loads correctly."""
        assert "AAPL" in data_handler.data
        assert "MSFT" in data_handler.data
        assert len(data_handler.data["AAPL"]) > 0
        assert len(data_handler.data["MSFT"]) > 0

    def test_required_columns(self, data_handler):
        """Test that all required columns are present."""
        required = ["open", "high", "low", "close", "volume"]
        for symbol in ["AAPL", "MSFT"]:
            for col in required:
                assert col in data_handler.data[symbol].columns

    def test_initial_state(self, data_handler):
        """Test initial state of data handler."""
        assert data_handler.current_index == 0
        assert data_handler.max_index > 0
        assert data_handler.continue_backtest is True

    def test_get_latest_bars_single(self, data_handler):
        """Test getting a single bar."""
        bars = data_handler.get_latest_bars("AAPL", 1)
        assert len(bars) == 1
        assert "close" in bars.columns

    def test_get_latest_bars_multiple(self, data_handler):
        """Test getting multiple bars."""
        # Advance a few bars first
        for _ in range(10):
            data_handler.update_bars()

        bars = data_handler.get_latest_bars("AAPL", 5)
        assert len(bars) == 5

    def test_get_latest_bars_more_than_available(self, data_handler):
        """Test requesting more bars than available."""
        # At start, only 1 bar is available
        bars = data_handler.get_latest_bars("AAPL", 100)
        assert len(bars) == 1

    def test_update_bars(self, data_handler):
        """Test advancing through bars."""
        initial_index = data_handler.current_index
        result = data_handler.update_bars()

        assert result is True
        assert data_handler.current_index == initial_index + 1

    def test_update_bars_until_exhausted(self, data_handler):
        """Test that update_bars returns False when data exhausted."""
        while data_handler.continue_backtest:
            data_handler.update_bars()

        # One more update should return False
        result = data_handler.update_bars()
        assert result is False

    def test_get_current_timestamp(self, data_handler):
        """Test getting current timestamp."""
        timestamp = data_handler.get_current_timestamp()
        assert isinstance(timestamp, datetime)

    def test_timestamp_advances(self, data_handler):
        """Test that timestamp advances with bars."""
        ts1 = data_handler.get_current_timestamp()
        data_handler.update_bars()
        ts2 = data_handler.get_current_timestamp()

        assert ts2 > ts1

    def test_continue_backtest(self, data_handler):
        """Test continue_backtest property."""
        assert data_handler.continue_backtest is True

        # Exhaust data
        while data_handler.continue_backtest:
            data_handler.update_bars()

        assert data_handler.continue_backtest is False

    def test_get_all_symbols(self, data_handler):
        """Test getting list of symbols."""
        symbols = data_handler.get_all_symbols()
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert len(symbols) == 2

    def test_symbol_not_found(self, data_handler):
        """Test that KeyError is raised for unknown symbol."""
        with pytest.raises(KeyError):
            data_handler.get_latest_bars("INVALID", 1)

    def test_get_latest_bar_value(self, data_handler):
        """Test getting a single value from latest bar."""
        close = data_handler.get_latest_bar_value("AAPL", "close")
        assert isinstance(close, (int, float))
        assert close > 0

    def test_reset(self, data_handler):
        """Test resetting the data handler."""
        # Advance some bars
        for _ in range(10):
            data_handler.update_bars()

        assert data_handler.current_index == 10

        # Reset
        data_handler.reset()
        assert data_handler.current_index == 0


class TestNoLookAheadBias:
    """Tests specifically for look-ahead bias prevention."""

    def test_no_future_data_at_start(self, data_handler):
        """Test that only first bar is available at start."""
        bars = data_handler.get_latest_bars("AAPL", 1000)
        # Should only get 1 bar (the first one)
        assert len(bars) == 1

    def test_no_future_data_after_update(self, data_handler):
        """Test that bars increase only after update."""
        # Get initial count
        bars1 = data_handler.get_latest_bars("AAPL", 1000)
        count1 = len(bars1)

        # Update
        data_handler.update_bars()

        # Get new count
        bars2 = data_handler.get_latest_bars("AAPL", 1000)
        count2 = len(bars2)

        assert count2 == count1 + 1

    def test_bars_are_copies(self, data_handler):
        """Test that returned bars are copies, not references."""
        bars = data_handler.get_latest_bars("AAPL", 1)

        # Modify the returned data
        original_close = bars["close"].iloc[0]
        bars["close"].iloc[0] = 999999

        # Get bars again
        bars2 = data_handler.get_latest_bars("AAPL", 1)

        # Original data should be unchanged
        assert bars2["close"].iloc[0] == original_close

    def test_index_matches_available_data(self, data_handler):
        """Test that current_index correctly limits available data."""
        for _ in range(20):
            bars = data_handler.get_latest_bars("AAPL", 1000)
            # Available bars should equal current_index + 1
            assert len(bars) == data_handler.current_index + 1
            data_handler.update_bars()


class TestDateFiltering:
    """Tests for date range filtering."""

    def test_date_range_filtering(self, sample_data_dir):
        """Test that data is filtered to specified date range."""
        handler = HistoricalCSVDataHandler(
            data_path=sample_data_dir,
            symbols=["AAPL"],
            start_date="2022-06-01",
            end_date="2022-06-30",
        )

        # Check that data is within range
        df = handler.data["AAPL"]
        assert df.index.min() >= pd.Timestamp("2022-06-01")
        assert df.index.max() <= pd.Timestamp("2022-06-30")


class TestMissingColumns:
    """Tests for handling missing columns."""

    def test_missing_columns_raises_error(self, tmp_path):
        """Test that missing required columns raise ValueError."""
        # Create CSV with missing 'volume' column
        df = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=10),
                "open": [100] * 10,
                "high": [101] * 10,
                "low": [99] * 10,
                "close": [100.5] * 10,
                # Missing 'volume'
            }
        )
        df.to_csv(tmp_path / "BAD.csv", index=False)

        with pytest.raises(ValueError, match="Missing columns"):
            HistoricalCSVDataHandler(
                data_path=str(tmp_path),
                symbols=["BAD"],
                start_date="2022-01-01",
                end_date="2022-12-31",
            )

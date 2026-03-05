"""Unit tests for YFinanceDataHandler with mocked yfinance."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.yfinance_handler import YFinanceDataHandler


@pytest.fixture
def mock_yf_data() -> pd.DataFrame:
    """Create realistic mock yfinance data."""
    dates = pd.date_range("2022-01-03", "2022-12-30", freq="B")
    np.random.seed(42)
    prices = 150.0 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02))

    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(1000000, 5000000, len(dates)),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def _make_cache_csv(tmp_path: object, symbol: str, start: str, end: str) -> None:
    """Create a pre-populated cache CSV file."""
    dates = pd.date_range(start, end, freq="B")
    np.random.seed(42)
    prices = 150.0 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02))
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, len(dates)),
        }
    )
    csv_path = tmp_path / f"{symbol}.csv"  # type: ignore[operator]
    df.to_csv(csv_path, index=False)


class TestYFinanceCaching:
    """Tests for download and caching behavior."""

    @patch("src.data.yfinance_handler.yf")
    def test_downloads_when_no_cache(
        self, mock_yf: MagicMock, mock_yf_data: pd.DataFrame, tmp_path: object
    ) -> None:
        """Downloads data when no cached file exists."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yf_data
        mock_yf.Ticker.return_value = mock_ticker

        YFinanceDataHandler(
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2022-12-31",
            cache_dir=str(tmp_path),
        )

        mock_yf.Ticker.assert_called_with("AAPL")
        mock_ticker.history.assert_called_once()
        assert (tmp_path / "AAPL.csv").exists()  # type: ignore[operator]

    @patch("src.data.yfinance_handler.yf")
    def test_uses_cache_when_valid(self, mock_yf: MagicMock, tmp_path: object) -> None:
        """Does not download when cache covers the date range."""
        _make_cache_csv(tmp_path, "AAPL", "2022-01-03", "2022-12-30")

        YFinanceDataHandler(
            symbols=["AAPL"],
            start_date="2022-06-01",
            end_date="2022-09-30",
            cache_dir=str(tmp_path),
        )

        mock_yf.Ticker.assert_not_called()

    @patch("src.data.yfinance_handler.yf")
    def test_redownloads_when_range_not_covered(
        self, mock_yf: MagicMock, mock_yf_data: pd.DataFrame, tmp_path: object
    ) -> None:
        """Re-downloads when cache does not cover the requested date range."""
        _make_cache_csv(tmp_path, "AAPL", "2022-01-03", "2022-06-30")

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yf_data
        mock_yf.Ticker.return_value = mock_ticker

        YFinanceDataHandler(
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2022-12-31",
            cache_dir=str(tmp_path),
        )

        mock_yf.Ticker.assert_called_with("AAPL")

    @patch("src.data.yfinance_handler.yf")
    def test_raises_on_empty_download(
        self, mock_yf: MagicMock, tmp_path: object
    ) -> None:
        """Raises ValueError when yfinance returns no data."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        with pytest.raises(ValueError, match="No data returned"):
            YFinanceDataHandler(
                symbols=["INVALID"],
                start_date="2022-01-01",
                end_date="2022-12-31",
                cache_dir=str(tmp_path),
            )


class TestYFinanceDelegation:
    """Tests that bar iteration delegates correctly to CSV handler."""

    @patch("src.data.yfinance_handler.yf")
    def test_get_latest_bars(
        self, mock_yf: MagicMock, mock_yf_data: pd.DataFrame, tmp_path: object
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yf_data
        mock_yf.Ticker.return_value = mock_ticker

        handler = YFinanceDataHandler(
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2022-12-31",
            cache_dir=str(tmp_path),
        )

        bars = handler.get_latest_bars("AAPL", 1)
        assert len(bars) == 1
        assert "close" in bars.columns

    @patch("src.data.yfinance_handler.yf")
    def test_update_bars_and_continue(
        self, mock_yf: MagicMock, mock_yf_data: pd.DataFrame, tmp_path: object
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yf_data
        mock_yf.Ticker.return_value = mock_ticker

        handler = YFinanceDataHandler(
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2022-12-31",
            cache_dir=str(tmp_path),
        )

        assert handler.continue_backtest is True
        result = handler.update_bars()
        assert result is True

    @patch("src.data.yfinance_handler.yf")
    def test_get_current_timestamp(
        self, mock_yf: MagicMock, mock_yf_data: pd.DataFrame, tmp_path: object
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yf_data
        mock_yf.Ticker.return_value = mock_ticker

        handler = YFinanceDataHandler(
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2022-12-31",
            cache_dir=str(tmp_path),
        )

        ts = handler.get_current_timestamp()
        assert isinstance(ts, datetime)

    @patch("src.data.yfinance_handler.yf")
    def test_reset(
        self, mock_yf: MagicMock, mock_yf_data: pd.DataFrame, tmp_path: object
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yf_data
        mock_yf.Ticker.return_value = mock_ticker

        handler = YFinanceDataHandler(
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2022-12-31",
            cache_dir=str(tmp_path),
        )

        for _ in range(5):
            handler.update_bars()

        handler.reset()
        bars = handler.get_latest_bars("AAPL", 1000)
        assert len(bars) == 1  # Back to start


class TestYFinanceCSVFormat:
    """Tests that cached CSV matches expected format."""

    @patch("src.data.yfinance_handler.yf")
    def test_csv_has_correct_columns(
        self, mock_yf: MagicMock, mock_yf_data: pd.DataFrame, tmp_path: object
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yf_data
        mock_yf.Ticker.return_value = mock_ticker

        YFinanceDataHandler(
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2022-12-31",
            cache_dir=str(tmp_path),
        )

        df = pd.read_csv(tmp_path / "AAPL.csv")  # type: ignore[operator]
        required = {"date", "open", "high", "low", "close", "volume"}
        assert required.issubset(set(df.columns))

    @patch("src.data.yfinance_handler.yf")
    def test_csv_columns_are_lowercase(
        self, mock_yf: MagicMock, mock_yf_data: pd.DataFrame, tmp_path: object
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yf_data
        mock_yf.Ticker.return_value = mock_ticker

        YFinanceDataHandler(
            symbols=["AAPL"],
            start_date="2022-01-01",
            end_date="2022-12-31",
            cache_dir=str(tmp_path),
        )

        df = pd.read_csv(tmp_path / "AAPL.csv")  # type: ignore[operator]
        for col in df.columns:
            assert col == col.lower()

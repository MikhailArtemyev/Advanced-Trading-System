"""Unit tests for technical indicator feature generators."""

import numpy as np
import pandas as pd
import pytest

from src.features.base_feature import FeatureGenerator, FeatureResult
from src.features.technical import (
    ATRFeature,
    BollingerBandFeature,
    MACDFeature,
    RSIFeature,
    SMAFeature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int = 100,
    start_price: float = 100.0,
    step: float = 0.0,
    vol: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame.

    Args:
        n:           Number of bars.
        start_price: Starting close price.
        step:        Deterministic daily drift added to price.
        vol:         Random noise std (set to 0 for deterministic series).
        seed:        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    noise = rng.standard_normal(n) * vol if vol > 0 else np.zeros(n)
    prices = start_price + step * np.arange(n) + np.cumsum(noise)
    prices = np.clip(prices, 0.01, None)  # avoid non-positive prices
    df = pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.ones(n) * 1_000_000,
        },
        index=dates,
    )
    return df


def _make_flat_ohlcv(n: int = 50, price: float = 100.0) -> pd.DataFrame:
    """OHLCV with constant price (zero volatility)."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1_000_000,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# FeatureResult
# ---------------------------------------------------------------------------


class TestFeatureResult:
    def test_creation_and_field_access(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = FeatureResult(features=df, feature_names=["x"], method="test")

        assert result.method == "test"
        assert result.feature_names == ["x"]
        assert list(result.features.columns) == ["x"]

    def test_frozen_cannot_mutate(self) -> None:
        df = pd.DataFrame({"x": [1.0]})
        result = FeatureResult(features=df, feature_names=["x"], method="test")

        with pytest.raises((AttributeError, TypeError)):
            result.method = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FeatureGenerator ABC
# ---------------------------------------------------------------------------


class TestFeatureGeneratorABC:
    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            FeatureGenerator()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# SMAFeature
# ---------------------------------------------------------------------------


class TestSMAFeature:
    def test_default_windows(self) -> None:
        feat = SMAFeature()
        assert feat.windows == [5, 10, 20, 50]

    def test_lookback_equals_max_window(self) -> None:
        assert SMAFeature([3, 7, 15]).lookback == 15
        assert SMAFeature([50]).lookback == 50

    def test_feature_names_correct(self) -> None:
        result = SMAFeature([5, 10]).compute(_make_ohlcv(50))
        assert result.feature_names == [
            "sma_5",
            "sma_5_ratio",
            "sma_10",
            "sma_10_ratio",
        ]
        assert result.method == "sma"

    def test_sma_formula_known_series(self) -> None:
        """With window=3 and close = [1,2,3,4,5], sma_3 = [NaN,NaN,2,3,4]."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "open": 1,
                "high": 1,
                "low": 1,
                "close": [1.0, 2.0, 3.0, 4.0, 5.0],
                "volume": 1,
            },
            index=dates,
        )
        result = SMAFeature([3]).compute(df)
        sma = result.features["sma_3"]

        assert np.isnan(sma.iloc[0])
        assert np.isnan(sma.iloc[1])
        assert sma.iloc[2] == pytest.approx(2.0)
        assert sma.iloc[3] == pytest.approx(3.0)
        assert sma.iloc[4] == pytest.approx(4.0)

    def test_ratio_equals_close_over_sma(self) -> None:
        ohlcv = _make_ohlcv(60)
        result = SMAFeature([10]).compute(ohlcv)
        expected_ratio = ohlcv["close"] / result.features["sma_10"]
        pd.testing.assert_series_equal(
            result.features["sma_10_ratio"], expected_ratio, check_names=False
        )

    def test_insufficient_data_all_nan(self) -> None:
        """With only 3 bars and window=10, all SMA values are NaN."""
        ohlcv = _make_ohlcv(3)
        result = SMAFeature([10]).compute(ohlcv)
        assert result.features["sma_10"].isna().all()

    def test_single_window(self) -> None:
        result = SMAFeature([5]).compute(_make_ohlcv(20))
        assert "sma_5" in result.features.columns
        assert "sma_5_ratio" in result.features.columns
        assert len(result.feature_names) == 2

    def test_result_index_matches_input(self) -> None:
        ohlcv = _make_ohlcv(30)
        result = SMAFeature([5]).compute(ohlcv)
        assert result.features.index.equals(ohlcv.index)


# ---------------------------------------------------------------------------
# RSIFeature
# ---------------------------------------------------------------------------


class TestRSIFeature:
    def test_default_period(self) -> None:
        assert RSIFeature().period == 14

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period must be >= 1"):
            RSIFeature(period=0)

    def test_lookback(self) -> None:
        assert RSIFeature(14).lookback == 15
        assert RSIFeature(7).lookback == 8

    def test_feature_name(self) -> None:
        result = RSIFeature(14).compute(_make_ohlcv(50))
        assert result.feature_names == ["rsi_14"]
        assert result.method == "rsi"

    def test_rsi_range_0_to_100(self) -> None:
        ohlcv = _make_ohlcv(200)
        result = RSIFeature(14).compute(ohlcv)
        valid = result.features["rsi_14"].dropna()

        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_all_gains_rsi_equals_100(self) -> None:
        """When every bar is an up day over the RSI window, RSI = 100."""
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        prices = np.arange(1.0, 31.0)  # monotonically increasing
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": 1,
            },
            index=dates,
        )
        result = RSIFeature(14).compute(df)
        valid = result.features["rsi_14"].dropna()
        assert (valid == 100.0).all()

    def test_all_losses_rsi_equals_0(self) -> None:
        """When every bar is a down day, RSI = 0."""
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        prices = np.arange(30.0, 0.0, -1.0)  # monotonically decreasing
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": 1,
            },
            index=dates,
        )
        result = RSIFeature(14).compute(df)
        valid = result.features["rsi_14"].dropna()
        assert (valid == 0.0).all()

    def test_nan_for_insufficient_bars(self) -> None:
        # rolling(14) on the gain/loss series (which starts with diff() → NaN at row 0)
        # gives first valid at row 14 (0-indexed), so rows 0-13 are NaN.
        ohlcv = _make_ohlcv(20, vol=0.01)
        result = RSIFeature(14).compute(ohlcv)
        rsi = result.features["rsi_14"]
        assert rsi.iloc[:14].isna().all()
        assert not np.isnan(rsi.iloc[14])

    def test_result_index_matches_input(self) -> None:
        ohlcv = _make_ohlcv(40)
        result = RSIFeature(10).compute(ohlcv)
        assert result.features.index.equals(ohlcv.index)


# ---------------------------------------------------------------------------
# MACDFeature
# ---------------------------------------------------------------------------


class TestMACDFeature:
    def test_default_params(self) -> None:
        feat = MACDFeature()
        assert feat.fast_period == 12
        assert feat.slow_period == 26
        assert feat.signal_period == 9

    def test_lookback(self) -> None:
        assert MACDFeature(12, 26, 9).lookback == 35
        assert MACDFeature(5, 10, 3).lookback == 13

    def test_feature_names(self) -> None:
        result = MACDFeature().compute(_make_ohlcv(100))
        assert result.feature_names == ["macd", "macd_signal", "macd_histogram"]
        assert result.method == "macd"

    def test_histogram_equals_macd_minus_signal(self) -> None:
        ohlcv = _make_ohlcv(150)
        result = MACDFeature().compute(ohlcv)
        f = result.features

        np.testing.assert_allclose(
            f["macd_histogram"].values,
            (f["macd"] - f["macd_signal"]).values,
            rtol=1e-10,
        )

    def test_macd_line_is_fast_minus_slow_ema(self) -> None:
        ohlcv = _make_ohlcv(100)
        close = ohlcv["close"]
        fast_ema = close.ewm(span=12, adjust=False).mean()
        slow_ema = close.ewm(span=26, adjust=False).mean()
        expected_macd = fast_ema - slow_ema

        result = MACDFeature().compute(ohlcv)
        pd.testing.assert_series_equal(
            result.features["macd"], expected_macd, check_names=False
        )

    def test_all_columns_present(self) -> None:
        result = MACDFeature().compute(_make_ohlcv(100))
        for col in ("macd", "macd_signal", "macd_histogram"):
            assert col in result.features.columns

    def test_result_index_matches_input(self) -> None:
        ohlcv = _make_ohlcv(60)
        result = MACDFeature().compute(ohlcv)
        assert result.features.index.equals(ohlcv.index)


# ---------------------------------------------------------------------------
# BollingerBandFeature
# ---------------------------------------------------------------------------


class TestBollingerBandFeature:
    def test_default_params(self) -> None:
        feat = BollingerBandFeature()
        assert feat.window == 20
        assert feat.num_std == 2.0

    def test_lookback(self) -> None:
        assert BollingerBandFeature(window=20).lookback == 20
        assert BollingerBandFeature(window=5).lookback == 5

    def test_feature_names(self) -> None:
        result = BollingerBandFeature().compute(_make_ohlcv(50))
        assert result.feature_names == ["bb_width", "bb_pct_b"]
        assert result.method == "bollinger"

    def test_bb_width_positive_for_volatile_series(self) -> None:
        ohlcv = _make_ohlcv(100, vol=0.02)
        result = BollingerBandFeature(window=10).compute(ohlcv)
        valid = result.features["bb_width"].dropna()
        assert (valid > 0).all()

    def test_bb_width_zero_for_constant_prices(self) -> None:
        """Constant price → std=0 → band_range=0 → bb_width=0, bb_pct_b=NaN."""
        ohlcv = _make_flat_ohlcv(50)
        result = BollingerBandFeature(window=10).compute(ohlcv)
        valid_width = result.features["bb_width"].dropna()
        valid_pct_b = result.features["bb_pct_b"].dropna()

        # bb_width = band_range / sma = 0 / 100 = 0 (not NaN)
        assert (valid_width == 0.0).all()
        # bb_pct_b = (close - lower) / band_range = 0 / 0 → NaN via replace(0, NaN)
        assert len(valid_pct_b) == 0

    def test_pct_b_has_valid_values(self) -> None:
        """With volatile data, %B produces non-empty valid output."""
        ohlcv = _make_ohlcv(100, vol=0.01, seed=1)
        result = BollingerBandFeature(window=20).compute(ohlcv)
        valid = result.features["bb_pct_b"].dropna()
        assert len(valid) > 0

    def test_pct_b_range_roughly_minus_one_to_two(self) -> None:
        """For normal price series, %B can go below 0 or above 1 during extremes."""
        ohlcv = _make_ohlcv(200, vol=0.02)
        result = BollingerBandFeature(window=20).compute(ohlcv)
        valid = result.features["bb_pct_b"].dropna()
        # %B has no hard bounds but should be in a sensible range for this data
        assert valid.min() > -3.0
        assert valid.max() < 4.0

    def test_nan_for_insufficient_data(self) -> None:
        ohlcv = _make_ohlcv(5)
        result = BollingerBandFeature(window=20).compute(ohlcv)
        assert result.features["bb_width"].isna().all()

    def test_result_index_matches_input(self) -> None:
        ohlcv = _make_ohlcv(60)
        result = BollingerBandFeature().compute(ohlcv)
        assert result.features.index.equals(ohlcv.index)


# ---------------------------------------------------------------------------
# ATRFeature
# ---------------------------------------------------------------------------


class TestATRFeature:
    def test_default_period(self) -> None:
        assert ATRFeature().period == 14

    def test_lookback(self) -> None:
        assert ATRFeature(14).lookback == 15
        assert ATRFeature(7).lookback == 8

    def test_feature_names(self) -> None:
        result = ATRFeature(14).compute(_make_ohlcv(50))
        assert result.feature_names == ["atr_14", "atr_14_pct"]
        assert result.method == "atr"

    def test_custom_period_feature_names(self) -> None:
        result = ATRFeature(7).compute(_make_ohlcv(30))
        assert "atr_7" in result.features.columns
        assert "atr_7_pct" in result.features.columns

    def test_atr_nonnegative(self) -> None:
        ohlcv = _make_ohlcv(100, vol=0.02)
        result = ATRFeature(14).compute(ohlcv)
        valid = result.features["atr_14"].dropna()
        assert (valid >= 0).all()

    def test_atr_zero_for_flat_ohlcv(self) -> None:
        """Flat prices → true range = 0 → ATR = 0."""
        ohlcv = _make_flat_ohlcv(50)
        result = ATRFeature(14).compute(ohlcv)
        valid = result.features["atr_14"].dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-12)

    def test_atr_pct_equals_atr_over_close(self) -> None:
        ohlcv = _make_ohlcv(100)
        result = ATRFeature(14).compute(ohlcv)
        expected_pct = result.features["atr_14"] / ohlcv["close"]
        pd.testing.assert_series_equal(
            result.features["atr_14_pct"], expected_pct, check_names=False
        )

    def test_nan_for_insufficient_data(self) -> None:
        ohlcv = _make_ohlcv(5)
        result = ATRFeature(14).compute(ohlcv)
        assert result.features["atr_14"].isna().all()

    def test_atr_captures_gap(self) -> None:
        """ATR after a price gap is larger than the simple H-L range."""
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        prices = np.ones(30) * 100.0
        prices[15:] = 110.0  # gap of 10 at bar 15
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices,
                "volume": 1,
            },
            index=dates,
        )
        result = ATRFeature(5).compute(df)
        # H-L range is always 2; ATR right after the gap should exceed that
        # Bar 15 is the gap day; rolling(5) from bars 11-15 includes the gap
        gap_atr = result.features["atr_5"].iloc[16]
        assert gap_atr > 2.0

    def test_result_index_matches_input(self) -> None:
        ohlcv = _make_ohlcv(60)
        result = ATRFeature().compute(ohlcv)
        assert result.features.index.equals(ohlcv.index)

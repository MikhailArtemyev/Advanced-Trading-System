"""Unit tests for statistical feature generators."""

import numpy as np
import pandas as pd
import pytest

from src.features.statistical import (
    HigherMomentFeature,
    HurstExponentFeature,
    ReturnFeature,
    VolatilityFeature,
    ZScoreFeature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n: int = 100,
    start_price: float = 100.0,
    daily_return: float = 0.0,
    vol: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    noise = rng.standard_normal(n) * vol if vol > 0 else np.zeros(n)
    log_prices = np.log(start_price) + daily_return * np.arange(n) + np.cumsum(noise)
    prices = np.exp(log_prices)
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.ones(n) * 1_000_000,
        },
        index=dates,
    )


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


def _make_trending_ohlcv(n: int = 500, rho: float = 0.7, seed: int = 99) -> pd.DataFrame:
    """Positively autocorrelated returns — designed to produce Hurst > 0.5.

    Uses an AR(1) process with positive autocorrelation so returns persist
    in direction, which is what the R/S Hurst estimator detects as trending.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    innovations = rng.standard_normal(n) * 0.02
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = rho * returns[i - 1] + innovations[i]
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1},
        index=dates,
    )


def _make_mean_reverting_ohlcv(n: int = 250) -> pd.DataFrame:
    """Strongly alternating prices — designed to produce Hurst < 0.5."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # Prices oscillate: 100, 101, 100, 101, ...
    prices = np.array([100.0 + (i % 2) for i in range(n)])
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1},
        index=dates,
    )


# ---------------------------------------------------------------------------
# ReturnFeature
# ---------------------------------------------------------------------------

class TestReturnFeature:
    def test_default_horizons(self) -> None:
        assert ReturnFeature().horizons == [1, 5, 10, 20]

    def test_lookback_equals_max_horizon_plus_one(self) -> None:
        assert ReturnFeature([1, 5, 10, 20]).lookback == 21
        assert ReturnFeature([3]).lookback == 4
        assert ReturnFeature([1]).lookback == 2

    def test_feature_names(self) -> None:
        result = ReturnFeature([1, 5]).compute(_make_ohlcv(30))
        assert result.feature_names == ["return_1d", "return_5d"]
        assert result.method == "returns"

    def test_log_return_formula_1d(self) -> None:
        """1-day log return: ln(close[t] / close[t-1])."""
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "open": [100, 110, 121],
                "high": [100, 110, 121],
                "low": [100, 110, 121],
                "close": [100.0, 110.0, 121.0],
                "volume": 1,
            },
            index=dates,
        )
        result = ReturnFeature([1]).compute(df)
        r = result.features["return_1d"]

        assert np.isnan(r.iloc[0])
        assert r.iloc[1] == pytest.approx(np.log(110.0 / 100.0))
        assert r.iloc[2] == pytest.approx(np.log(121.0 / 110.0))

    def test_log_return_formula_multi_day(self) -> None:
        """5-day log return: ln(close[t] / close[t-5])."""
        n = 10
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = np.arange(1.0, n + 1.0)
        df = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1},
            index=dates,
        )
        result = ReturnFeature([5]).compute(df)
        r = result.features["return_5d"]

        assert r.iloc[5] == pytest.approx(np.log(6.0 / 1.0))
        assert r.iloc[9] == pytest.approx(np.log(10.0 / 5.0))

    def test_nan_for_insufficient_bars(self) -> None:
        ohlcv = _make_ohlcv(30)
        result = ReturnFeature([1, 5, 20]).compute(ohlcv)
        # return_20d: first 20 bars are NaN
        r20 = result.features["return_20d"]
        assert r20.iloc[:20].isna().all()
        assert not np.isnan(r20.iloc[20])

    def test_multiple_horizons_produce_correct_columns(self) -> None:
        result = ReturnFeature([1, 3, 7]).compute(_make_ohlcv(50))
        for col in ("return_1d", "return_3d", "return_7d"):
            assert col in result.features.columns

    def test_result_index_matches_input(self) -> None:
        ohlcv = _make_ohlcv(50)
        result = ReturnFeature([1]).compute(ohlcv)
        assert result.features.index.equals(ohlcv.index)


# ---------------------------------------------------------------------------
# ZScoreFeature
# ---------------------------------------------------------------------------

class TestZScoreFeature:
    def test_default_window(self) -> None:
        assert ZScoreFeature().window == 20

    def test_lookback(self) -> None:
        assert ZScoreFeature(20).lookback == 20
        assert ZScoreFeature(5).lookback == 5

    def test_feature_name(self) -> None:
        result = ZScoreFeature(20).compute(_make_ohlcv(50))
        assert result.feature_names == ["zscore_20"]
        assert result.method == "zscore"

    def test_zscore_formula(self) -> None:
        """Verify z-score = (close - rolling_mean) / rolling_std."""
        ohlcv = _make_ohlcv(60, seed=1)
        window = 10
        result = ZScoreFeature(window).compute(ohlcv)

        close = ohlcv["close"]
        mean = close.rolling(window).mean()
        std = close.rolling(window).std()
        expected = (close - mean) / std
        expected.name = f"zscore_{window}"

        pd.testing.assert_series_equal(
            result.features[f"zscore_{window}"].dropna(),
            expected.dropna(),
            check_names=False,
        )

    def test_price_above_mean_gives_positive_zscore(self) -> None:
        """Linearly increasing prices → current price always above rolling mean."""
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = np.arange(1.0, n + 1.0)
        df = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1},
            index=dates,
        )
        result = ZScoreFeature(10).compute(df)
        valid = result.features["zscore_10"].dropna()
        assert (valid > 0).all()

    def test_constant_price_gives_nan(self) -> None:
        """std = 0 → denominator replaced with NaN."""
        ohlcv = _make_flat_ohlcv(50)
        result = ZScoreFeature(10).compute(ohlcv)
        valid = result.features["zscore_10"].dropna()
        assert len(valid) == 0

    def test_nan_for_insufficient_data(self) -> None:
        ohlcv = _make_ohlcv(5)
        result = ZScoreFeature(20).compute(ohlcv)
        assert result.features["zscore_20"].isna().all()

    def test_result_index_matches_input(self) -> None:
        ohlcv = _make_ohlcv(60)
        result = ZScoreFeature(20).compute(ohlcv)
        assert result.features.index.equals(ohlcv.index)


# ---------------------------------------------------------------------------
# HigherMomentFeature
# ---------------------------------------------------------------------------

class TestHigherMomentFeature:
    def test_default_window(self) -> None:
        assert HigherMomentFeature().window == 20

    def test_lookback(self) -> None:
        assert HigherMomentFeature(20).lookback == 21
        assert HigherMomentFeature(10).lookback == 11

    def test_feature_names(self) -> None:
        result = HigherMomentFeature(20).compute(_make_ohlcv(60))
        assert result.feature_names == ["skew_20", "kurt_20"]
        assert result.method == "higher_moments"

    def test_both_columns_present(self) -> None:
        result = HigherMomentFeature(10).compute(_make_ohlcv(50))
        assert "skew_10" in result.features.columns
        assert "kurt_10" in result.features.columns

    def test_nan_for_insufficient_data(self) -> None:
        ohlcv = _make_ohlcv(5)
        result = HigherMomentFeature(20).compute(ohlcv)
        assert result.features["skew_20"].isna().all()
        assert result.features["kurt_20"].isna().all()

    def test_constant_returns_give_degenerate_moments(self) -> None:
        """Constant price → constant 0 returns → degenerate skew/kurt."""
        ohlcv = _make_flat_ohlcv(50)
        result = HigherMomentFeature(10).compute(ohlcv)
        skew = result.features["skew_10"].dropna()
        kurt = result.features["kurt_10"].dropna()
        # pandas rolling skew of all-zero returns = 0.0
        if len(skew) > 0:
            np.testing.assert_allclose(skew.values, 0.0, atol=1e-12)
        # pandas rolling excess kurtosis of all-zero returns = -3.0
        if len(kurt) > 0:
            np.testing.assert_allclose(kurt.values, -3.0, atol=1e-12)

    def test_right_skewed_returns_give_positive_skew(self) -> None:
        """Distribution with large positive outlier → positive skewness."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Mostly small returns, one large spike at bar 50
        rng = np.random.default_rng(0)
        returns = rng.standard_normal(n) * 0.001
        returns[50] = 0.20  # large positive outlier
        prices = 100.0 * np.exp(np.cumsum(returns))
        df = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1},
            index=dates,
        )
        result = HigherMomentFeature(window=30).compute(df)
        skew = result.features["skew_30"]
        # The window containing bar 50 should show positive skew
        assert skew.iloc[55] > 0

    def test_result_index_matches_input(self) -> None:
        ohlcv = _make_ohlcv(60)
        result = HigherMomentFeature().compute(ohlcv)
        assert result.features.index.equals(ohlcv.index)


# ---------------------------------------------------------------------------
# HurstExponentFeature
# ---------------------------------------------------------------------------

class TestHurstExponentFeature:
    def test_default_window(self) -> None:
        assert HurstExponentFeature().window == 100

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 20"):
            HurstExponentFeature(window=19)

    def test_window_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            HurstExponentFeature(window=0)

    def test_valid_window_at_boundary(self) -> None:
        feat = HurstExponentFeature(window=20)
        assert feat.window == 20

    def test_lookback(self) -> None:
        assert HurstExponentFeature(100).lookback == 100
        assert HurstExponentFeature(50).lookback == 50

    def test_feature_name(self) -> None:
        ohlcv = _make_ohlcv(150, seed=0)
        result = HurstExponentFeature(window=100).compute(ohlcv)
        assert result.feature_names == ["hurst_100"]
        assert result.method == "hurst"

    def test_nan_before_window(self) -> None:
        ohlcv = _make_ohlcv(150, seed=0)
        result = HurstExponentFeature(window=100).compute(ohlcv)
        hurst = result.features["hurst_100"]
        # First 100 bars should be NaN
        assert hurst.iloc[:100].isna().all()

    def test_values_in_range_0_to_1(self) -> None:
        ohlcv = _make_ohlcv(200, vol=0.01, seed=5)
        result = HurstExponentFeature(window=100).compute(ohlcv)
        valid = result.features["hurst_100"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_trending_series_gives_hurst_above_half(self) -> None:
        """Positively autocorrelated returns → H > 0.5."""
        ohlcv = _make_trending_ohlcv(n=500, rho=0.7, seed=99)
        result = HurstExponentFeature(window=100).compute(ohlcv)
        valid = result.features["hurst_100"].dropna()
        assert len(valid) > 0
        assert valid.mean() > 0.5

    def test_mean_reverting_series_gives_hurst_below_half(self) -> None:
        """Alternating prices → H < 0.5."""
        ohlcv = _make_mean_reverting_ohlcv(n=250)
        result = HurstExponentFeature(window=100).compute(ohlcv)
        valid = result.features["hurst_100"].dropna()
        assert len(valid) > 0
        assert (valid < 0.5).mean() > 0.5

    def test_result_index_matches_input(self) -> None:
        ohlcv = _make_ohlcv(150, seed=0)
        result = HurstExponentFeature(window=100).compute(ohlcv)
        assert result.features.index.equals(ohlcv.index)


# ---------------------------------------------------------------------------
# VolatilityFeature
# ---------------------------------------------------------------------------

class TestVolatilityFeature:
    def test_default_windows(self) -> None:
        assert VolatilityFeature().windows == [5, 20, 60]

    def test_lookback(self) -> None:
        assert VolatilityFeature([5, 20, 60]).lookback == 61
        assert VolatilityFeature([10]).lookback == 11

    def test_feature_names_multi_window(self) -> None:
        result = VolatilityFeature([5, 20, 60]).compute(_make_ohlcv(100))
        assert "vol_5" in result.feature_names
        assert "vol_20" in result.feature_names
        assert "vol_60" in result.feature_names
        assert "vol_ratio_5_60" in result.feature_names
        assert result.method == "volatility"

    def test_no_ratio_column_for_single_window(self) -> None:
        result = VolatilityFeature([10]).compute(_make_ohlcv(50))
        assert "vol_10" in result.features.columns
        ratio_cols = [c for c in result.features.columns if "ratio" in c]
        assert len(ratio_cols) == 0

    def test_vol_nonnegative(self) -> None:
        ohlcv = _make_ohlcv(100, vol=0.02)
        result = VolatilityFeature([10, 20]).compute(ohlcv)
        for w in (10, 20):
            valid = result.features[f"vol_{w}"].dropna()
            assert (valid >= 0).all()

    def test_constant_prices_give_zero_vol(self) -> None:
        ohlcv = _make_flat_ohlcv(50)
        result = VolatilityFeature([5, 10]).compute(ohlcv)
        for w in (5, 10):
            valid = result.features[f"vol_{w}"].dropna()
            np.testing.assert_allclose(valid.values, 0.0, atol=1e-12)

    def test_annualisation(self) -> None:
        """Constant daily return of r% → vol ≈ 0 (zero return dispersion)."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        daily_r = 0.01  # 1% per day
        prices = 100.0 * (1 + daily_r) ** np.arange(n)
        df = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1},
            index=dates,
        )
        result = VolatilityFeature([20]).compute(df)
        # pct_change of a geometric series → near-constant returns,
        # but floating-point noise makes std ≈ 1e-15 rather than exactly 0
        valid_vol = result.features["vol_20"].dropna()
        np.testing.assert_allclose(valid_vol.values, 0.0, atol=1e-10)

    def test_vol_ratio_short_over_long(self) -> None:
        """vol_ratio = vol_short / vol_long."""
        ohlcv = _make_ohlcv(150, vol=0.02, seed=7)
        result = VolatilityFeature([5, 60]).compute(ohlcv)
        f = result.features.dropna()

        expected_ratio = f["vol_5"] / f["vol_60"]
        pd.testing.assert_series_equal(
            f["vol_ratio_5_60"], expected_ratio, check_names=False, rtol=1e-10
        )

    def test_result_index_matches_input(self) -> None:
        ohlcv = _make_ohlcv(100)
        result = VolatilityFeature().compute(ohlcv)
        assert result.features.index.equals(ohlcv.index)

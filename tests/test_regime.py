"""Tests for HMM regime detection."""

import numpy as np
import pandas as pd
import pytest

from src.regime.hmm import RegimeDetector, RegimeResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trending_data(
    n_bars: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic OHLCV data with clear bull/bear regimes.

    First half trends up (bull), second half trends down (bear).
    """
    rng = np.random.RandomState(seed)

    # Bull regime: positive drift, low vol
    bull_returns = rng.normal(0.002, 0.008, n_bars // 2)
    # Bear regime: negative drift, high vol
    bear_returns = rng.normal(-0.003, 0.015, n_bars - n_bars // 2)

    returns = np.concatenate([bull_returns, bear_returns])
    close = 100.0 * np.exp(np.cumsum(returns))

    dates = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n_bars)),
            "high": close * (1 + abs(rng.normal(0, 0.005, n_bars))),
            "low": close * (1 - abs(rng.normal(0, 0.005, n_bars))),
            "close": close,
            "volume": rng.randint(1000000, 5000000, n_bars),
        },
        index=dates,
    )
    return df


def _make_three_regime_data(
    n_per_regime: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic data with 3 clear regimes: bull, sideways, bear."""
    rng = np.random.RandomState(seed)

    bull = rng.normal(0.003, 0.008, n_per_regime)
    sideways = rng.normal(0.0, 0.005, n_per_regime)
    bear = rng.normal(-0.003, 0.015, n_per_regime)

    returns = np.concatenate([bull, sideways, bear])
    n_bars = len(returns)
    close = 100.0 * np.exp(np.cumsum(returns))

    dates = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.ones(n_bars) * 1000000,
        },
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# TestRegimeResult
# ---------------------------------------------------------------------------


class TestRegimeResult:
    def test_creation(self):
        regimes = pd.Series([0, 1, 0, 1], name="regime")
        result = RegimeResult(regimes=regimes)
        assert len(result.regimes) == 4
        assert result.current_regime == 0

    def test_with_all_fields(self):
        regimes = pd.Series([0, 1, 2], name="regime")
        result = RegimeResult(
            regimes=regimes,
            regime_names={0: "bear", 1: "sideways", 2: "bull"},
            transition_matrix=np.eye(3),
            regime_means={0: -0.001, 1: 0.0, 2: 0.002},
            regime_vols={0: 0.02, 1: 0.01, 2: 0.015},
            current_regime=2,
        )
        assert result.regime_names[2] == "bull"
        assert result.current_regime == 2
        assert result.transition_matrix.shape == (3, 3)


# ---------------------------------------------------------------------------
# TestRegimeDetectorInit
# ---------------------------------------------------------------------------


class TestRegimeDetectorInit:
    def test_default_parameters(self):
        detector = RegimeDetector()
        assert detector.n_regimes == 3
        assert detector.vol_window == 20
        assert detector.n_iter == 100
        assert detector.random_state == 42
        assert not detector.is_fitted

    def test_custom_parameters(self):
        detector = RegimeDetector(n_regimes=2, vol_window=10, n_iter=50, random_state=0)
        assert detector.n_regimes == 2
        assert detector.vol_window == 10

    def test_n_regimes_too_small_raises(self):
        with pytest.raises(ValueError, match="n_regimes must be >= 2"):
            RegimeDetector(n_regimes=1)

    def test_n_regimes_zero_raises(self):
        with pytest.raises(ValueError, match="n_regimes must be >= 2"):
            RegimeDetector(n_regimes=0)

    def test_vol_window_too_small_raises(self):
        with pytest.raises(ValueError, match="vol_window must be >= 2"):
            RegimeDetector(vol_window=1)

    def test_is_fitted_false_initially(self):
        detector = RegimeDetector()
        assert detector.is_fitted is False
        assert detector.model is None


# ---------------------------------------------------------------------------
# TestFitPredict
# ---------------------------------------------------------------------------


class TestFitPredict:
    def test_returns_regime_result(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        result = detector.fit_predict(data)
        assert isinstance(result, RegimeResult)

    def test_regime_series_length(self):
        data = _make_trending_data(n_bars=300)
        detector = RegimeDetector(n_regimes=2, vol_window=20)
        result = detector.fit_predict(data)
        # After dropna: lose 1 for pct_change + 19 for rolling window = 20
        expected_len = 300 - 20
        assert len(result.regimes) == expected_len

    def test_regime_values_in_range(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=3)
        result = detector.fit_predict(data)
        unique_values = set(result.regimes.unique())
        assert unique_values.issubset({0, 1, 2})

    def test_two_regime_labeling(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        result = detector.fit_predict(data)
        assert set(result.regime_names.values()) == {"bull", "bear"}

    def test_three_regime_labeling(self):
        data = _make_three_regime_data()
        detector = RegimeDetector(n_regimes=3)
        result = detector.fit_predict(data)
        assert set(result.regime_names.values()) == {
            "bull",
            "bear",
            "sideways",
        }

    def test_transition_matrix_shape(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=3)
        result = detector.fit_predict(data)
        assert result.transition_matrix.shape == (3, 3)

    def test_transition_matrix_rows_sum_to_one(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        result = detector.fit_predict(data)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_regime_means_populated(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        result = detector.fit_predict(data)
        assert len(result.regime_means) == 2
        for v in result.regime_means.values():
            assert isinstance(v, float)

    def test_regime_vols_populated(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        result = detector.fit_predict(data)
        assert len(result.regime_vols) == 2
        for v in result.regime_vols.values():
            assert v >= 0.0

    def test_current_regime_is_valid(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=3)
        result = detector.fit_predict(data)
        assert result.current_regime in range(3)

    def test_is_fitted_after_fit_predict(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        detector.fit_predict(data)
        assert detector.is_fitted is True
        assert detector.model is not None

    def test_missing_close_column_raises(self):
        data = pd.DataFrame({"price": [1, 2, 3]})
        detector = RegimeDetector(n_regimes=2)
        with pytest.raises(ValueError, match="close"):
            detector.fit_predict(data)

    def test_insufficient_data_raises(self):
        # Only 5 rows, vol_window=20 → not enough after dropna
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104]},
            index=pd.date_range("2020-01-01", periods=5),
        )
        detector = RegimeDetector(n_regimes=2, vol_window=20)
        with pytest.raises(ValueError, match="Insufficient data"):
            detector.fit_predict(data)

    def test_bull_regime_has_higher_mean(self):
        """The regime labeled 'bull' should have higher mean than 'bear'."""
        data = _make_trending_data(n_bars=500, seed=42)
        detector = RegimeDetector(n_regimes=2)
        result = detector.fit_predict(data)

        bull_id = [k for k, v in result.regime_names.items() if v == "bull"][0]
        bear_id = [k for k, v in result.regime_names.items() if v == "bear"][0]
        assert result.regime_means[bull_id] > result.regime_means[bear_id]

    def test_deterministic_with_same_seed(self):
        data = _make_trending_data()
        d1 = RegimeDetector(n_regimes=2, random_state=42)
        d2 = RegimeDetector(n_regimes=2, random_state=42)
        r1 = d1.fit_predict(data)
        r2 = d2.fit_predict(data)
        pd.testing.assert_series_equal(r1.regimes, r2.regimes)


# ---------------------------------------------------------------------------
# TestPredictCurrent
# ---------------------------------------------------------------------------


class TestPredictCurrent:
    def test_returns_integer(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        detector.fit_predict(data)
        regime = detector.predict_current(data)
        assert isinstance(regime, int)
        assert regime in range(2)

    def test_unfitted_raises(self):
        detector = RegimeDetector()
        data = _make_trending_data()
        with pytest.raises(RuntimeError, match="fitted"):
            detector.predict_current(data)

    def test_consistent_with_fit_predict(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        result = detector.fit_predict(data)
        current = detector.predict_current(data)
        assert current == result.current_regime

    def test_with_subset_of_data(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        detector.fit_predict(data)
        # Predict using only recent data
        recent = data.tail(100)
        regime = detector.predict_current(recent)
        assert regime in range(2)

    def test_insufficient_data_raises(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2, vol_window=20)
        detector.fit_predict(data)
        tiny = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        with pytest.raises(ValueError, match="Insufficient data"):
            detector.predict_current(tiny)


# ---------------------------------------------------------------------------
# TestPredictProba
# ---------------------------------------------------------------------------


class TestPredictProba:
    def test_returns_array(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=3)
        detector.fit_predict(data)
        proba = detector.predict_proba(data)
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (3,)

    def test_probabilities_sum_to_one(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        detector.fit_predict(data)
        proba = detector.predict_proba(data)
        np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-6)

    def test_probabilities_non_negative(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=3)
        detector.fit_predict(data)
        proba = detector.predict_proba(data)
        assert (proba >= 0.0).all()

    def test_unfitted_raises(self):
        detector = RegimeDetector()
        data = _make_trending_data()
        with pytest.raises(RuntimeError, match="fitted"):
            detector.predict_proba(data)

    def test_highest_proba_matches_predict(self):
        data = _make_trending_data()
        detector = RegimeDetector(n_regimes=2)
        detector.fit_predict(data)
        proba = detector.predict_proba(data)
        predicted = detector.predict_current(data)
        assert np.argmax(proba) == predicted


# ---------------------------------------------------------------------------
# TestBuildObservations
# ---------------------------------------------------------------------------


class TestBuildObservations:
    def test_columns(self):
        data = _make_trending_data()
        detector = RegimeDetector()
        obs = detector._build_observations(data)
        assert list(obs.columns) == ["return", "volatility"]

    def test_length_before_dropna(self):
        data = _make_trending_data(n_bars=100)
        detector = RegimeDetector(vol_window=20)
        obs = detector._build_observations(data)
        # Raw length = 100 (NaN values present but not dropped)
        assert len(obs) == 100

    def test_volatility_uses_window(self):
        data = _make_trending_data(n_bars=100)
        detector = RegimeDetector(vol_window=10)
        obs = detector._build_observations(data).dropna()
        # Lose 1 for pct_change + 9 for rolling(10)
        assert len(obs) == 100 - 10


# ---------------------------------------------------------------------------
# TestLabelRegimes
# ---------------------------------------------------------------------------


class TestLabelRegimes:
    def test_two_regimes(self):
        detector = RegimeDetector(n_regimes=2)
        names = detector._label_regimes({0: -0.001, 1: 0.002})
        assert names[0] == "bear"
        assert names[1] == "bull"

    def test_three_regimes(self):
        detector = RegimeDetector(n_regimes=3)
        names = detector._label_regimes({0: 0.002, 1: -0.003, 2: 0.0})
        assert names[1] == "bear"  # lowest
        assert names[0] == "bull"  # highest
        assert names[2] == "sideways"  # middle

    def test_four_regimes(self):
        detector = RegimeDetector(n_regimes=4)
        means = {0: 0.003, 1: -0.002, 2: 0.001, 3: -0.001}
        names = detector._label_regimes(means)
        assert names[1] == "bear"  # lowest
        assert names[0] == "bull"  # highest
        # Middle two are "sideways"
        assert names[3] == "sideways"
        assert names[2] == "sideways"


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_custom_vol_window(self):
        data = _make_trending_data(n_bars=200)
        detector = RegimeDetector(n_regimes=2, vol_window=5)
        result = detector.fit_predict(data)
        assert len(result.regimes) == 200 - 5  # 1 pct_change + 4 rolling

    def test_many_regimes(self):
        data = _make_trending_data(n_bars=500)
        detector = RegimeDetector(n_regimes=5)
        result = detector.fit_predict(data)
        assert len(result.regime_names) == 5
        assert "bull" in result.regime_names.values()
        assert "bear" in result.regime_names.values()

    def test_refit_on_new_data(self):
        """Fitting on different data should produce different results."""
        d1 = _make_trending_data(seed=42)
        d2 = _make_trending_data(seed=99)
        detector = RegimeDetector(n_regimes=2)
        r1 = detector.fit_predict(d1)
        r2 = detector.fit_predict(d2)
        # Both should work — model refits
        assert detector.is_fitted
        assert len(r1.regimes) > 0
        assert len(r2.regimes) > 0


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_detect_bull_and_bear_periods(self):
        """On strongly trending data, HMM should detect at least 2
        distinct regimes and the bull regime should have higher mean."""
        # Use extreme separation: strong bull + crash bear
        rng = np.random.RandomState(42)
        bull = rng.normal(0.005, 0.005, 400)
        bear = rng.normal(-0.008, 0.025, 400)
        returns = np.concatenate([bull, bear])
        close = 100.0 * np.exp(np.cumsum(returns))
        n = len(returns)
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        data = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.ones(n) * 1e6,
            },
            index=dates,
        )

        detector = RegimeDetector(n_regimes=2, vol_window=20)
        result = detector.fit_predict(data)

        # Both regimes should appear
        assert len(result.regimes.unique()) == 2

        # Bull regime should have higher mean return than bear
        bull_id = [k for k, v in result.regime_names.items() if v == "bull"][0]
        bear_id = [k for k, v in result.regime_names.items() if v == "bear"][0]
        assert result.regime_means[bull_id] > result.regime_means[bear_id]

    def test_full_workflow(self):
        """Fit, predict current, get probabilities."""
        data = _make_trending_data(n_bars=400)
        detector = RegimeDetector(n_regimes=3)

        result = detector.fit_predict(data)
        assert len(result.regime_names) == 3

        current = detector.predict_current(data.tail(100))
        assert current in range(3)

        proba = detector.predict_proba(data.tail(100))
        assert proba.shape == (3,)
        np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-6)

"""Unit tests for FeaturePipeline and PipelineResult."""

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline, PipelineResult
from src.features.statistical import ReturnFeature, ZScoreFeature
from src.features.technical import ATRFeature, RSIFeature, SMAFeature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with n bars."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.standard_normal(n) * 0.01
    prices = 100.0 * np.exp(np.cumsum(returns))
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


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_creation_all_fields(self) -> None:
        features = pd.DataFrame({"a": [1.0, 2.0]})
        target = pd.Series([0.1, 0.2], name="target")
        result = PipelineResult(
            features=features,
            feature_names=["a"],
            target=target,
            rows_dropped=5,
        )
        assert result.feature_names == ["a"]
        assert result.rows_dropped == 5
        assert result.target is not None
        assert len(result.target) == 2

    def test_default_target_is_none(self) -> None:
        result = PipelineResult(
            features=pd.DataFrame(),
            feature_names=[],
        )
        assert result.target is None
        assert result.rows_dropped == 0


# ---------------------------------------------------------------------------
# FeaturePipeline — construction
# ---------------------------------------------------------------------------


class TestFeaturePipelineConstruction:
    def test_empty_generators_raises(self) -> None:
        with pytest.raises(
            ValueError, match="At least one FeatureGenerator is required"
        ):
            FeaturePipeline([])

    def test_single_generator_total_lookback(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([10])])
        assert pipeline.total_lookback == 10

    def test_multiple_generators_total_lookback_is_max(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5]), RSIFeature(14), ATRFeature(7)])
        # lookbacks: 5, 15, 8 → max = 15
        assert pipeline.total_lookback == 15

    def test_generators_stored(self) -> None:
        gens = [SMAFeature([5]), RSIFeature(7)]
        pipeline = FeaturePipeline(gens)
        assert pipeline.generators == gens
        assert pipeline.generators is not gens  # defensive copy


# ---------------------------------------------------------------------------
# FeaturePipeline.run()
# ---------------------------------------------------------------------------


class TestFeaturePipelineRun:
    def test_returns_pipeline_result(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        result = pipeline.run(_make_ohlcv(100))
        assert isinstance(result, PipelineResult)

    def test_no_nan_in_output_features(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5]), RSIFeature(10)])
        result = pipeline.run(_make_ohlcv(200))
        assert not result.features.isna().any().any()

    def test_no_nan_in_output_target(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        result = pipeline.run(_make_ohlcv(200), target_horizon=5)
        assert result.target is not None
        assert not result.target.isna().any()

    def test_column_count_matches_generators(self) -> None:
        """Total columns = sum of each generator's feature count."""
        # SMAFeature([5,10]) → 4 cols; RSIFeature(7) → 1 col
        pipeline = FeaturePipeline([SMAFeature([5, 10]), RSIFeature(7)])
        result = pipeline.run(_make_ohlcv(200))
        assert len(result.features.columns) == 5
        assert len(result.feature_names) == 5

    def test_rows_dropped_is_positive(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([20])])
        result = pipeline.run(_make_ohlcv(200), target_horizon=5)
        assert result.rows_dropped > 0

    def test_rows_dropped_accounts_for_lookback_and_horizon(self) -> None:
        """rows_dropped = NaN-from-features + NaN-from-target-shift."""
        pipeline = FeaturePipeline([SMAFeature([20])])
        ohlcv = _make_ohlcv(200)
        result = pipeline.run(ohlcv, target_horizon=5)
        # rolling(20) gives first valid at row 19 (0-indexed), so 19 NaN feature rows.
        # Forward shift of 5 makes last 5 rows NaN. Total = 19 + 5 = 24.
        nan_from_features = pipeline.total_lookback - 1
        assert result.rows_dropped == nan_from_features + 5

    def test_output_row_count(self) -> None:
        n = 200
        pipeline = FeaturePipeline([SMAFeature([20])])
        result = pipeline.run(_make_ohlcv(n), target_horizon=5)
        nan_from_features = pipeline.total_lookback - 1
        expected_rows = n - nan_from_features - 5
        assert len(result.features) == expected_rows

    def test_target_type_return_is_float(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        result = pipeline.run(_make_ohlcv(200), target_type="return")
        assert result.target is not None
        assert result.target.dtype == float

    def test_target_type_return_formula(self) -> None:
        """Return target = ln(close[t+h] / close[t])."""
        ohlcv = _make_ohlcv(200)
        pipeline = FeaturePipeline([SMAFeature([5])])
        result = pipeline.run(ohlcv, target_horizon=1, target_type="return")

        # Verify a few rows manually
        close = ohlcv["close"]
        for idx in result.features.index[:5]:
            loc = ohlcv.index.get_loc(idx)
            expected_return = np.log(close.iloc[loc + 1] / close.iloc[loc])
            assert result.target[idx] == pytest.approx(expected_return, rel=1e-10)

    def test_target_type_direction_is_binary(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        result = pipeline.run(_make_ohlcv(200), target_type="direction")
        assert result.target is not None
        unique = set(result.target.unique())
        assert unique <= {0.0, 1.0}

    def test_target_type_direction_consistent_with_return(self) -> None:
        """direction=1 whenever the return target > 0."""
        ohlcv = _make_ohlcv(200, seed=7)
        pipeline_ret = FeaturePipeline([SMAFeature([5])])
        pipeline_dir = FeaturePipeline([SMAFeature([5])])
        result_ret = pipeline_ret.run(ohlcv, target_horizon=5, target_type="return")
        result_dir = pipeline_dir.run(ohlcv, target_horizon=5, target_type="direction")

        # Align on common index
        common = result_ret.features.index.intersection(result_dir.features.index)
        for idx in common:
            expected_dir = 1.0 if result_ret.target[idx] > 0 else 0.0
            assert result_dir.target[idx] == expected_dir

    def test_invalid_target_type_raises(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        with pytest.raises(ValueError, match="Unknown target_type"):
            pipeline.run(_make_ohlcv(100), target_type="invalid")

    def test_target_horizon_zero_raises(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        with pytest.raises(ValueError, match="target_horizon must be >= 1"):
            pipeline.run(_make_ohlcv(100), target_horizon=0)

    def test_target_horizon_negative_raises(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        with pytest.raises(ValueError, match="target_horizon must be >= 1"):
            pipeline.run(_make_ohlcv(100), target_horizon=-3)

    def test_feature_names_in_result_match_columns(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5, 10]), ATRFeature(7)])
        result = pipeline.run(_make_ohlcv(200))
        assert list(result.features.columns) == result.feature_names

    def test_output_index_is_datetimeindex(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        result = pipeline.run(_make_ohlcv(100))
        assert isinstance(result.features.index, pd.DatetimeIndex)

    def test_multiple_generators_columns_concatenated(self) -> None:
        """Feature columns from all generators appear in result."""
        pipeline = FeaturePipeline([SMAFeature([5]), RSIFeature(7), ATRFeature(5)])
        result = pipeline.run(_make_ohlcv(200))
        # SMA: sma_5, sma_5_ratio; RSI: rsi_7; ATR: atr_5, atr_5_pct
        for col in ("sma_5", "sma_5_ratio", "rsi_7", "atr_5", "atr_5_pct"):
            assert col in result.features.columns


# ---------------------------------------------------------------------------
# FeaturePipeline.compute_features_only()
# ---------------------------------------------------------------------------


class TestFeaturePipelineComputeFeaturesOnly:
    def test_returns_dataframe(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        result = pipeline.compute_features_only(_make_ohlcv(100))
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_in_output(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5]), RSIFeature(10)])
        result = pipeline.compute_features_only(_make_ohlcv(200))
        assert not result.isna().any().any()

    def test_no_target_column(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([5])])
        result = pipeline.compute_features_only(_make_ohlcv(100))
        assert "target" not in result.columns

    def test_correct_columns_present(self) -> None:
        pipeline = FeaturePipeline([SMAFeature([10]), ATRFeature(5)])
        result = pipeline.compute_features_only(_make_ohlcv(100))
        for col in ("sma_10", "sma_10_ratio", "atr_5", "atr_5_pct"):
            assert col in result.columns

    def test_fewer_rows_than_input(self) -> None:
        """NaN rows from lookback are dropped."""
        n = 100
        pipeline = FeaturePipeline([SMAFeature([20])])
        result = pipeline.compute_features_only(_make_ohlcv(n))
        # rolling(20) produces NaN for the first 19 rows (indices 0-18)
        assert len(result) == n - 19

    def test_consistent_with_run_features(self) -> None:
        """compute_features_only output matches the features from run()."""
        ohlcv = _make_ohlcv(200, seed=3)
        pipeline_a = FeaturePipeline([SMAFeature([10]), RSIFeature(7)])
        pipeline_b = FeaturePipeline([SMAFeature([10]), RSIFeature(7)])

        run_result = pipeline_a.run(ohlcv, target_horizon=5)
        feat_only = pipeline_b.compute_features_only(ohlcv)

        # feat_only has more rows (not trimmed by target_horizon)
        assert len(feat_only) >= len(run_result.features)
        # All rows in run_result.features should also appear in feat_only
        common_idx = run_result.features.index
        pd.testing.assert_frame_equal(
            feat_only.loc[common_idx],
            run_result.features,
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Integration — realistic full-pipeline scenario
# ---------------------------------------------------------------------------


class TestFeaturePipelineIntegration:
    def test_full_pipeline_sma_rsi_atr(self) -> None:
        """End-to-end with 3 generators on 500 bars."""
        ohlcv = _make_ohlcv(500)
        pipeline = FeaturePipeline(
            [
                SMAFeature([5, 10, 20]),
                RSIFeature(14),
                ATRFeature(14),
            ]
        )
        result = pipeline.run(ohlcv, target_horizon=5, target_type="return")

        assert not result.features.isna().any().any()
        assert not result.target.isna().any()
        assert result.rows_dropped > 0
        # Expected columns: 6 (SMA) + 1 (RSI) + 2 (ATR) = 9
        assert len(result.features.columns) == 9

    def test_full_pipeline_with_statistical_features(self) -> None:
        """Pipeline mixing technical + statistical generators."""
        ohlcv = _make_ohlcv(500)
        pipeline = FeaturePipeline(
            [
                SMAFeature([10, 20]),
                RSIFeature(14),
                ReturnFeature([1, 5]),
                ZScoreFeature(20),
            ]
        )
        result = pipeline.run(ohlcv, target_horizon=10, target_type="direction")

        assert not result.features.isna().any().any()
        assert result.target is not None
        assert set(result.target.unique()) <= {0.0, 1.0}
        # Check lookback: max(20 SMA, 15 RSI, 6 Return, 20 ZScore) = 20
        assert pipeline.total_lookback == 20

    def test_pipeline_direction_target_has_both_classes(self) -> None:
        """With enough data and volatile prices, both 0 and 1 should appear."""
        ohlcv = _make_ohlcv(500, seed=99)
        pipeline = FeaturePipeline([SMAFeature([10])])
        result = pipeline.run(ohlcv, target_horizon=5, target_type="direction")
        assert result.target is not None
        assert 0.0 in result.target.values
        assert 1.0 in result.target.values

    def test_large_dataset_performance(self) -> None:
        """Pipeline with 2000 bars completes without error."""
        ohlcv = _make_ohlcv(2000)
        pipeline = FeaturePipeline([SMAFeature([5, 20, 50]), RSIFeature(14)])
        result = pipeline.run(ohlcv, target_horizon=5)
        assert len(result.features) > 0
        assert not result.features.isna().any().any()

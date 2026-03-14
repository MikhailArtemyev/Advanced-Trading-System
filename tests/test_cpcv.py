"""Unit tests for Combinatorial Purged Cross-Validation (CPCV)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.base_model import MLModel, ModelPrediction, TrainResult
from src.validation.cpcv import (
    CPCVResult,
    CPCVSplit,
    CPCVValidator,
    _default_scoring_fn,
    _find_contiguous_blocks,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubMLModel(MLModel):
    """Deterministic model that returns fixed or sign-based predictions."""

    def __init__(self, mode: str = "fixed", signal: float = 0.5) -> None:
        self._mode = mode
        self._signal = signal
        self._trained = False
        self.feature_names: list[str] = []
        self.train_count = 0

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | None = None,
    ) -> TrainResult:
        self.feature_names = list(features.columns)
        self._trained = True
        self.train_count += 1
        self._target_mean = float(target.mean()) if len(target) > 0 else 0.0
        return TrainResult(
            train_score=0.8,
            n_train_samples=len(features),
            n_features=features.shape[1],
        )

    def predict(self, features: pd.DataFrame) -> list[ModelPrediction]:
        return [
            ModelPrediction(
                signal=self._signal if self._mode == "fixed" else self._target_mean,
                confidence=0.5,
                features_used=features.shape[1],
            )
            for _ in range(len(features))
        ]

    def save(self, path: str | Path) -> None:
        pass

    def load(self, path: str | Path) -> None:
        pass

    def get_feature_importance(self) -> dict[str, float]:
        return {}

    @property
    def is_trained(self) -> bool:
        return self._trained


class PerfectMLModel(MLModel):
    """Model that memorizes target signs and reproduces them perfectly."""

    def __init__(self) -> None:
        self._trained = False
        self._last_target: pd.Series | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | None = None,
    ) -> TrainResult:
        self.feature_names = list(features.columns)
        self._trained = True
        return TrainResult(train_score=1.0, n_train_samples=len(features))

    def predict(self, features: pd.DataFrame) -> list[ModelPrediction]:
        # Return positive signal for all (will work when actual target is mostly positive)
        return [
            ModelPrediction(signal=1.0, confidence=1.0, features_used=features.shape[1])
            for _ in range(len(features))
        ]

    def save(self, path: str | Path) -> None:
        pass

    def load(self, path: str | Path) -> None:
        pass

    def get_feature_importance(self) -> dict[str, float]:
        return {}

    @property
    def is_trained(self) -> bool:
        return self._trained


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(n: int = 100, n_cols: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic feature DataFrame."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, n_cols))
    columns = [f"feature_{i}" for i in range(n_cols)]
    return pd.DataFrame(data, columns=columns)


def _make_target(n: int = 100, seed: int = 42) -> pd.Series:
    """Generate synthetic binary target."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice([0, 1], size=n), name="target")


def _make_regression_target(n: int = 100, seed: int = 42) -> pd.Series:
    """Generate synthetic regression target (small returns)."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n) * 0.01, name="target")


# ---------------------------------------------------------------------------
# CPCVSplit dataclass
# ---------------------------------------------------------------------------


class TestCPCVSplit:
    def test_creation(self) -> None:
        split = CPCVSplit(
            train_indices=np.array([0, 1, 2]),
            test_indices=np.array([3, 4]),
            path_id=0,
        )
        assert len(split.train_indices) == 3
        assert len(split.test_indices) == 2
        assert split.path_id == 0

    def test_train_test_no_overlap(self) -> None:
        split = CPCVSplit(
            train_indices=np.array([0, 1, 2, 5, 6]),
            test_indices=np.array([3, 4]),
            path_id=0,
        )
        overlap = set(split.train_indices) & set(split.test_indices)
        assert overlap == set()


# ---------------------------------------------------------------------------
# CPCVResult dataclass
# ---------------------------------------------------------------------------


class TestCPCVResult:
    def test_creation(self) -> None:
        result = CPCVResult(
            scores=[0.6, 0.7, 0.8],
            mean_score=0.7,
            std_score=0.08,
            n_paths=3,
        )
        assert result.mean_score == pytest.approx(0.7)
        assert result.n_paths == 3

    def test_default_values(self) -> None:
        result = CPCVResult()
        assert result.scores == []
        assert result.mean_score == 0.0
        assert result.std_score == 0.0
        assert result.n_paths == 0
        assert result.predictions is None
        assert result.train_results == []


# ---------------------------------------------------------------------------
# CPCVValidator construction
# ---------------------------------------------------------------------------


class TestCPCVValidatorInit:
    def test_valid_initialization(self) -> None:
        v = CPCVValidator()
        assert v.n_splits == 6
        assert v.n_test_splits == 2
        assert v.purge_window == 5
        assert v.embargo_pct == 0.01

    def test_custom_parameters(self) -> None:
        v = CPCVValidator(
            n_splits=10, n_test_splits=3, purge_window=10, embargo_pct=0.05
        )
        assert v.n_splits == 10
        assert v.n_test_splits == 3
        assert v.purge_window == 10
        assert v.embargo_pct == 0.05

    def test_n_paths_property(self) -> None:
        v = CPCVValidator(n_splits=6, n_test_splits=2)
        assert v.n_paths == 15  # C(6,2)

    def test_n_paths_single_test(self) -> None:
        v = CPCVValidator(n_splits=5, n_test_splits=1)
        assert v.n_paths == 5  # C(5,1)

    def test_n_paths_three_test(self) -> None:
        v = CPCVValidator(n_splits=6, n_test_splits=3)
        assert v.n_paths == 20  # C(6,3)

    def test_n_splits_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="n_splits must be >= 3"):
            CPCVValidator(n_splits=2)

    def test_n_test_splits_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_test_splits must be >= 1"):
            CPCVValidator(n_test_splits=0)

    def test_n_test_splits_ge_n_splits_raises(self) -> None:
        with pytest.raises(ValueError, match="n_test_splits.*must be < n_splits"):
            CPCVValidator(n_splits=5, n_test_splits=5)

    def test_n_test_splits_gt_n_splits_raises(self) -> None:
        with pytest.raises(ValueError, match="n_test_splits.*must be < n_splits"):
            CPCVValidator(n_splits=5, n_test_splits=6)

    def test_negative_purge_window_raises(self) -> None:
        with pytest.raises(ValueError, match="purge_window must be >= 0"):
            CPCVValidator(purge_window=-1)

    def test_embargo_pct_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="embargo_pct must be in"):
            CPCVValidator(embargo_pct=1.5)

    def test_embargo_pct_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="embargo_pct must be in"):
            CPCVValidator(embargo_pct=-0.1)


# ---------------------------------------------------------------------------
# Helper: _find_contiguous_blocks
# ---------------------------------------------------------------------------


class TestFindContiguousBlocks:
    def test_single_block(self) -> None:
        blocks = _find_contiguous_blocks(np.array([3, 4, 5, 6]))
        assert blocks == [(3, 6)]

    def test_two_blocks(self) -> None:
        blocks = _find_contiguous_blocks(np.array([1, 2, 3, 7, 8, 9]))
        assert blocks == [(1, 3), (7, 9)]

    def test_single_element(self) -> None:
        blocks = _find_contiguous_blocks(np.array([5]))
        assert blocks == [(5, 5)]

    def test_all_separated(self) -> None:
        blocks = _find_contiguous_blocks(np.array([1, 3, 5, 7]))
        assert blocks == [(1, 1), (3, 3), (5, 5), (7, 7)]

    def test_empty_array(self) -> None:
        blocks = _find_contiguous_blocks(np.array([]))
        assert blocks == []

    def test_unsorted_input(self) -> None:
        blocks = _find_contiguous_blocks(np.array([5, 3, 4, 1, 2]))
        assert blocks == [(1, 5)]


# ---------------------------------------------------------------------------
# get_splits
# ---------------------------------------------------------------------------


class TestGetSplits:
    def test_correct_number_of_paths(self) -> None:
        v = CPCVValidator(n_splits=6, n_test_splits=2, purge_window=0, embargo_pct=0.0)
        splits = v.get_splits(60)
        assert len(splits) == 15

    def test_no_train_test_overlap(self) -> None:
        v = CPCVValidator(n_splits=5, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        splits = v.get_splits(50)
        for split in splits:
            overlap = set(split.train_indices) & set(split.test_indices)
            assert overlap == set()

    def test_all_samples_appear_in_test(self) -> None:
        """Every sample must appear as a test sample in at least one path."""
        v = CPCVValidator(n_splits=5, n_test_splits=2, purge_window=0, embargo_pct=0.0)
        n = 50
        splits = v.get_splits(n)
        all_test = set()
        for split in splits:
            all_test.update(split.test_indices)
        assert all_test == set(range(n))

    def test_indices_within_range(self) -> None:
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        n = 40
        splits = v.get_splits(n)
        for split in splits:
            assert all(0 <= i < n for i in split.train_indices)
            assert all(0 <= i < n for i in split.test_indices)

    def test_path_ids_sequential(self) -> None:
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        splits = v.get_splits(40)
        path_ids = [s.path_id for s in splits]
        assert path_ids == list(range(len(splits)))

    def test_small_n_samples_equal_n_splits(self) -> None:
        """One sample per group should still work."""
        v = CPCVValidator(n_splits=3, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        splits = v.get_splits(3)
        assert len(splits) == 3
        for split in splits:
            assert len(split.test_indices) == 1
            assert len(split.train_indices) == 2

    def test_n_samples_less_than_n_splits_raises(self) -> None:
        v = CPCVValidator(n_splits=5)
        with pytest.raises(ValueError, match="n_samples.*must be >= n_splits"):
            v.get_splits(4)

    def test_group_boundaries(self) -> None:
        """First group starts at 0, last group ends at n_samples-1."""
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        n = 40
        splits = v.get_splits(n)

        # Collect all test indices across all paths
        all_test = set()
        for split in splits:
            all_test.update(split.test_indices)

        assert 0 in all_test
        assert n - 1 in all_test

    def test_deterministic_splits(self) -> None:
        """Same inputs always produce same splits."""
        v = CPCVValidator(n_splits=5, n_test_splits=2, purge_window=3, embargo_pct=0.01)
        splits1 = v.get_splits(100)
        splits2 = v.get_splits(100)
        for s1, s2 in zip(splits1, splits2, strict=True):
            np.testing.assert_array_equal(s1.train_indices, s2.train_indices)
            np.testing.assert_array_equal(s1.test_indices, s2.test_indices)

    def test_remainder_absorbed_by_last_group(self) -> None:
        """With 13 samples and 4 splits, last group gets 4 samples."""
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        splits = v.get_splits(13)

        # Find the split where test includes the last index (12)
        last_split = [s for s in splits if 12 in s.test_indices][0]
        # Last group: indices 9, 10, 11, 12 (4 items vs 3 for other groups)
        assert len(last_split.test_indices) == 4


# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------


class TestPurging:
    def test_purge_removes_boundary_samples(self) -> None:
        """Samples near test boundaries should be purged from train."""
        v = CPCVValidator(n_splits=5, n_test_splits=1, purge_window=3, embargo_pct=0.0)
        splits = v.get_splits(50)

        # First test group is indices 0-9. With purge_window=3,
        # train indices 7,8,9 before are test (already excluded),
        # and indices 10,11,12 after should be purged.
        first_split = splits[0]
        train_set = set(first_split.train_indices)
        assert 10 not in train_set
        assert 11 not in train_set
        assert 12 not in train_set
        # But 13 should be in train
        assert 13 in train_set

    def test_purge_zero_window(self) -> None:
        """With purge_window=0, no purging occurs."""
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        splits = v.get_splits(40)
        for split in splits:
            # Train + test should cover all samples
            combined = set(split.train_indices) | set(split.test_indices)
            assert combined == set(range(40))

    def test_purge_does_not_remove_test_indices(self) -> None:
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=5, embargo_pct=0.0)
        splits = v.get_splits(40)
        for split in splits:
            # Test indices should be unchanged
            assert len(split.test_indices) == 10

    def test_purge_symmetric(self) -> None:
        """Purging removes samples on BOTH sides of test block."""
        # Test middle group (group 2 of 5) => test indices 20-29
        v = CPCVValidator(n_splits=5, n_test_splits=1, purge_window=3, embargo_pct=0.0)
        splits = v.get_splits(50)
        middle_split = splits[2]  # test group 2: indices 20-29

        train_set = set(middle_split.train_indices)
        # Before test block: 17,18,19 should be purged
        assert 17 not in train_set
        assert 18 not in train_set
        assert 19 not in train_set
        # After test block: 30,31,32 should be purged
        assert 30 not in train_set
        assert 31 not in train_set
        assert 32 not in train_set
        # But 16 and 33 should remain
        assert 16 in train_set
        assert 33 in train_set

    def test_purge_with_non_adjacent_test_groups(self) -> None:
        """When test groups are non-adjacent, purge around each block."""
        v = CPCVValidator(n_splits=5, n_test_splits=2, purge_window=2, embargo_pct=0.0)
        splits = v.get_splits(50)

        # Find the split with test groups 0 and 2 (non-adjacent)
        # Groups: 0=[0-9], 1=[10-19], 2=[20-29], 3=[30-39], 4=[40-49]
        # Test groups (0,2): test = [0-9, 20-29], train = [10-19, 30-39, 40-49]
        split_02 = splits[1]  # combo (0,2) is index 1 in C(5,2)
        train_set = set(split_02.train_indices)

        # After block [0-9]: 10,11 should be purged
        assert 10 not in train_set
        assert 11 not in train_set
        assert 12 in train_set
        # Before block [20-29]: 18,19 should be purged
        assert 18 not in train_set
        assert 19 not in train_set
        # After block [20-29]: 30,31 should be purged
        assert 30 not in train_set
        assert 31 not in train_set
        assert 32 in train_set

    def test_purge_large_window_reduces_train(self) -> None:
        """Large purge window can significantly reduce train set."""
        v = CPCVValidator(n_splits=3, n_test_splits=1, purge_window=20, embargo_pct=0.0)
        splits = v.get_splits(30)
        # Each group is 10 samples. Purge window of 20 can eliminate
        # most of the adjacent group.
        for split in splits:
            # Train should be smaller than 20 (original train = 20)
            assert len(split.train_indices) < 20


# ---------------------------------------------------------------------------
# Embargo
# ---------------------------------------------------------------------------


class TestEmbargo:
    def test_embargo_removes_post_test_samples(self) -> None:
        """Embargo removes samples immediately after test block."""
        v = CPCVValidator(n_splits=5, n_test_splits=1, purge_window=0, embargo_pct=0.10)
        splits = v.get_splits(50)

        # Test group 0: indices 0-9. embargo_size = max(1, int(10 * 0.10)) = 1
        # So index 10 should be embargoed
        first_split = splits[0]
        train_set = set(first_split.train_indices)
        assert 10 not in train_set
        assert 11 in train_set

    def test_embargo_zero_pct(self) -> None:
        """With embargo_pct=0.0, no embargo is applied."""
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        splits = v.get_splits(40)
        for split in splits:
            combined = set(split.train_indices) | set(split.test_indices)
            assert combined == set(range(40))

    def test_embargo_multiple_test_blocks(self) -> None:
        """Embargo applies after EACH contiguous test block independently."""
        v = CPCVValidator(n_splits=5, n_test_splits=2, purge_window=0, embargo_pct=0.10)
        splits = v.get_splits(50)

        # Find split with test groups (0, 2): non-adjacent blocks
        # Groups: 0=[0-9], 1=[10-19], 2=[20-29], 3=[30-39], 4=[40-49]
        # Test = [0-9, 20-29], embargo_size = max(1, int(20 * 0.10)) = 2
        split_02 = splits[1]  # combo (0,2)
        train_set = set(split_02.train_indices)

        # After block [0-9]: embargo indices 10, 11
        assert 10 not in train_set
        assert 11 not in train_set
        assert 12 in train_set

        # After block [20-29]: embargo indices 30, 31
        assert 30 not in train_set
        assert 31 not in train_set
        assert 32 in train_set

    def test_embargo_at_end_of_data(self) -> None:
        """Embargo at data end should not cause errors."""
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=0, embargo_pct=0.10)
        splits = v.get_splits(40)

        # Last test group is indices 30-39 (end of data)
        # Embargo after 39 would be indices 40+ which don't exist
        last_split = splits[3]
        train_set = set(last_split.train_indices)
        # Should not crash, train set should be all non-test indices
        assert len(train_set) == 30

    def test_embargo_combined_with_purge(self) -> None:
        """Purge and embargo work together to reduce train set."""
        v_no_embargo = CPCVValidator(
            n_splits=5, n_test_splits=1, purge_window=3, embargo_pct=0.0
        )
        v_with_embargo = CPCVValidator(
            n_splits=5, n_test_splits=1, purge_window=3, embargo_pct=0.10
        )

        splits_no = v_no_embargo.get_splits(50)
        splits_with = v_with_embargo.get_splits(50)

        # Embargo should remove additional samples beyond purge
        for s_no, s_with in zip(splits_no, splits_with, strict=True):
            assert len(s_with.train_indices) <= len(s_no.train_indices)


# ---------------------------------------------------------------------------
# Default scoring function
# ---------------------------------------------------------------------------


class TestDefaultScoringFn:
    def test_perfect_predictions(self) -> None:
        actual = pd.Series([1.0, -1.0, 1.0, -1.0])
        predictions = [
            ModelPrediction(signal=1.0, confidence=1.0, features_used=5),
            ModelPrediction(signal=-1.0, confidence=1.0, features_used=5),
            ModelPrediction(signal=1.0, confidence=1.0, features_used=5),
            ModelPrediction(signal=-1.0, confidence=1.0, features_used=5),
        ]
        assert _default_scoring_fn(actual, predictions) == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        actual = pd.Series([1.0, -1.0])
        predictions = [
            ModelPrediction(signal=-1.0, confidence=1.0, features_used=5),
            ModelPrediction(signal=1.0, confidence=1.0, features_used=5),
        ]
        assert _default_scoring_fn(actual, predictions) == pytest.approx(0.0)

    def test_half_correct(self) -> None:
        actual = pd.Series([1.0, -1.0, 1.0, -1.0])
        predictions = [
            ModelPrediction(signal=1.0, confidence=1.0, features_used=5),
            ModelPrediction(signal=1.0, confidence=1.0, features_used=5),  # wrong
            ModelPrediction(signal=-1.0, confidence=1.0, features_used=5),  # wrong
            ModelPrediction(signal=-1.0, confidence=1.0, features_used=5),
        ]
        assert _default_scoring_fn(actual, predictions) == pytest.approx(0.5)

    def test_empty_predictions(self) -> None:
        actual = pd.Series([], dtype=float)
        assert _default_scoring_fn(actual, []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CPCVValidator.run
# ---------------------------------------------------------------------------


class TestCPCVRun:
    def test_run_returns_result(self) -> None:
        features = _make_features(100)
        target = _make_target(100)
        model = StubMLModel(signal=0.5)
        v = CPCVValidator(n_splits=5, n_test_splits=1, purge_window=2, embargo_pct=0.01)
        result = v.run(features, target, model)
        assert isinstance(result, CPCVResult)
        assert result.n_paths == 5

    def test_run_scores_populated(self) -> None:
        features = _make_features(100)
        target = _make_target(100)
        model = StubMLModel(signal=0.5)
        v = CPCVValidator(n_splits=5, n_test_splits=1, purge_window=2, embargo_pct=0.01)
        result = v.run(features, target, model)
        assert len(result.scores) == 5
        assert all(0.0 <= s <= 1.0 for s in result.scores)

    def test_run_mean_std_correct(self) -> None:
        features = _make_features(100)
        target = _make_target(100)
        model = StubMLModel(signal=0.5)
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        result = v.run(features, target, model)
        assert result.mean_score == pytest.approx(np.mean(result.scores))
        assert result.std_score == pytest.approx(np.std(result.scores))

    def test_run_train_results_populated(self) -> None:
        features = _make_features(100)
        target = _make_target(100)
        model = StubMLModel(signal=0.5)
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        result = v.run(features, target, model)
        assert len(result.train_results) == 4
        for tr in result.train_results:
            assert isinstance(tr, TrainResult)
            assert tr.n_train_samples > 0

    def test_run_predictions_collected(self) -> None:
        features = _make_features(100)
        target = _make_target(100)
        model = StubMLModel(signal=0.5)
        v = CPCVValidator(n_splits=4, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        result = v.run(features, target, model)
        assert result.predictions is not None
        # Every sample appears in exactly one test set with n_test_splits=1
        # (with purge/embargo=0, all 100 samples should be tested)
        assert len(result.predictions) == 100

    def test_run_custom_scoring_fn(self) -> None:
        features = _make_features(60)
        target = _make_target(60)
        model = StubMLModel(signal=0.5)

        custom_fn_called = []

        def custom_scorer(actual: pd.Series, preds: list[ModelPrediction]) -> float:
            custom_fn_called.append(True)
            return 0.42

        v = CPCVValidator(n_splits=3, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        result = v.run(features, target, model, scoring_fn=custom_scorer)
        assert len(custom_fn_called) == 3
        assert all(s == pytest.approx(0.42) for s in result.scores)

    def test_run_model_retrained_each_path(self) -> None:
        features = _make_features(60)
        target = _make_target(60)
        model = StubMLModel(signal=0.5)
        v = CPCVValidator(n_splits=3, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        v.run(features, target, model)
        assert model.train_count == 3

    def test_run_with_purge_and_embargo(self) -> None:
        """Full run with purging and embargo produces valid results."""
        features = _make_features(200)
        target = _make_target(200)
        model = StubMLModel(signal=0.5)
        v = CPCVValidator(n_splits=6, n_test_splits=2, purge_window=5, embargo_pct=0.01)
        result = v.run(features, target, model)
        assert result.n_paths == 15
        assert len(result.scores) == 15
        assert result.mean_score > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_length_mismatch_raises(self) -> None:
        features = _make_features(100)
        target = _make_target(50)
        model = StubMLModel()
        v = CPCVValidator()
        with pytest.raises(ValueError, match="same length"):
            v.run(features, target, model)

    def test_empty_features_raises(self) -> None:
        features = pd.DataFrame()
        target = pd.Series([], dtype=float)
        model = StubMLModel()
        v = CPCVValidator()
        with pytest.raises(ValueError, match="must not be empty"):
            v.run(features, target, model)

    def test_nan_features_raises(self) -> None:
        features = _make_features(100)
        features.iloc[0, 0] = np.nan
        target = _make_target(100)
        model = StubMLModel()
        v = CPCVValidator()
        with pytest.raises(ValueError, match="NaN"):
            v.run(features, target, model)

    def test_nan_target_raises(self) -> None:
        features = _make_features(100)
        target = _make_target(100)
        target.iloc[0] = np.nan
        model = StubMLModel()
        v = CPCVValidator()
        with pytest.raises(ValueError, match="NaN"):
            v.run(features, target, model)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_test_split_like_kfold(self) -> None:
        """n_test_splits=1 behaves like purged k-fold."""
        v = CPCVValidator(n_splits=5, n_test_splits=1, purge_window=0, embargo_pct=0.0)
        splits = v.get_splits(50)
        assert len(splits) == 5
        # Each split has 40 train and 10 test
        for split in splits:
            assert len(split.test_indices) == 10
            assert len(split.train_indices) == 40

    def test_many_splits_few_samples(self) -> None:
        v = CPCVValidator(n_splits=5, n_test_splits=2, purge_window=0, embargo_pct=0.0)
        splits = v.get_splits(10)  # 2 samples per group
        assert len(splits) == 10
        for split in splits:
            assert len(split.test_indices) == 4  # 2 groups * 2 samples
            assert len(split.train_indices) == 6  # 3 groups * 2 samples

    def test_purge_window_exceeds_group_size(self) -> None:
        """Large purge can leave small train sets but should not crash."""
        v = CPCVValidator(n_splits=3, n_test_splits=1, purge_window=50, embargo_pct=0.0)
        splits = v.get_splits(30)  # 10 per group, purge window 50
        # Should not crash; train sets will be very small or empty
        for split in splits:
            assert len(split.test_indices) == 10

    def test_full_coverage_no_purge(self) -> None:
        """Without purge/embargo, train+test cover all samples."""
        v = CPCVValidator(n_splits=4, n_test_splits=2, purge_window=0, embargo_pct=0.0)
        splits = v.get_splits(40)
        for split in splits:
            combined = set(split.train_indices) | set(split.test_indices)
            assert combined == set(range(40))

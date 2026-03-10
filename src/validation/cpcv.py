"""Combinatorial Purged Cross-Validation (CPCV).

Implements the CPCV framework from Marcos López de Prado for
proper validation of ML models on financial time-series data.

Standard k-fold CV leaks information through overlapping
forward-return windows. CPCV fixes this via:
    - Purging: removing train samples whose target overlaps with test
    - Embargo: adding a buffer after purge boundaries
    - Combinatorial paths: C(N, k) train/test splits for robust estimates

Reference:
    de Prado, M. L. (2018). Advances in Financial Machine Learning.
    Chapter 12: Backtesting through Cross-Validation.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..ml.base_model import MLModel, ModelPrediction, TrainResult


@dataclass
class CPCVSplit:
    """A single CPCV train/test split.

    Attributes:
        train_indices: Integer indices into the feature matrix for training.
        test_indices: Integer indices into the feature matrix for testing.
        path_id: Sequential identifier for this path.
    """

    train_indices: np.ndarray
    test_indices: np.ndarray
    path_id: int


@dataclass
class CPCVResult:
    """Aggregated results from CPCV validation.

    Attributes:
        scores: One score per path.
        mean_score: Average score across all paths.
        std_score: Standard deviation of scores.
        n_paths: Number of combinatorial paths evaluated.
        train_results: Per-path training results (feature importance, etc.).
        predictions: Concatenated OOS predictions indexed by original row.
            When a sample appears in multiple paths, the first occurrence is kept.
    """

    scores: list[float] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    n_paths: int = 0
    train_results: list[TrainResult] = field(default_factory=list)
    predictions: pd.Series | None = None


class CPCVValidator:
    """Combinatorial Purged Cross-Validation validator.

    Generates all C(n_splits, n_test_splits) train/test combinations,
    applies purging and embargo to prevent information leakage, trains
    a model on each path, and aggregates out-of-sample scores.

    Args:
        n_splits: Number of contiguous groups to divide data into.
        n_test_splits: Number of groups used as test per path.
        purge_window: Number of samples to remove on each side of
            test boundaries. Should be >= target_horizon used in
            feature pipeline.
        embargo_pct: Fraction of test size to use as embargo buffer
            after each test block. Set to 0.0 to disable.

    Raises:
        ValueError: If parameters are invalid.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_splits < 3:
            raise ValueError(f"n_splits must be >= 3, got {n_splits}")
        if n_test_splits < 1:
            raise ValueError(f"n_test_splits must be >= 1, got {n_test_splits}")
        if n_test_splits >= n_splits:
            raise ValueError(
                f"n_test_splits ({n_test_splits}) must be < n_splits ({n_splits})"
            )
        if purge_window < 0:
            raise ValueError(f"purge_window must be >= 0, got {purge_window}")
        if not 0.0 <= embargo_pct <= 1.0:
            raise ValueError(f"embargo_pct must be in [0, 1], got {embargo_pct}")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    @property
    def n_paths(self) -> int:
        """Number of combinatorial paths: C(n_splits, n_test_splits)."""
        return math.comb(self.n_splits, self.n_test_splits)

    def get_splits(self, n_samples: int) -> list[CPCVSplit]:
        """Generate all combinatorial purged train/test splits.

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            List of CPCVSplit objects, one per combinatorial path.

        Raises:
            ValueError: If n_samples < n_splits.
        """
        if n_samples < self.n_splits:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_splits ({self.n_splits})"
            )

        groups = self._make_groups(n_samples)
        splits: list[CPCVSplit] = []

        for path_id, test_group_indices in enumerate(
            itertools.combinations(range(self.n_splits), self.n_test_splits)
        ):
            test_idx = np.concatenate([groups[g] for g in test_group_indices])
            train_groups = [
                g for g in range(self.n_splits) if g not in test_group_indices
            ]
            train_idx = np.concatenate([groups[g] for g in train_groups])

            # Apply purging and embargo
            train_idx = self._purge(train_idx, test_idx, n_samples)
            train_idx = self._embargo(train_idx, test_idx, n_samples)

            splits.append(
                CPCVSplit(
                    train_indices=train_idx,
                    test_indices=test_idx,
                    path_id=path_id,
                )
            )

        return splits

    def run(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model: MLModel,
        scoring_fn: Callable[[pd.Series, list[ModelPrediction]], float] | None = None,
    ) -> CPCVResult:
        """Run CPCV validation on a model.

        Trains the model on each combinatorial path's train set,
        predicts on the test set, and scores the predictions.

        Args:
            features: Feature DataFrame (from FeaturePipeline.run().features).
            target: Target Series (from FeaturePipeline.run().target).
            model: MLModel implementation to validate.
            scoring_fn: Optional scoring function. Takes (actual_target, predictions)
                and returns a float score. Defaults to direction accuracy.

        Returns:
            CPCVResult with per-path scores and aggregated statistics.

        Raises:
            ValueError: If features and target have different lengths or contain NaN.
        """
        _validate_inputs(features, target)

        if scoring_fn is None:
            scoring_fn = _default_scoring_fn

        n_samples = len(features)
        splits = self.get_splits(n_samples)

        scores: list[float] = []
        train_results: list[TrainResult] = []
        all_predictions: dict[int, float] = {}

        for split in splits:
            if len(split.train_indices) == 0:
                continue

            train_features = features.iloc[split.train_indices]
            train_target = target.iloc[split.train_indices]
            test_features = features.iloc[split.test_indices]
            test_target = target.iloc[split.test_indices]

            train_result = model.train(train_features, train_target)
            train_results.append(train_result)

            predictions = model.predict(test_features)
            score = scoring_fn(test_target, predictions)
            scores.append(score)

            # Collect OOS predictions (first occurrence wins)
            for idx, pred in zip(split.test_indices, predictions, strict=True):
                if idx not in all_predictions:
                    all_predictions[idx] = pred.signal

        # Build predictions series
        pred_series = None
        if all_predictions:
            pred_indices = sorted(all_predictions.keys())
            pred_values = [all_predictions[i] for i in pred_indices]
            pred_series = pd.Series(
                pred_values,
                index=features.index[pred_indices],
                name="oos_signal",
            )

        mean_score = float(np.mean(scores)) if scores else 0.0
        std_score = float(np.std(scores)) if scores else 0.0

        return CPCVResult(
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            n_paths=len(splits),
            train_results=train_results,
            predictions=pred_series,
        )

    def _make_groups(self, n_samples: int) -> list[np.ndarray]:
        """Divide samples into n_splits contiguous groups.

        The last group absorbs any remainder from integer division.

        Args:
            n_samples: Total number of samples.

        Returns:
            List of arrays, each containing integer indices for one group.
        """
        group_size = n_samples // self.n_splits
        groups: list[np.ndarray] = []

        for i in range(self.n_splits):
            start = i * group_size
            if i == self.n_splits - 1:
                end = n_samples
            else:
                end = (i + 1) * group_size
            groups.append(np.arange(start, end))

        return groups

    def _purge(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Remove train samples within purge_window of test boundaries.

        Purging prevents information leakage from overlapping
        forward-return targets near train/test boundaries.

        Args:
            train_idx: Current train indices.
            test_idx: Test indices for this path.
            n_samples: Total number of samples.

        Returns:
            Purged train indices.
        """
        if self.purge_window == 0:
            return train_idx

        # Find boundaries of contiguous test blocks
        blocks = _find_contiguous_blocks(test_idx)
        purge_set: set[int] = set()

        for block_start, block_end in blocks:
            low = max(0, block_start - self.purge_window)
            high = min(n_samples - 1, block_end + self.purge_window)
            purge_set.update(range(low, high + 1))

        # Remove purged indices from train (but never remove test indices)
        test_set = set(test_idx)
        purge_train = purge_set - test_set
        return np.array([i for i in train_idx if i not in purge_train])

    def _embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Remove train samples in the embargo zone after test blocks.

        Embargo addresses serial correlation that persists even
        after purging.

        Args:
            train_idx: Current train indices (already purged).
            test_idx: Test indices for this path.
            n_samples: Total number of samples.

        Returns:
            Train indices with embargo applied.
        """
        if self.embargo_pct == 0.0:
            return train_idx

        embargo_size = max(1, int(len(test_idx) * self.embargo_pct))
        blocks = _find_contiguous_blocks(test_idx)
        embargo_set: set[int] = set()

        for _, block_end in blocks:
            embargo_start = block_end + 1
            embargo_end = min(embargo_start + embargo_size, n_samples)
            embargo_set.update(range(embargo_start, embargo_end))

        return np.array([i for i in train_idx if i not in embargo_set])


def _find_contiguous_blocks(indices: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous blocks in a sorted array of indices.

    Args:
        indices: Sorted integer indices.

    Returns:
        List of (start, end) tuples for each contiguous block.
    """
    if len(indices) == 0:
        return []

    sorted_idx = np.sort(indices)
    blocks: list[tuple[int, int]] = []
    block_start = int(sorted_idx[0])

    for i in range(1, len(sorted_idx)):
        if sorted_idx[i] != sorted_idx[i - 1] + 1:
            blocks.append((block_start, int(sorted_idx[i - 1])))
            block_start = int(sorted_idx[i])

    blocks.append((block_start, int(sorted_idx[-1])))
    return blocks


def _default_scoring_fn(
    actual: pd.Series,
    predictions: list[ModelPrediction],
) -> float:
    """Default scoring: direction accuracy.

    Computes the fraction of predictions where the signal direction
    matches the actual target direction.

    Args:
        actual: True target values.
        predictions: Model predictions.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if len(predictions) == 0:
        return 0.0

    actual_signs = np.sign(actual.values)
    pred_signs = np.sign([p.signal for p in predictions])
    return float(np.mean(actual_signs == pred_signs))


def _validate_inputs(features: pd.DataFrame, target: pd.Series) -> None:
    """Validate features and target for CPCV.

    Args:
        features: Feature DataFrame.
        target: Target Series.

    Raises:
        ValueError: If inputs are invalid.
    """
    if len(features) != len(target):
        raise ValueError(
            f"features ({len(features)}) and target ({len(target)}) "
            f"must have the same length"
        )
    if len(features) == 0:
        raise ValueError("features and target must not be empty")
    if features.isna().any().any():
        raise ValueError("features contain NaN values")
    if target.isna().any():
        raise ValueError("target contains NaN values")

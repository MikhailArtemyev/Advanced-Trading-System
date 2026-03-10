"""Feature pipeline that composes multiple FeatureGenerators.

Orchestrates feature computation, handles NaN trimming from lookback
windows and forward-shifted targets, and produces a clean feature matrix
ready for ML model training or live prediction.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base_feature import FeatureGenerator, FeatureResult


@dataclass
class PipelineResult:
    """Result of running the full feature pipeline.

    Attributes:
        features: Clean feature DataFrame with NaN rows removed.
        feature_names: List of all feature column names.
        target: Forward-return target Series, or None for predict-only runs.
        rows_dropped: Number of rows trimmed due to NaN (lookback + forward shift).
    """

    features: pd.DataFrame
    feature_names: list[str]
    target: pd.Series | None = None
    rows_dropped: int = 0


class FeaturePipeline:
    """Composes multiple FeatureGenerators into a single pipeline.

    Usage:
        pipeline = FeaturePipeline([SMAFeature(), RSIFeature()])
        result = pipeline.run(ohlcv, target_horizon=5)

    The pipeline:
        1. Runs each generator on the full OHLCV data.
        2. Concatenates all feature columns.
        3. Creates a forward-return target column (shifted by target_horizon bars).
        4. Drops all rows containing any NaN (lookback rows at start, target NaN at end).
        5. Returns a clean PipelineResult ready for training.

    For live prediction (no target needed), use compute_features_only().

    Args:
        generators: Non-empty list of FeatureGenerator instances.

    Raises:
        ValueError: If generators is empty.
    """

    def __init__(self, generators: list[FeatureGenerator]) -> None:
        if not generators:
            raise ValueError("At least one FeatureGenerator is required")
        self.generators = list(generators)

    @property
    def total_lookback(self) -> int:
        """Maximum lookback across all generators."""
        return max(g.lookback for g in self.generators)

    def run(
        self,
        ohlcv: pd.DataFrame,
        target_horizon: int = 5,
        target_type: str = "return",
    ) -> PipelineResult:
        """Run all generators and produce a clean feature matrix with target.

        Args:
            ohlcv: OHLCV DataFrame with DatetimeIndex.
            target_horizon: Number of bars ahead for the target variable. Must be >= 1.
            target_type: How to construct the target:
                - "return":    Forward log return ln(close[t+h] / close[t])
                - "direction": Binary 1 (positive return) / 0 (negative or zero return)

        Returns:
            PipelineResult with clean features, target, and diagnostics.

        Raises:
            ValueError: If target_horizon < 1 or target_type is unrecognised.
        """
        if target_horizon < 1:
            raise ValueError(f"target_horizon must be >= 1, got {target_horizon}")

        all_features, all_names = self._compute_all_features(ohlcv)

        close = ohlcv["close"]
        forward_log_return = np.log(close.shift(-target_horizon) / close)

        if target_type == "return":
            target = forward_log_return
        elif target_type == "direction":
            target = (forward_log_return > 0).astype(int).astype(float)
            target = target.where(forward_log_return.notna(), other=np.nan)
        else:
            raise ValueError(
                f"Unknown target_type: '{target_type}'. Must be 'return' or 'direction'."
            )

        target.name = "target"

        combined = pd.concat([all_features, target], axis=1)
        initial_rows = len(combined)
        combined = combined.dropna()
        rows_dropped = initial_rows - len(combined)

        return PipelineResult(
            features=combined[all_names],
            feature_names=all_names,
            target=combined["target"],
            rows_dropped=rows_dropped,
        )

    def compute_features_only(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute features without a target (for live / inference-time prediction).

        Returns the feature DataFrame after dropping rows with any NaN.
        Typically called with the most recent N bars to get the latest feature row.

        Args:
            ohlcv: OHLCV DataFrame with DatetimeIndex.

        Returns:
            Clean feature DataFrame with no NaN values.
        """
        all_features, _ = self._compute_all_features(ohlcv)
        return all_features.dropna()

    def _compute_all_features(
        self, ohlcv: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Run all generators and concatenate their outputs.

        Returns:
            Tuple of (concatenated feature DataFrame, list of all column names).
        """
        frames: list[pd.DataFrame] = []
        all_names: list[str] = []

        for gen in self.generators:
            result: FeatureResult = gen.compute(ohlcv)
            frames.append(result.features)
            all_names.extend(result.feature_names)

        all_features = (
            pd.concat(frames, axis=1) if frames else pd.DataFrame(index=ohlcv.index)
        )
        return all_features, all_names

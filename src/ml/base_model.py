"""Abstract base class for ML signal models.

ML models in this system:
1. Are trained on historical feature+target data
2. Produce signal predictions (probability or direction)
3. Must support serialization (save/load)
4. Report feature importance for interpretability
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ModelPrediction:
    """Result of a model prediction.

    Attributes:
        signal: Predicted signal strength (-1.0 to 1.0).
            Positive = long, negative = short, near 0 = no action.
        confidence: Model confidence (0.0 to 1.0).
        features_used: Number of features in the input.
    """

    signal: float
    confidence: float
    features_used: int


@dataclass
class TrainResult:
    """Result of model training.

    Attributes:
        train_score: In-sample score (accuracy for classification, R2 for regression).
        val_score: Validation score (if validation set provided).
        feature_importance: Feature name -> importance mapping.
        n_train_samples: Number of training samples.
        n_features: Number of features.
    """

    train_score: float
    val_score: float | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    n_train_samples: int = 0
    n_features: int = 0


class MLModel(ABC):
    """Abstract base class for ML signal generation models."""

    @abstractmethod
    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | None = None,
    ) -> TrainResult:
        """Train the model on feature+target data.

        Args:
            features: Training feature matrix.
            target: Training target series.
            val_features: Optional validation features.
            val_target: Optional validation target.

        Returns:
            TrainResult with scores and feature importance.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> list[ModelPrediction]:
        """Generate predictions for each row of features.

        Args:
            features: Feature matrix (one or more rows).

        Returns:
            List of ModelPrediction, one per row.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save trained model to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load trained model from disk."""
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance mapping."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Whether the model has been trained or loaded."""
        raise NotImplementedError

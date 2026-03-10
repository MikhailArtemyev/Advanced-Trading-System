"""XGBoost-based signal generation model.

Uses gradient boosted trees for binary classification
(up/down direction) or regression (forward return prediction).
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .base_model import MLModel, ModelPrediction, TrainResult


class XGBoostSignalModel(MLModel):
    """XGBoost signal model for trading.

    Supports both classification (direction prediction) and
    regression (return prediction) modes.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns per tree.
        mode: "classification" or "regression".
        random_state: Random seed for reproducibility.
        regression_signal_scale: Scaling factor for regression predictions
            when mapping to [-1, 1] signal range.
        **extra_params: Additional XGBoost parameters.

    Raises:
        ValueError: If mode is not "classification" or "regression".
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        mode: str = "classification",
        random_state: int = 42,
        regression_signal_scale: float = 100.0,
        **extra_params: Any,
    ) -> None:
        if mode not in ("classification", "regression"):
            raise ValueError(
                f"mode must be 'classification' or 'regression', got '{mode}'"
            )

        self.mode = mode
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.regression_signal_scale = regression_signal_scale

        self.params: dict[str, Any] = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            **extra_params,
        }

        if mode == "classification":
            self.params["objective"] = "binary:logistic"
            self.params["eval_metric"] = "auc"
        else:
            self.params["objective"] = "reg:squarederror"
            self.params["eval_metric"] = "rmse"

        self._model: xgb.XGBClassifier | xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | None = None,
    ) -> TrainResult:
        self.feature_names = list(features.columns)

        if self.mode == "classification" and target.nunique() < 2:
            raise ValueError(
                "Target must have at least 2 classes for classification mode"
            )

        if self.mode == "classification":
            self._model = xgb.XGBClassifier(
                n_estimators=self.n_estimators, **self.params
            )
        else:
            self._model = xgb.XGBRegressor(
                n_estimators=self.n_estimators, **self.params
            )

        eval_set = None
        if val_features is not None and val_target is not None:
            eval_set = [(val_features, val_target)]

        self._model.fit(features, target, eval_set=eval_set, verbose=False)

        train_score = float(self._model.score(features, target))
        val_score = None
        if val_features is not None and val_target is not None:
            val_score = float(self._model.score(val_features, val_target))

        importance = self.get_feature_importance()

        return TrainResult(
            train_score=train_score,
            val_score=val_score,
            feature_importance=importance,
            n_train_samples=len(features),
            n_features=len(self.feature_names),
        )

    def predict(self, features: pd.DataFrame) -> list[ModelPrediction]:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        if len(features) == 0:
            return []

        if list(features.columns) != self.feature_names:
            raise ValueError(
                f"Feature columns {list(features.columns)} do not match "
                f"training columns {self.feature_names}"
            )

        n_features = features.shape[1]
        assert self._model is not None  # guaranteed by is_trained check above

        if self.mode == "classification":
            probas = self._model.predict_proba(features)
            signals = (probas[:, 1] - 0.5) * 2
            confidences = np.abs(probas[:, 1] - 0.5) * 2
        else:
            raw_preds = self._model.predict(features)
            signals = np.clip(raw_preds * self.regression_signal_scale, -1.0, 1.0)
            confidences = np.minimum(
                np.abs(raw_preds) * self.regression_signal_scale / 2, 1.0
            )

        return [
            ModelPrediction(
                signal=float(signals[i]),
                confidence=float(confidences[i]),
                features_used=n_features,
            )
            for i in range(len(features))
        ]

    def save(self, path: str | Path) -> None:
        if not self.is_trained:
            raise RuntimeError("No trained model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model, str(path) + ".joblib")

        meta = {
            "feature_names": self.feature_names,
            "mode": self.mode,
            "regression_signal_scale": self.regression_signal_scale,
            "model_class": "XGBoostSignalModel",
        }
        with open(str(path) + ".meta.json", "w") as f:
            json.dump(meta, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)

        with open(str(path) + ".meta.json") as f:
            meta = json.load(f)

        if meta["mode"] != self.mode:
            raise ValueError(
                f"Saved model mode '{meta['mode']}' does not match "
                f"instance mode '{self.mode}'"
            )

        self._model = joblib.load(str(path) + ".joblib")
        self.feature_names = meta["feature_names"]
        self.regression_signal_scale = meta.get(
            "regression_signal_scale", self.regression_signal_scale
        )

    def get_feature_importance(self) -> dict[str, float]:
        if not self.is_trained:
            return {}
        assert self._model is not None  # guaranteed by is_trained check above
        importance = self._model.feature_importances_
        return dict(zip(self.feature_names, importance, strict=True))

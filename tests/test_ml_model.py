"""Unit tests for ML model base classes, XGBoost, and LightGBM models."""

import numpy as np
import pandas as pd
import pytest

from src.ml.base_model import MLModel, ModelPrediction, TrainResult
from src.ml.lightgbm_model import LightGBMSignalModel
from src.ml.xgboost_model import XGBoostSignalModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classification_data(
    n: int = 500, n_features: int = 10, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic binary classification dataset."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    raw = X["feat_0"] * 0.5 + X["feat_1"] * 0.3 + rng.standard_normal(n) * 0.2
    y = pd.Series((raw > 0).astype(float), name="target")
    return X, y


def _make_regression_data(
    n: int = 500, n_features: int = 10, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic regression dataset (simulating forward returns)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(X["feat_0"] * 0.001 + rng.standard_normal(n) * 0.005, name="target")
    return X, y


# ---------------------------------------------------------------------------
# ModelPrediction
# ---------------------------------------------------------------------------


class TestModelPrediction:
    def test_creation(self) -> None:
        pred = ModelPrediction(signal=0.5, confidence=0.8, features_used=10)
        assert pred.signal == 0.5
        assert pred.confidence == 0.8
        assert pred.features_used == 10

    def test_negative_signal(self) -> None:
        pred = ModelPrediction(signal=-0.7, confidence=0.3, features_used=5)
        assert pred.signal == -0.7


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------


class TestTrainResult:
    def test_creation_all_fields(self) -> None:
        result = TrainResult(
            train_score=0.85,
            val_score=0.80,
            feature_importance={"feat_0": 0.5, "feat_1": 0.3},
            n_train_samples=500,
            n_features=10,
        )
        assert result.train_score == 0.85
        assert result.val_score == 0.80
        assert result.n_train_samples == 500

    def test_defaults(self) -> None:
        result = TrainResult(train_score=0.9)
        assert result.val_score is None
        assert result.feature_importance == {}
        assert result.n_train_samples == 0
        assert result.n_features == 0


# ---------------------------------------------------------------------------
# MLModel ABC
# ---------------------------------------------------------------------------


class TestMLModelABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            MLModel()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# XGBoost — Classification
# ---------------------------------------------------------------------------


class TestXGBoostClassification:
    def test_train_returns_train_result(self) -> None:
        X, y = _make_classification_data()
        model = XGBoostSignalModel(n_estimators=10, mode="classification")
        result = model.train(X, y)
        assert isinstance(result, TrainResult)
        assert result.n_train_samples == len(X)
        assert result.n_features == X.shape[1]

    def test_train_score_between_0_and_1(self) -> None:
        X, y = _make_classification_data()
        model = XGBoostSignalModel(n_estimators=10)
        result = model.train(X, y)
        assert 0.0 <= result.train_score <= 1.0

    def test_val_score_returned_when_provided(self) -> None:
        X, y = _make_classification_data(n=600)
        X_train, y_train = X.iloc[:400], y.iloc[:400]
        X_val, y_val = X.iloc[400:], y.iloc[400:]
        model = XGBoostSignalModel(n_estimators=10)
        result = model.train(X_train, y_train, X_val, y_val)
        assert result.val_score is not None
        assert 0.0 <= result.val_score <= 1.0

    def test_predict_returns_list_of_predictions(self) -> None:
        X, y = _make_classification_data()
        model = XGBoostSignalModel(n_estimators=10)
        model.train(X, y)
        preds = model.predict(X.iloc[:5])
        assert len(preds) == 5
        assert all(isinstance(p, ModelPrediction) for p in preds)

    def test_predict_signal_in_minus1_to_1(self) -> None:
        X, y = _make_classification_data()
        model = XGBoostSignalModel(n_estimators=10)
        model.train(X, y)
        preds = model.predict(X)
        for p in preds:
            assert -1.0 <= p.signal <= 1.0

    def test_predict_confidence_in_0_to_1(self) -> None:
        X, y = _make_classification_data()
        model = XGBoostSignalModel(n_estimators=10)
        model.train(X, y)
        preds = model.predict(X)
        for p in preds:
            assert 0.0 <= p.confidence <= 1.0

    def test_predict_features_used_matches_input(self) -> None:
        X, y = _make_classification_data(n_features=7)
        model = XGBoostSignalModel(n_estimators=10)
        model.train(X, y)
        preds = model.predict(X.iloc[:1])
        assert preds[0].features_used == 7

    def test_predict_untrained_raises(self) -> None:
        model = XGBoostSignalModel(n_estimators=10)
        X, _ = _make_classification_data()
        with pytest.raises(RuntimeError, match="trained"):
            model.predict(X.iloc[:1])

    def test_predict_empty_returns_empty(self) -> None:
        X, y = _make_classification_data()
        model = XGBoostSignalModel(n_estimators=10)
        model.train(X, y)
        preds = model.predict(X.iloc[:0])
        assert preds == []

    def test_predict_column_mismatch_raises(self) -> None:
        X, y = _make_classification_data(n_features=5)
        model = XGBoostSignalModel(n_estimators=10)
        model.train(X, y)
        bad_X = X.rename(columns={"feat_0": "wrong"})
        with pytest.raises(ValueError, match="do not match"):
            model.predict(bad_X.iloc[:1])

    def test_feature_importance_keys_match(self) -> None:
        X, y = _make_classification_data()
        model = XGBoostSignalModel(n_estimators=10)
        model.train(X, y)
        importance = model.get_feature_importance()
        assert set(importance.keys()) == set(X.columns)

    def test_feature_importance_untrained_returns_empty(self) -> None:
        model = XGBoostSignalModel(n_estimators=10)
        assert model.get_feature_importance() == {}

    def test_save_load_roundtrip(self, tmp_path) -> None:
        X, y = _make_classification_data()
        model = XGBoostSignalModel(n_estimators=10)
        model.train(X, y)
        preds_before = model.predict(X.iloc[:5])

        save_path = tmp_path / "model"
        model.save(save_path)

        model2 = XGBoostSignalModel(n_estimators=10)
        model2.load(save_path)
        preds_after = model2.predict(X.iloc[:5])

        for pb, pa in zip(preds_before, preds_after, strict=True):
            assert pb.signal == pytest.approx(pa.signal, abs=1e-6)

    def test_save_untrained_raises(self, tmp_path) -> None:
        model = XGBoostSignalModel(n_estimators=10)
        with pytest.raises(RuntimeError, match="No trained model"):
            model.save(tmp_path / "model")

    def test_load_restores_feature_names(self, tmp_path) -> None:
        X, y = _make_classification_data(n_features=5)
        model = XGBoostSignalModel(n_estimators=10)
        model.train(X, y)
        model.save(tmp_path / "model")

        model2 = XGBoostSignalModel(n_estimators=10)
        model2.load(tmp_path / "model")
        assert model2.feature_names == list(X.columns)

    def test_load_mode_mismatch_raises(self, tmp_path) -> None:
        X, y = _make_classification_data()
        model = XGBoostSignalModel(n_estimators=10, mode="classification")
        model.train(X, y)
        model.save(tmp_path / "model")

        model2 = XGBoostSignalModel(n_estimators=10, mode="regression")
        with pytest.raises(ValueError, match="does not match"):
            model2.load(tmp_path / "model")

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            XGBoostSignalModel(mode="invalid")

    def test_is_trained_property(self) -> None:
        model = XGBoostSignalModel(n_estimators=10)
        assert not model.is_trained
        X, y = _make_classification_data()
        model.train(X, y)
        assert model.is_trained

    def test_single_class_target_raises(self) -> None:
        X, _ = _make_classification_data()
        y = pd.Series(np.ones(len(X)), name="target")
        model = XGBoostSignalModel(n_estimators=10)
        with pytest.raises(ValueError, match="at least 2 classes"):
            model.train(X, y)


# ---------------------------------------------------------------------------
# XGBoost — Regression
# ---------------------------------------------------------------------------


class TestXGBoostRegression:
    def test_train_regression_mode(self) -> None:
        X, y = _make_regression_data()
        model = XGBoostSignalModel(n_estimators=10, mode="regression")
        result = model.train(X, y)
        assert isinstance(result, TrainResult)

    def test_predict_signal_range(self) -> None:
        X, y = _make_regression_data()
        model = XGBoostSignalModel(n_estimators=10, mode="regression")
        model.train(X, y)
        preds = model.predict(X)
        for p in preds:
            assert -1.0 <= p.signal <= 1.0
            assert 0.0 <= p.confidence <= 1.0

    def test_custom_regression_signal_scale(self) -> None:
        X, y = _make_regression_data()
        model_default = XGBoostSignalModel(
            n_estimators=10, mode="regression", regression_signal_scale=100.0
        )
        model_custom = XGBoostSignalModel(
            n_estimators=10, mode="regression", regression_signal_scale=10.0
        )
        model_default.train(X, y)
        model_custom.train(X, y)

        preds_default = model_default.predict(X.iloc[:5])
        preds_custom = model_custom.predict(X.iloc[:5])

        # With lower scale, signals should generally be smaller in magnitude
        default_avg = np.mean([abs(p.signal) for p in preds_default])
        custom_avg = np.mean([abs(p.signal) for p in preds_custom])
        assert custom_avg <= default_avg + 0.1  # custom scale 10x smaller

    def test_save_load_roundtrip_regression(self, tmp_path) -> None:
        X, y = _make_regression_data()
        model = XGBoostSignalModel(n_estimators=10, mode="regression")
        model.train(X, y)
        preds_before = model.predict(X.iloc[:3])

        model.save(tmp_path / "reg_model")

        model2 = XGBoostSignalModel(n_estimators=10, mode="regression")
        model2.load(tmp_path / "reg_model")
        preds_after = model2.predict(X.iloc[:3])

        for pb, pa in zip(preds_before, preds_after, strict=True):
            assert pb.signal == pytest.approx(pa.signal, abs=1e-6)


# ---------------------------------------------------------------------------
# LightGBM — Classification
# ---------------------------------------------------------------------------


class TestLightGBMClassification:
    def test_train_returns_train_result(self) -> None:
        X, y = _make_classification_data()
        model = LightGBMSignalModel(n_estimators=10, mode="classification")
        result = model.train(X, y)
        assert isinstance(result, TrainResult)
        assert result.n_train_samples == len(X)
        assert result.n_features == X.shape[1]

    def test_train_score_between_0_and_1(self) -> None:
        X, y = _make_classification_data()
        model = LightGBMSignalModel(n_estimators=10)
        result = model.train(X, y)
        assert 0.0 <= result.train_score <= 1.0

    def test_val_score_returned_when_provided(self) -> None:
        X, y = _make_classification_data(n=600)
        X_train, y_train = X.iloc[:400], y.iloc[:400]
        X_val, y_val = X.iloc[400:], y.iloc[400:]
        model = LightGBMSignalModel(n_estimators=10)
        result = model.train(X_train, y_train, X_val, y_val)
        assert result.val_score is not None
        assert 0.0 <= result.val_score <= 1.0

    def test_predict_returns_list_of_predictions(self) -> None:
        X, y = _make_classification_data()
        model = LightGBMSignalModel(n_estimators=10)
        model.train(X, y)
        preds = model.predict(X.iloc[:5])
        assert len(preds) == 5
        assert all(isinstance(p, ModelPrediction) for p in preds)

    def test_predict_signal_in_minus1_to_1(self) -> None:
        X, y = _make_classification_data()
        model = LightGBMSignalModel(n_estimators=10)
        model.train(X, y)
        preds = model.predict(X)
        for p in preds:
            assert -1.0 <= p.signal <= 1.0

    def test_predict_confidence_in_0_to_1(self) -> None:
        X, y = _make_classification_data()
        model = LightGBMSignalModel(n_estimators=10)
        model.train(X, y)
        preds = model.predict(X)
        for p in preds:
            assert 0.0 <= p.confidence <= 1.0

    def test_predict_untrained_raises(self) -> None:
        model = LightGBMSignalModel(n_estimators=10)
        X, _ = _make_classification_data()
        with pytest.raises(RuntimeError, match="trained"):
            model.predict(X.iloc[:1])

    def test_predict_empty_returns_empty(self) -> None:
        X, y = _make_classification_data()
        model = LightGBMSignalModel(n_estimators=10)
        model.train(X, y)
        preds = model.predict(X.iloc[:0])
        assert preds == []

    def test_predict_column_mismatch_raises(self) -> None:
        X, y = _make_classification_data(n_features=5)
        model = LightGBMSignalModel(n_estimators=10)
        model.train(X, y)
        bad_X = X.rename(columns={"feat_0": "wrong"})
        with pytest.raises(ValueError, match="do not match"):
            model.predict(bad_X.iloc[:1])

    def test_feature_importance_keys_match(self) -> None:
        X, y = _make_classification_data()
        model = LightGBMSignalModel(n_estimators=10)
        model.train(X, y)
        importance = model.get_feature_importance()
        assert set(importance.keys()) == set(X.columns)

    def test_feature_importance_untrained_returns_empty(self) -> None:
        model = LightGBMSignalModel(n_estimators=10)
        assert model.get_feature_importance() == {}

    def test_save_load_roundtrip(self, tmp_path) -> None:
        X, y = _make_classification_data()
        model = LightGBMSignalModel(n_estimators=10)
        model.train(X, y)
        preds_before = model.predict(X.iloc[:5])

        save_path = tmp_path / "lgb_model"
        model.save(save_path)

        model2 = LightGBMSignalModel(n_estimators=10)
        model2.load(save_path)
        preds_after = model2.predict(X.iloc[:5])

        for pb, pa in zip(preds_before, preds_after, strict=True):
            assert pb.signal == pytest.approx(pa.signal, abs=1e-6)

    def test_save_untrained_raises(self, tmp_path) -> None:
        model = LightGBMSignalModel(n_estimators=10)
        with pytest.raises(RuntimeError, match="No trained model"):
            model.save(tmp_path / "model")

    def test_load_restores_feature_names(self, tmp_path) -> None:
        X, y = _make_classification_data(n_features=5)
        model = LightGBMSignalModel(n_estimators=10)
        model.train(X, y)
        model.save(tmp_path / "model")

        model2 = LightGBMSignalModel(n_estimators=10)
        model2.load(tmp_path / "model")
        assert model2.feature_names == list(X.columns)

    def test_load_mode_mismatch_raises(self, tmp_path) -> None:
        X, y = _make_classification_data()
        model = LightGBMSignalModel(n_estimators=10, mode="classification")
        model.train(X, y)
        model.save(tmp_path / "model")

        model2 = LightGBMSignalModel(n_estimators=10, mode="regression")
        with pytest.raises(ValueError, match="does not match"):
            model2.load(tmp_path / "model")

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            LightGBMSignalModel(mode="invalid")

    def test_is_trained_property(self) -> None:
        model = LightGBMSignalModel(n_estimators=10)
        assert not model.is_trained
        X, y = _make_classification_data()
        model.train(X, y)
        assert model.is_trained

    def test_single_class_target_raises(self) -> None:
        X, _ = _make_classification_data()
        y = pd.Series(np.ones(len(X)), name="target")
        model = LightGBMSignalModel(n_estimators=10)
        with pytest.raises(ValueError, match="at least 2 classes"):
            model.train(X, y)


# ---------------------------------------------------------------------------
# LightGBM — Regression
# ---------------------------------------------------------------------------


class TestLightGBMRegression:
    def test_train_regression_mode(self) -> None:
        X, y = _make_regression_data()
        model = LightGBMSignalModel(n_estimators=10, mode="regression")
        result = model.train(X, y)
        assert isinstance(result, TrainResult)

    def test_predict_signal_range(self) -> None:
        X, y = _make_regression_data()
        model = LightGBMSignalModel(n_estimators=10, mode="regression")
        model.train(X, y)
        preds = model.predict(X)
        for p in preds:
            assert -1.0 <= p.signal <= 1.0
            assert 0.0 <= p.confidence <= 1.0

    def test_save_load_roundtrip_regression(self, tmp_path) -> None:
        X, y = _make_regression_data()
        model = LightGBMSignalModel(n_estimators=10, mode="regression")
        model.train(X, y)
        preds_before = model.predict(X.iloc[:3])

        model.save(tmp_path / "reg_model")

        model2 = LightGBMSignalModel(n_estimators=10, mode="regression")
        model2.load(tmp_path / "reg_model")
        preds_after = model2.predict(X.iloc[:3])

        for pb, pa in zip(preds_before, preds_after, strict=True):
            assert pb.signal == pytest.approx(pa.signal, abs=1e-6)

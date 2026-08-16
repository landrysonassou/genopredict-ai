"""Tests unitaires pour le module `src.models`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import models


@pytest.fixture
def toy_dataset():
    """Jeu de données jouet avec un signal linéaire clair pour valider l'apprentissage."""
    rng = np.random.default_rng(7)
    n = 120
    X = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(0, 1, n),
            "polygenic_risk_score": rng.normal(0, 1, n),
        },
        index=[f"S{i}" for i in range(n)],
    )
    logits = 2.0 * X["polygenic_risk_score"] + 0.5 * X["feature_a"]
    prob = 1 / (1 + np.exp(-logits))
    y = pd.Series((rng.random(n) < prob).astype(int), index=X.index)
    return X, y


class TestRiskCategory:
    @pytest.mark.parametrize(
        "probability,expected",
        [
            (0.0, models.RiskCategory.LOW),
            (0.1, models.RiskCategory.LOW),
            (0.25, models.RiskCategory.MODERATE),
            (0.4, models.RiskCategory.MODERATE),
            (0.5, models.RiskCategory.HIGH),
            (0.74, models.RiskCategory.HIGH),
            (0.75, models.RiskCategory.VERY_HIGH),
            (1.0, models.RiskCategory.VERY_HIGH),
        ],
    )
    def test_boundaries(self, probability, expected) -> None:
        assert models.RiskCategory.from_probability(probability) == expected

    @pytest.mark.parametrize("bad_value", [-0.1, 1.1])
    def test_out_of_range_raises(self, bad_value) -> None:
        with pytest.raises(ValueError):
            models.RiskCategory.from_probability(bad_value)


class TestTrainModel:
    def test_xgboost_training_returns_model_and_report(self, toy_dataset) -> None:
        X, y = toy_dataset
        model, report = models.train_model(X, y, model_type="xgboost", n_splits=4)
        assert model is not None
        assert isinstance(report, models.CrossValidationReport)
        assert 0.0 <= report.mean_roc_auc <= 1.0
        assert len(report.fold_metrics) == 4

    def test_random_forest_training_returns_model_and_report(self, toy_dataset) -> None:
        X, y = toy_dataset
        model, report = models.train_model(X, y, model_type="random_forest", n_splits=3)
        assert model is not None
        assert len(report.fold_metrics) == 3

    def test_learns_better_than_random_on_separable_signal(self, toy_dataset) -> None:
        X, y = toy_dataset
        _, report = models.train_model(X, y, model_type="xgboost", n_splits=5)
        # Le signal injecté est fortement corrélé à la cible : le modèle
        # doit significativement dépasser la performance aléatoire (0.5).
        assert report.mean_roc_auc > 0.65

    def test_invalid_model_type_raises(self, toy_dataset) -> None:
        X, y = toy_dataset
        with pytest.raises(models.ModelTrainingError):
            models.train_model(X, y, model_type="linear_regression")  # type: ignore[arg-type]

    def test_too_few_splits_raises(self, toy_dataset) -> None:
        X, y = toy_dataset
        with pytest.raises(models.ModelTrainingError):
            models.train_model(X, y, n_splits=1)

    def test_mismatched_lengths_raise(self, toy_dataset) -> None:
        X, y = toy_dataset
        with pytest.raises(models.ModelTrainingError):
            models.train_model(X, y.iloc[:-5], n_splits=3)

    def test_non_binary_target_raises(self, toy_dataset) -> None:
        X, y = toy_dataset
        bad_y = y.copy()
        bad_y.iloc[0] = 5
        with pytest.raises(models.ModelTrainingError):
            models.train_model(X, bad_y, n_splits=3)

    def test_confusion_matrix_is_consistent(self, toy_dataset) -> None:
        X, y = toy_dataset
        _, report = models.train_model(X, y, n_splits=5)
        cm = np.array(report.aggregate_confusion_matrix)
        assert cm.sum() == len(y)


class TestPredictRisk:
    def test_output_shape_and_columns(self, toy_dataset) -> None:
        X, y = toy_dataset
        model, _ = models.train_model(X, y, n_splits=3)
        risk_df = models.predict_risk(model, X)
        assert list(risk_df.columns) == ["probability", "risk_category"]
        assert risk_df.shape[0] == X.shape[0]
        assert risk_df["probability"].between(0, 1).all()

    def test_index_preserved(self, toy_dataset) -> None:
        X, y = toy_dataset
        model, _ = models.train_model(X, y, n_splits=3)
        risk_df = models.predict_risk(model, X)
        assert list(risk_df.index) == list(X.index)


class TestShapExplanations:
    def test_compute_shap_values_shape(self, toy_dataset) -> None:
        X, y = toy_dataset
        model, _ = models.train_model(X, y, model_type="xgboost", n_splits=3)
        explanation = models.compute_shap_values(model, X)
        assert explanation.values.shape[0] == X.shape[0]

    def test_top_shap_features_returns_sorted_dataframe(self, toy_dataset) -> None:
        X, y = toy_dataset
        model, _ = models.train_model(X, y, model_type="xgboost", n_splits=3)
        explanation = models.compute_shap_values(model, X)
        top = models.top_shap_features(explanation, sample_index=0, top_k=2)
        assert top.shape[0] == 2
        assert list(top.columns) == ["feature", "shap_value", "feature_value"]
        abs_vals = top["shap_value"].abs().tolist()
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_out_of_bounds_index_raises(self, toy_dataset) -> None:
        X, y = toy_dataset
        model, _ = models.train_model(X, y, model_type="xgboost", n_splits=3)
        explanation = models.compute_shap_values(model, X)
        with pytest.raises(IndexError):
            models.top_shap_features(explanation, sample_index=99999, top_k=2)

    def test_random_forest_shap_values_shape(self, toy_dataset) -> None:
        X, y = toy_dataset
        model, _ = models.train_model(X, y, model_type="random_forest", n_splits=3)
        explanation = models.compute_shap_values(model, X)
        assert explanation.values.shape[0] == X.shape[0]

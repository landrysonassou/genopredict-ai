"""Module de modélisation prédictive et d'interprétabilité.

Ce module entraîne des modèles de classification (XGBoost ou Random
Forest) pour prédire la susceptibilité/pathogénicité d'une maladie
génétique à partir de la matrice de caractéristiques produite par
`features.FeatureBuilder`. Il fournit une validation croisée stratifiée
rigoureuse, des métriques cliniques standard (sensibilité, spécificité,
ROC-AUC, matrice de confusion) et une interprétabilité par valeurs SHAP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

ModelType = Literal["xgboost", "random_forest"]

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


class ModelTrainingError(Exception):
    """Erreur levée lors d'une configuration ou d'une exécution invalide de l'entraînement."""


class RiskCategory(str, Enum):
    """Catégorie de risque clinique dérivée de la probabilité prédite."""

    LOW = "faible"
    MODERATE = "modéré"
    HIGH = "élevé"
    VERY_HIGH = "très élevé"

    @classmethod
    def from_probability(cls, probability: float) -> "RiskCategory":
        """Détermine la catégorie de risque à partir d'une probabilité prédite.

        Seuils cliniques (ajustables selon la maladie étudiée) :
            [0.00, 0.25) -> faible
            [0.25, 0.50) -> modéré
            [0.50, 0.75) -> élevé
            [0.75, 1.00] -> très élevé

        Args:
            probability: Probabilité prédite de pathogénicité (entre 0 et 1).

        Returns:
            La catégorie de risque correspondante.

        Raises:
            ValueError: Si `probability` n'est pas compris entre 0 et 1.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"La probabilité doit être comprise entre 0 et 1 (reçu {probability}).")
        if probability < 0.25:
            return cls.LOW
        if probability < 0.50:
            return cls.MODERATE
        if probability < 0.75:
            return cls.HIGH
        return cls.VERY_HIGH


@dataclass
class FoldMetrics:
    """Métriques calculées sur un pli (fold) de validation croisée."""

    fold_index: int
    roc_auc: float
    sensitivity: float
    specificity: float
    f1: float
    accuracy: float
    confusion_matrix: list[list[int]]


@dataclass
class CrossValidationReport:
    """Rapport agrégé de validation croisée stratifiée.

    Attributes:
        model_type: Type de modèle évalué.
        n_splits: Nombre de plis utilisés.
        fold_metrics: Métriques détaillées par pli.
        mean_roc_auc: Moyenne du ROC-AUC sur l'ensemble des plis.
        std_roc_auc: Écart-type du ROC-AUC sur l'ensemble des plis.
        mean_sensitivity: Moyenne de la sensibilité (rappel classe positive).
        mean_specificity: Moyenne de la spécificité (rappel classe négative).
        mean_f1: Moyenne du F1-score.
        mean_accuracy: Moyenne de l'exactitude.
        aggregate_confusion_matrix: Matrice de confusion agrégée (somme des
            matrices de confusion out-of-fold sur l'ensemble du jeu de données).
    """

    model_type: str
    n_splits: int
    fold_metrics: list[FoldMetrics] = field(default_factory=list)
    mean_roc_auc: float = 0.0
    std_roc_auc: float = 0.0
    mean_sensitivity: float = 0.0
    mean_specificity: float = 0.0
    mean_f1: float = 0.0
    mean_accuracy: float = 0.0
    aggregate_confusion_matrix: list[list[int]] = field(default_factory=list)

    def summary(self) -> str:
        """Retourne un résumé lisible des performances du modèle."""
        return (
            f"=== Validation Croisée Stratifiée ({self.model_type}, "
            f"{self.n_splits} plis) ===\n"
            f"ROC-AUC     : {self.mean_roc_auc:.4f} (+/- {self.std_roc_auc:.4f})\n"
            f"Sensibilité : {self.mean_sensitivity:.4f}\n"
            f"Spécificité : {self.mean_specificity:.4f}\n"
            f"F1-score    : {self.mean_f1:.4f}\n"
            f"Exactitude  : {self.mean_accuracy:.4f}\n"
            f"Matrice de confusion agrégée (TN, FP / FN, TP) :\n"
            f"{np.array(self.aggregate_confusion_matrix)}"
        )


def _build_model(model_type: ModelType, params: dict[str, Any] | None):
    """Instancie un classifieur non entraîné selon `model_type`."""
    if model_type == "xgboost":
        merged = {**DEFAULT_XGB_PARAMS, **(params or {})}
        return XGBClassifier(**merged)
    if model_type == "random_forest":
        merged = {**DEFAULT_RF_PARAMS, **(params or {})}
        return RandomForestClassifier(**merged)
    raise ModelTrainingError(
        f"model_type invalide : '{model_type}'. Attendu 'xgboost' ou 'random_forest'."
    )


def _fold_metrics_from_predictions(
    fold_index: int, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> FoldMetrics:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        # Se produit si un pli ne contient qu'une seule classe.
        auc = float("nan")
    accuracy = (tp + tn) / cm.sum() if cm.sum() > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return FoldMetrics(
        fold_index=fold_index,
        roc_auc=float(auc),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        f1=float(f1),
        accuracy=float(accuracy),
        confusion_matrix=cm.tolist(),
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: ModelType = "xgboost",
    n_splits: int = 5,
    random_state: int = 42,
    model_params: dict[str, Any] | None = None,
) -> tuple[Any, CrossValidationReport]:
    """Entraîne un modèle prédictif avec validation croisée stratifiée.

    La performance est évaluée de manière non biaisée par validation
    croisée stratifiée à `n_splits` plis (chaque pli conserve la
    proportion originale de cas positifs/négatifs). Le modèle final
    retourné est ensuite ré-entraîné sur l'intégralité de `X`/`y` pour
    maximiser l'usage des données disponibles en vue du déploiement ;
    les métriques rapportées proviennent exclusivement des plis de
    validation (jamais du modèle final).

    Args:
        X: Matrice de caractéristiques (échantillons x features), déjà
            produite par `features.FeatureBuilder` (aucune fuite de
            données ne doit avoir été introduite en amont).
        y: Vecteur cible binaire (statut de la maladie, 0/1), aligné sur
            l'index de `X`.
        model_type: "xgboost" ou "random_forest".
        n_splits: Nombre de plis de la validation croisée stratifiée
            (minimum 2).
        random_state: Graine aléatoire pour la reproductibilité.
        model_params: Hyperparamètres additionnels ou de remplacement
            pour le modèle sous-jacent (fusionnés avec les valeurs par
            défaut).

    Returns:
        Un tuple (modèle final entraîné sur toutes les données,
        CrossValidationReport agrégé sur les plis de validation).

    Raises:
        ModelTrainingError: Si `n_splits` < 2, si `X` et `y` ont des
            tailles incohérentes, ou si `y` n'est pas binaire.
    """
    if n_splits < 2:
        raise ModelTrainingError(f"n_splits doit être >= 2 (reçu {n_splits}).")
    if X.shape[0] != y.shape[0]:
        raise ModelTrainingError(
            f"X ({X.shape[0]} échantillons) et y ({y.shape[0]} échantillons) "
            "ont des tailles incompatibles."
        )
    unique_y = set(pd.unique(y))
    if not unique_y.issubset({0, 1}):
        raise ModelTrainingError(f"y doit être binaire (0/1). Valeurs trouvées : {unique_y}")

    X_array = X.to_numpy()
    y_array = y.to_numpy().astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_reports: list[FoldMetrics] = []
    oof_confusion = np.zeros((2, 2), dtype=int)

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(X_array, y_array), start=1):
        X_train, X_val = X_array[train_idx], X_array[val_idx]
        y_train, y_val = y_array[train_idx], y_array[val_idx]

        fold_model = _build_model(model_type, model_params)
        fold_model.fit(X_train, y_train)

        y_proba = fold_model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = _fold_metrics_from_predictions(fold_index, y_val, y_pred, y_proba)
        fold_reports.append(metrics)
        oof_confusion += np.array(metrics.confusion_matrix)

    valid_aucs = [m.roc_auc for m in fold_reports if not np.isnan(m.roc_auc)]
    report = CrossValidationReport(
        model_type=model_type,
        n_splits=n_splits,
        fold_metrics=fold_reports,
        mean_roc_auc=float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
        std_roc_auc=float(np.std(valid_aucs)) if valid_aucs else float("nan"),
        mean_sensitivity=float(np.mean([m.sensitivity for m in fold_reports])),
        mean_specificity=float(np.mean([m.specificity for m in fold_reports])),
        mean_f1=float(np.mean([m.f1 for m in fold_reports])),
        mean_accuracy=float(np.mean([m.accuracy for m in fold_reports])),
        aggregate_confusion_matrix=oof_confusion.tolist(),
    )
    logger.info(report.summary())

    # Le modèle final est ré-entraîné sur un DataFrame (plutôt qu'un ndarray) afin
    # que les noms de colonnes soient conservés nativement (`feature_names_in_`),
    # ce qui est requis par shap.TreeExplainer et utile pour la traçabilité.
    final_model = _build_model(model_type, model_params)
    final_model.fit(X, y_array)

    return final_model, report


def compute_shap_values(model: Any, X: pd.DataFrame) -> shap.Explanation:
    """Calcule les valeurs SHAP pour interpréter les prédictions du modèle.

    Utilise `shap.TreeExplainer`, adapté aux modèles à base d'arbres
    (XGBoost, Random Forest) pour un calcul exact et efficace des
    contributions marginales de chaque caractéristique à la prédiction.

    Args:
        model: Modèle entraîné (XGBClassifier ou RandomForestClassifier),
            typiquement issu de `train_model`.
        X: Matrice de caractéristiques pour laquelle calculer les
            explications (mêmes colonnes/ordre que lors de l'entraînement).

    Returns:
        Un objet `shap.Explanation` contenant les valeurs SHAP pour
        chaque échantillon et chaque caractéristique.
    """
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    return explanation


def top_shap_features(
    explanation: shap.Explanation, sample_index: int, top_k: int = 5
) -> pd.DataFrame:
    """Extrait les `top_k` caractéristiques les plus influentes pour un échantillon donné.

    Args:
        explanation: Objet retourné par `compute_shap_values`.
        sample_index: Index positionnel de l'échantillon à expliquer.
        top_k: Nombre de caractéristiques à retourner, triées par
            importance absolue décroissante.

    Returns:
        Un DataFrame avec les colonnes `feature`, `shap_value` et
        `feature_value`, trié par |shap_value| décroissant.

    Raises:
        IndexError: Si `sample_index` est hors des bornes de `explanation`.
    """
    if sample_index < 0 or sample_index >= explanation.values.shape[0]:
        raise IndexError(
            f"sample_index={sample_index} hors bornes (0 à {explanation.values.shape[0] - 1})."
        )
    values = explanation.values[sample_index]
    # Cas multi-classe (ex: XGBoost binaire retourne parfois (n_features, 2))
    if values.ndim == 2:
        values = values[:, 1]
    data = explanation.data[sample_index]
    feature_names = explanation.feature_names
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": values,
            "feature_value": data,
        }
    )
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).drop(columns="abs_shap").head(top_k)
    return df.reset_index(drop=True)


def predict_risk(model: Any, X_new: pd.DataFrame) -> pd.DataFrame:
    """Prédit la probabilité de pathogénicité et la catégorie de risque clinique.

    Args:
        model: Modèle entraîné (issu de `train_model`).
        X_new: Matrice de caractéristiques pour les nouveaux échantillons
            à prédire (mêmes colonnes/ordre que lors de l'entraînement).

    Returns:
        Un DataFrame indexé comme `X_new`, avec les colonnes
        `probability` (probabilité prédite de pathogénicité, entre 0 et 1)
        et `risk_category` (catégorie clinique dérivée, voir `RiskCategory`).
    """
    probabilities = model.predict_proba(X_new.to_numpy())[:, 1]
    categories = [RiskCategory.from_probability(float(p)).value for p in probabilities]
    return pd.DataFrame(
        {"probability": probabilities, "risk_category": categories}, index=X_new.index
    )

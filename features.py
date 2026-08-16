"""Module d'ingénierie des caractéristiques bio-informatiques.

Ce module transforme les données génotypiques et cliniques nettoyées
(issues de `ingestion.quality_control`) en une matrice de caractéristiques
numériques exploitable par les modèles de Machine Learning, en respectant
scrupuleusement la séparation train/test pour éviter toute fuite de
données (data leakage) : tout objet de transformation (imputation,
normalisation, encodage) doit être *fit* uniquement sur les données
d'entraînement puis appliqué (*transform*) sur les données de test.

Un score de risque polygénique (PRS) simplifié est également calculé à
partir des dosages alléliques et des tailles d'effet (`effect_size`)
fournies par les annotations fonctionnelles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineeringError(Exception):
    """Erreur levée lors d'une étape invalide d'ingénierie des caractéristiques."""


def compute_polygenic_risk_score(
    dosage_df: pd.DataFrame,
    effect_sizes: pd.Series,
    standardize: bool = True,
) -> pd.Series:
    """Calcule un score de risque polygénique (PRS) simplifié par échantillon.

    Le PRS est calculé comme la somme pondérée des dosages alléliques
    (0, 1 ou 2 allèles alternatifs) par leur taille d'effet (`beta`,
    typiquement issue d'une étude d'association pangénomique publique) :

        PRS_i = sum_v (dosage_{i,v} * beta_v)

    Les valeurs manquantes de dosage sont imputées par la moyenne du
    variant (calculée sur les échantillons fournis) avant sommation, afin
    qu'un génotype manquant n'annule pas la contribution des autres
    variants pour un même échantillon.

    Args:
        dosage_df: DataFrame (échantillons x variants) de dosages
            alléliques additifs (valeurs 0, 1, 2 ou NaN).
        effect_sizes: Série indexée par identifiant de variant, donnant
            la taille d'effet (beta) de chaque variant. Seuls les
            variants présents à la fois dans `dosage_df.columns` et
            `effect_sizes.index` sont utilisés.
        standardize: Si True, standardise le PRS résultant (moyenne 0,
            écart-type 1) sur la population fournie.

    Returns:
        Une Series indexée par échantillon contenant le PRS de chaque
        individu.

    Raises:
        FeatureEngineeringError: Si aucun variant commun n'est trouvé
            entre `dosage_df` et `effect_sizes`.
    """
    common_variants = [v for v in dosage_df.columns if v in effect_sizes.index]
    if not common_variants:
        raise FeatureEngineeringError(
            "Aucun variant commun entre la matrice de dosages et les tailles "
            "d'effet fournies : impossible de calculer le PRS."
        )

    dosage_subset = dosage_df[common_variants].apply(
        lambda col: col.fillna(col.mean()), axis=0
    )
    betas = effect_sizes.loc[common_variants]
    prs = dosage_subset.mul(betas, axis=1).sum(axis=1)
    prs.name = "polygenic_risk_score"

    if standardize:
        std = prs.std(ddof=0)
        if std > 0:
            prs = (prs - prs.mean()) / std
        else:
            logger.warning(
                "Écart-type du PRS nul (population homogène) : standardisation ignorée."
            )

    logger.info(
        "PRS calculé sur %d variants communs pour %d échantillons.",
        len(common_variants),
        prs.shape[0],
    )
    return prs


@dataclass
class FeatureSpec:
    """Spécification des colonnes utilisées pour construire la matrice de features.

    Attributes:
        numeric_clinical_columns: Colonnes cliniques numériques continues
            (ex: "age", "bmi").
        categorical_clinical_columns: Colonnes cliniques catégorielles
            (ex: "sex", "smoking_status").
        variant_columns: Colonnes de dosages génotypiques additifs.
        include_prs: Si True, une colonne "polygenic_risk_score" doit être
            fournie séparément et sera intégrée telle quelle (déjà
            calculée hors du ColumnTransformer pour éviter les fuites,
            car son calcul dépend des tailles d'effet, indépendantes du
            split train/test).
    """

    numeric_clinical_columns: list[str]
    categorical_clinical_columns: list[str]
    variant_columns: list[str]
    include_prs: bool = True


class FeatureBuilder:
    """Construit la matrice de caractéristiques finale sans fuite de données.

    Cette classe encapsule un `sklearn.compose.ColumnTransformer` qui doit
    être ajusté (`fit`) exclusivement sur l'ensemble d'entraînement, puis
    appliqué (`transform`) sur l'ensemble de test, garantissant qu'aucune
    statistique (moyenne, écart-type, catégories) issue du test ne
    contamine l'entraînement.

    Attributes:
        spec: La spécification des colonnes (`FeatureSpec`).
        is_fitted: Indique si le pipeline interne a été ajusté.
    """

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec
        self.is_fitted: bool = False
        self._feature_names: list[str] = []

        numeric_cols = list(spec.numeric_clinical_columns) + list(spec.variant_columns)
        categorical_cols = list(spec.categorical_clinical_columns)

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        self._transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_cols),
                ("categorical", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
        )
        self._numeric_cols = numeric_cols
        self._categorical_cols = categorical_cols

    def fit_transform(self, df: pd.DataFrame, prs: pd.Series | None = None) -> pd.DataFrame:
        """Ajuste le pipeline sur `df` (données d'ENTRAÎNEMENT uniquement) et transforme.

        Args:
            df: DataFrame contenant les colonnes cliniques et de variants
                spécifiées dans `self.spec`, pour les échantillons
                d'entraînement uniquement.
            prs: Série optionnelle de scores de risque polygénique
                pré-calculés (via `compute_polygenic_risk_score`), alignée
                sur l'index de `df`. Requis si `spec.include_prs` est True.

        Returns:
            Un DataFrame de caractéristiques numériques, indexé comme `df`.

        Raises:
            FeatureEngineeringError: Si `include_prs` est True mais `prs`
                n'est pas fourni.
        """
        self._validate_columns(df)
        transformed = self._transformer.fit_transform(df)
        self.is_fitted = True
        self._feature_names = self._build_feature_names()
        feature_df = pd.DataFrame(transformed, columns=self._feature_names, index=df.index)
        feature_df = self._attach_prs(feature_df, prs, df.index)
        logger.info(
            "FeatureBuilder ajusté (fit_transform) : %d échantillons, %d features.",
            feature_df.shape[0],
            feature_df.shape[1],
        )
        return feature_df

    def transform(self, df: pd.DataFrame, prs: pd.Series | None = None) -> pd.DataFrame:
        """Applique le pipeline déjà ajusté à de nouvelles données (ex: test set).

        Args:
            df: DataFrame de nouveaux échantillons (mêmes colonnes que
                lors de `fit_transform`).
            prs: Série optionnelle de PRS pré-calculés pour ces échantillons.

        Returns:
            Un DataFrame de caractéristiques numériques, avec les MÊMES
            colonnes (mêmes statistiques d'imputation/normalisation, mêmes
            catégories one-hot) que celles apprises lors du `fit_transform`.

        Raises:
            FeatureEngineeringError: Si le pipeline n'a pas encore été ajusté.
        """
        if not self.is_fitted:
            raise FeatureEngineeringError(
                "FeatureBuilder.transform() appelé avant fit_transform(). "
                "Ajustez d'abord le pipeline sur l'ensemble d'entraînement."
            )
        self._validate_columns(df)
        transformed = self._transformer.transform(df)
        feature_df = pd.DataFrame(transformed, columns=self._feature_names, index=df.index)
        feature_df = self._attach_prs(feature_df, prs, df.index)
        return feature_df

    def _attach_prs(
        self, feature_df: pd.DataFrame, prs: pd.Series | None, index: pd.Index
    ) -> pd.DataFrame:
        if not self.spec.include_prs:
            return feature_df
        if prs is None:
            raise FeatureEngineeringError(
                "spec.include_prs=True mais aucun PRS n'a été fourni. "
                "Calculez-le au préalable avec compute_polygenic_risk_score()."
            )
        aligned_prs = prs.reindex(index)
        if aligned_prs.isna().any():
            missing = aligned_prs[aligned_prs.isna()].index.tolist()
            raise FeatureEngineeringError(
                f"PRS manquant pour les échantillons : {missing}"
            )
        feature_df = feature_df.copy()
        feature_df["polygenic_risk_score"] = aligned_prs.values
        return feature_df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = set(self._numeric_cols) | set(self._categorical_cols)
        missing = required - set(df.columns)
        if missing:
            raise FeatureEngineeringError(
                f"Colonnes manquantes dans le DataFrame fourni : {sorted(missing)}"
            )

    def _build_feature_names(self) -> list[str]:
        names: list[str] = list(self._numeric_cols)
        cat_transformer = self._transformer.named_transformers_["categorical"]
        encoder: OneHotEncoder = cat_transformer.named_steps["encoder"]
        if self._categorical_cols:
            encoded_names = encoder.get_feature_names_out(self._categorical_cols)
            names.extend(encoded_names.tolist())
        return names

    @property
    def feature_names(self) -> list[str]:
        """Retourne les noms des caractéristiques générées (après `fit_transform`)."""
        if not self.is_fitted:
            raise FeatureEngineeringError("Le pipeline n'a pas encore été ajusté.")
        output = list(self._feature_names)
        if self.spec.include_prs:
            output.append("polygenic_risk_score")
        return output

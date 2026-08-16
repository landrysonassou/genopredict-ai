"""Tests unitaires pour le module `src.features`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features, ingestion


@pytest.fixture
def clean_data(small_variant_csv, small_annotations):
    df = ingestion.load_variant_matrix(small_variant_csv)
    variant_cols = ingestion.identify_variant_columns(
        df, ["sex", "age", "bmi", "smoking_status"]
    )
    clean_df, report = ingestion.quality_control(
        df, variant_cols, call_rate_threshold=0.0, maf_threshold=0.0
    )
    kept_cols = [c for c in variant_cols if c not in report.dropped_variants]
    return clean_df, kept_cols


class TestComputePolygenicRiskScore:
    def test_returns_series_indexed_like_input(self, clean_data, small_annotations) -> None:
        clean_df, kept_cols = clean_data
        effect_sizes = small_annotations.set_index("variant_id")["effect_size"]
        prs = features.compute_polygenic_risk_score(clean_df[kept_cols], effect_sizes)
        assert isinstance(prs, pd.Series)
        assert list(prs.index) == list(clean_df.index)

    def test_standardized_prs_has_zero_mean(self, clean_data, small_annotations) -> None:
        clean_df, kept_cols = clean_data
        effect_sizes = small_annotations.set_index("variant_id")["effect_size"]
        prs = features.compute_polygenic_risk_score(
            clean_df[kept_cols], effect_sizes, standardize=True
        )
        assert abs(prs.mean()) < 1e-8

    def test_no_common_variants_raises(self, clean_data) -> None:
        clean_df, kept_cols = clean_data
        unrelated_effects = pd.Series({"rsUNRELATED": 0.5})
        with pytest.raises(features.FeatureEngineeringError):
            features.compute_polygenic_risk_score(clean_df[kept_cols], unrelated_effects)

    def test_handles_missing_dosages_via_mean_imputation(
        self, clean_data, small_annotations
    ) -> None:
        clean_df, kept_cols = clean_data
        dosage = clean_df[kept_cols].copy()
        dosage.iloc[0, 0] = np.nan
        effect_sizes = small_annotations.set_index("variant_id")["effect_size"]
        prs = features.compute_polygenic_risk_score(dosage, effect_sizes)
        assert not prs.isna().any()


class TestFeatureBuilder:
    def _make_spec(self, kept_cols) -> features.FeatureSpec:
        return features.FeatureSpec(
            numeric_clinical_columns=["age", "bmi"],
            categorical_clinical_columns=["sex", "smoking_status"],
            variant_columns=kept_cols,
            include_prs=True,
        )

    def test_fit_transform_produces_numeric_matrix(
        self, clean_data, small_annotations
    ) -> None:
        clean_df, kept_cols = clean_data
        effect_sizes = small_annotations.set_index("variant_id")["effect_size"]
        prs = features.compute_polygenic_risk_score(clean_df[kept_cols], effect_sizes)

        builder = features.FeatureBuilder(self._make_spec(kept_cols))
        X = builder.fit_transform(clean_df, prs=prs)

        assert X.shape[0] == clean_df.shape[0]
        assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
        assert not X.isna().any().any()
        assert "polygenic_risk_score" in X.columns

    def test_transform_before_fit_raises(self, clean_data) -> None:
        clean_df, kept_cols = clean_data
        builder = features.FeatureBuilder(self._make_spec(kept_cols))
        with pytest.raises(features.FeatureEngineeringError):
            builder.transform(clean_df)

    def test_transform_reuses_fitted_statistics(self, clean_data, small_annotations) -> None:
        clean_df, kept_cols = clean_data
        effect_sizes = small_annotations.set_index("variant_id")["effect_size"]
        prs = features.compute_polygenic_risk_score(clean_df[kept_cols], effect_sizes)

        train_df = clean_df.iloc[:15]
        test_df = clean_df.iloc[15:]

        builder = features.FeatureBuilder(self._make_spec(kept_cols))
        X_train = builder.fit_transform(train_df, prs=prs.loc[train_df.index])
        X_test = builder.transform(test_df, prs=prs.loc[test_df.index])

        assert list(X_train.columns) == list(X_test.columns)
        assert X_test.shape[0] == test_df.shape[0]

    def test_missing_prs_raises_when_required(self, clean_data) -> None:
        clean_df, kept_cols = clean_data
        builder = features.FeatureBuilder(self._make_spec(kept_cols))
        with pytest.raises(features.FeatureEngineeringError):
            builder.fit_transform(clean_df, prs=None)

    def test_missing_columns_raise(self, clean_data) -> None:
        clean_df, kept_cols = clean_data
        builder = features.FeatureBuilder(self._make_spec(kept_cols))
        broken_df = clean_df.drop(columns=["age"])
        with pytest.raises(features.FeatureEngineeringError):
            builder.fit_transform(broken_df, prs=pd.Series(dtype=float))

    def test_feature_names_available_after_fit(self, clean_data, small_annotations) -> None:
        clean_df, kept_cols = clean_data
        effect_sizes = small_annotations.set_index("variant_id")["effect_size"]
        prs = features.compute_polygenic_risk_score(clean_df[kept_cols], effect_sizes)
        builder = features.FeatureBuilder(self._make_spec(kept_cols))
        builder.fit_transform(clean_df, prs=prs)
        assert "polygenic_risk_score" in builder.feature_names
        assert "age" in builder.feature_names

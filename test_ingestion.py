"""Tests unitaires pour le module `src.ingestion`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import ingestion


class TestGenotypeToDosage:
    @pytest.mark.parametrize(
        "genotype,expected",
        [
            ("0/0", 0.0),
            ("0/1", 1.0),
            ("1/0", 1.0),
            ("1/1", 2.0),
            ("0|0", 0.0),
            ("1|1", 2.0),
        ],
    )
    def test_valid_genotypes(self, genotype: str, expected: float) -> None:
        assert ingestion.genotype_to_dosage(genotype) == expected

    @pytest.mark.parametrize("missing", ["./.", ".|.", "", None, np.nan])
    def test_missing_genotypes_return_nan(self, missing) -> None:
        assert np.isnan(ingestion.genotype_to_dosage(missing))

    def test_invalid_genotype_raises(self) -> None:
        with pytest.raises(ingestion.GenotypeParsingError):
            ingestion.genotype_to_dosage("2/2")

    def test_garbage_string_raises(self) -> None:
        with pytest.raises(ingestion.GenotypeParsingError):
            ingestion.genotype_to_dosage("not_a_genotype")


class TestLoadVariantMatrix:
    def test_loads_valid_file(self, small_variant_csv) -> None:
        df = ingestion.load_variant_matrix(small_variant_csv)
        assert df.index.name == "sample_id"
        assert "disease_status" in df.columns
        assert df.shape[0] == 20

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(ingestion.IngestionError):
            ingestion.load_variant_matrix(tmp_path / "does_not_exist.csv")

    def test_missing_required_column_raises(self, tmp_path) -> None:
        df = pd.DataFrame({"sample_id": ["A", "B"], "age": [30, 40]})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ingestion.IngestionError, match="disease_status"):
            ingestion.load_variant_matrix(path)

    def test_duplicate_sample_ids_raise(self, tmp_path) -> None:
        df = pd.DataFrame(
            {
                "sample_id": ["A", "A"],
                "disease_status": [0, 1],
            }
        )
        path = tmp_path / "dup.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ingestion.IngestionError, match="dupliqué"):
            ingestion.load_variant_matrix(path)

    def test_non_binary_label_raises(self, tmp_path) -> None:
        df = pd.DataFrame(
            {
                "sample_id": ["A", "B"],
                "disease_status": [0, 2],
            }
        )
        path = tmp_path / "nonbinary.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ingestion.IngestionError, match="binaire"):
            ingestion.load_variant_matrix(path)

    def test_empty_file_raises(self, tmp_path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(ingestion.IngestionError):
            ingestion.load_variant_matrix(path)


class TestLoadAnnotations:
    def test_loads_valid_file(self, small_annotations_csv) -> None:
        df = ingestion.load_annotations(small_annotations_csv)
        assert df.index.name == "variant_id"
        assert "effect_size" in df.columns
        assert df.shape[0] == 6

    def test_missing_columns_raise(self, tmp_path) -> None:
        df = pd.DataFrame({"variant_id": ["rs1"], "gene": ["APOE"]})
        path = tmp_path / "bad_annotations.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ingestion.IngestionError):
            ingestion.load_annotations(path)


class TestIdentifyVariantColumns:
    def test_identifies_correct_columns(self, small_variant_csv) -> None:
        df = ingestion.load_variant_matrix(small_variant_csv)
        clinical_cols = ["sex", "age", "bmi", "smoking_status"]
        variant_cols = ingestion.identify_variant_columns(df, clinical_cols)
        assert len(variant_cols) == 6
        assert all(v.startswith("rsTEST") for v in variant_cols)

    def test_raises_when_no_variant_columns(self) -> None:
        df = pd.DataFrame(
            {"sex": ["F"], "age": [30], "disease_status": [0]}
        ).set_index(pd.Index(["S1"], name="sample_id"))
        with pytest.raises(ingestion.IngestionError):
            ingestion.identify_variant_columns(df, ["sex", "age"])


class TestQualityControl:
    def test_qc_returns_numeric_dosages(self, small_variant_csv) -> None:
        df = ingestion.load_variant_matrix(small_variant_csv)
        variant_cols = ingestion.identify_variant_columns(
            df, ["sex", "age", "bmi", "smoking_status"]
        )
        clean_df, report = ingestion.quality_control(
            df, variant_cols, call_rate_threshold=0.0, maf_threshold=0.0
        )
        for variant in report.variant_call_rates:
            if variant not in report.dropped_variants:
                assert pd.api.types.is_numeric_dtype(clean_df[variant])

    def test_qc_drops_monomorphic_variant(self, small_variant_matrix) -> None:
        # Rendre un variant totalement monomorphe (MAF = 0)
        df = small_variant_matrix.copy()
        df["rsTEST0"] = "0/0"
        df = df.set_index("sample_id")
        variant_cols = ["rsTEST0", "rsTEST1", "rsTEST2", "rsTEST3", "rsTEST4", "rsTEST5"]
        _, report = ingestion.quality_control(
            df, variant_cols, call_rate_threshold=0.0, maf_threshold=0.01
        )
        assert "rsTEST0" in report.dropped_variants

    def test_qc_drops_low_call_rate_variant(self, small_variant_matrix) -> None:
        df = small_variant_matrix.copy()
        df.loc[df.index[:15], "rsTEST1"] = "./."  # 75% manquant sur 20 échantillons
        df = df.set_index("sample_id")
        variant_cols = ["rsTEST0", "rsTEST1", "rsTEST2", "rsTEST3", "rsTEST4", "rsTEST5"]
        _, report = ingestion.quality_control(
            df, variant_cols, call_rate_threshold=0.9, maf_threshold=0.0
        )
        assert "rsTEST1" in report.dropped_variants

    def test_qc_invalid_threshold_raises(self, small_variant_matrix) -> None:
        df = small_variant_matrix.set_index("sample_id")
        variant_cols = ["rsTEST0"]
        with pytest.raises(ingestion.IngestionError):
            ingestion.quality_control(df, variant_cols, call_rate_threshold=1.5)

    def test_qc_report_summary_is_string(self, small_variant_csv) -> None:
        df = ingestion.load_variant_matrix(small_variant_csv)
        variant_cols = ingestion.identify_variant_columns(
            df, ["sex", "age", "bmi", "smoking_status"]
        )
        _, report = ingestion.quality_control(df, variant_cols)
        assert isinstance(report.summary(), str)
        assert "Rapport de Contrôle Qualité" in report.summary()

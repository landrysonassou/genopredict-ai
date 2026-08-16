"""Tests unitaires pour le module `src.treatment`."""

from __future__ import annotations

import pytest

from src import treatment
from src.models import RiskCategory


class TestRecommendCare:
    def test_returns_correct_risk_category(self) -> None:
        profile = treatment.recommend_care("S1", risk_probability=0.9, carried_genes=[])
        assert profile.risk_category == RiskCategory.VERY_HIGH

    def test_low_risk_has_minimal_recommendations(self) -> None:
        profile = treatment.recommend_care("S1", risk_probability=0.05, carried_genes=[])
        assert profile.risk_category == RiskCategory.LOW
        assert len(profile.preventive_recommendations) > 0

    def test_known_gene_triggers_pharmacogenomic_alert(self) -> None:
        profile = treatment.recommend_care("S1", risk_probability=0.5, carried_genes=["BRCA1"])
        assert len(profile.pharmacogenomic_alerts) == 1
        assert profile.pharmacogenomic_alerts[0].gene == "BRCA1"

    def test_gene_matching_is_case_insensitive(self) -> None:
        profile = treatment.recommend_care("S1", risk_probability=0.5, carried_genes=["brca1"])
        assert len(profile.pharmacogenomic_alerts) == 1

    def test_unknown_gene_triggers_no_alert(self) -> None:
        profile = treatment.recommend_care(
            "S1", risk_probability=0.5, carried_genes=["NOT_A_REAL_GENE"]
        )
        assert profile.pharmacogenomic_alerts == []

    def test_multiple_genes_accumulate_alerts(self) -> None:
        profile = treatment.recommend_care(
            "S1", risk_probability=0.5, carried_genes=["BRCA1", "TPMT", "VKORC1"]
        )
        genes_found = {alert.gene for alert in profile.pharmacogenomic_alerts}
        assert genes_found == {"BRCA1", "TPMT", "VKORC1"}

    def test_out_of_range_probability_raises(self) -> None:
        with pytest.raises(ValueError):
            treatment.recommend_care("S1", risk_probability=1.5, carried_genes=[])

    def test_disclaimer_always_present(self) -> None:
        profile = treatment.recommend_care("S1", risk_probability=0.5, carried_genes=[])
        assert "recherche" in profile.disclaimer.lower()
        assert len(profile.disclaimer) > 20

    def test_to_dict_is_json_serializable_shape(self) -> None:
        import json

        profile = treatment.recommend_care(
            "S1", risk_probability=0.8, carried_genes=["BRCA1"]
        )
        payload = profile.to_dict()
        serialized = json.dumps(payload)  # ne doit pas lever d'exception
        assert "BRCA1" in serialized
        assert payload["risk_category"] == "très élevé"

    def test_custom_knowledge_base_is_used(self) -> None:
        custom_kb = {
            "MYGENE": [
                treatment.PharmacogenomicEntry(
                    gene="MYGENE",
                    drug_class="test_class",
                    guidance="test guidance",
                    cpic_level="B",
                )
            ]
        }
        profile = treatment.recommend_care(
            "S1", risk_probability=0.5, carried_genes=["MYGENE"], knowledge_base=custom_kb
        )
        assert len(profile.pharmacogenomic_alerts) == 1
        assert profile.pharmacogenomic_alerts[0].drug_class == "test_class"

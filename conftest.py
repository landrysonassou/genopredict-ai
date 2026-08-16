"""Fixtures pytest partagées pour l'ensemble de la suite de tests GenoPredict-AI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def rng() -> np.random.Generator:
    """Générateur aléatoire déterministe pour des tests reproductibles."""
    return np.random.default_rng(123)


@pytest.fixture
def small_variant_matrix(rng: np.random.Generator) -> pd.DataFrame:
    """Petite matrice de variants synthétique (20 échantillons, 6 variants)."""
    n_samples = 20
    variant_ids = [f"rsTEST{i}" for i in range(6)]

    data = {
        "sample_id": [f"S{i:03d}" for i in range(n_samples)],
        "sex": rng.choice(["F", "M"], size=n_samples),
        "age": rng.normal(50, 10, size=n_samples).round(1),
        "bmi": rng.normal(26, 3, size=n_samples).round(1),
        "smoking_status": rng.choice(["never", "former", "current"], size=n_samples),
    }
    genotype_options = ["0/0", "0/1", "1/1"]
    for variant_id in variant_ids:
        data[variant_id] = rng.choice(genotype_options, size=n_samples, p=[0.6, 0.3, 0.1])
    data["disease_status"] = rng.integers(0, 2, size=n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def small_variant_csv(tmp_path: Path, small_variant_matrix: pd.DataFrame) -> Path:
    """Écrit la petite matrice de variants sur disque et retourne son chemin."""
    path = tmp_path / "variant_matrix.csv"
    small_variant_matrix.to_csv(path, index=False)
    return path


@pytest.fixture
def small_annotations(rng: np.random.Generator) -> pd.DataFrame:
    """Petit fichier d'annotations correspondant à `small_variant_matrix`."""
    variant_ids = [f"rsTEST{i}" for i in range(6)]
    return pd.DataFrame(
        {
            "variant_id": variant_ids,
            "gene": ["APOE", "BRCA1", "CYP2D6", "TPMT", "VKORC1", "MTHFR"],
            "consequence": ["missense_variant"] * 6,
            "cadd_score": rng.uniform(1, 30, size=6).round(2),
            "effect_allele_freq": rng.uniform(0.05, 0.4, size=6).round(3),
            "effect_size": rng.normal(0, 0.3, size=6).round(3),
        }
    )


@pytest.fixture
def small_annotations_csv(tmp_path: Path, small_annotations: pd.DataFrame) -> Path:
    """Écrit le fichier d'annotations sur disque et retourne son chemin."""
    path = tmp_path / "annotations.csv"
    small_annotations.to_csv(path, index=False)
    return path

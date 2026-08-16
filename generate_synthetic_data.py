"""Génère un jeu de données synthétique de variants génétiques et d'annotations.

Ce script produit des données ENTIÈREMENT SIMULÉES (aucune donnée
patient réelle) pour permettre l'exécution de bout en bout du pipeline
GenoPredict-AI à des fins de démonstration, de test et de développement.

Usage:
    python data/generate_synthetic_data.py --n-samples 600 --n-variants 40 \
        --output-dir data/

Fichiers produits :
    - variant_matrix.csv : matrice échantillons x (clinique + génotypes) + label
    - annotations.csv    : annotations fonctionnelles par variant
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE_GENES = [
    "APOE", "BRCA1", "BRCA2", "CYP2C19", "CYP2D6", "TPMT", "VKORC1",
    "MTHFR", "FTO", "UCP2", "TCF7L2", "PCSK9", "LDLR", "HFE",
]
CONSEQUENCES = ["missense_variant", "synonymous_variant", "intron_variant", "stop_gained"]


def _simulate_genotype(maf: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Simule des génotypes sous équilibre de Hardy-Weinberg pour une MAF donnée."""
    p_ref, p_het, p_alt = (1 - maf) ** 2, 2 * maf * (1 - maf), maf**2
    return rng.choice(["0/0", "0/1", "1/1"], size=size, p=[p_ref, p_het, p_alt])


def generate_dataset(
    n_samples: int = 600,
    n_variants: int = 40,
    missing_rate: float = 0.02,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Génère la matrice de variants et le fichier d'annotations simulés.

    Args:
        n_samples: Nombre d'échantillons (individus) à simuler.
        n_variants: Nombre de variants génétiques à simuler.
        missing_rate: Taux de génotypes manquants injectés aléatoirement.
        random_state: Graine aléatoire pour la reproductibilité.

    Returns:
        Un tuple (variant_matrix_df, annotations_df).
    """
    rng = np.random.default_rng(random_state)

    variant_ids = [f"rs{rng.integers(1_000_00, 9_999_999)}" for _ in range(n_variants)]
    mafs = rng.uniform(0.02, 0.45, size=n_variants)
    effect_sizes = rng.normal(loc=0.0, scale=0.35, size=n_variants)
    genes = rng.choice(CANDIDATE_GENES, size=n_variants)
    consequences = rng.choice(
        CONSEQUENCES, size=n_variants, p=[0.4, 0.3, 0.25, 0.05]
    )
    cadd_scores = rng.gamma(shape=2.0, scale=4.0, size=n_variants).clip(0, 40)

    annotations_df = pd.DataFrame(
        {
            "variant_id": variant_ids,
            "gene": genes,
            "consequence": consequences,
            "cadd_score": np.round(cadd_scores, 2),
            "effect_allele_freq": np.round(mafs, 4),
            "effect_size": np.round(effect_sizes, 4),
        }
    )

    genotype_columns: dict[str, np.ndarray] = {}
    dosage_matrix = np.zeros((n_samples, n_variants))
    for j, (variant_id, maf) in enumerate(zip(variant_ids, mafs)):
        genotypes = _simulate_genotype(maf, n_samples, rng)
        dosage = np.array([{"0/0": 0, "0/1": 1, "1/1": 2}[g] for g in genotypes], dtype=float)
        # Injection de valeurs manquantes aléatoires
        missing_mask = rng.random(n_samples) < missing_rate
        genotypes = genotypes.astype(object)
        genotypes[missing_mask] = "./."
        dosage[missing_mask] = np.nan
        genotype_columns[variant_id] = genotypes
        dosage_matrix[:, j] = dosage

    sample_ids = [f"SAMPLE_{i:04d}" for i in range(1, n_samples + 1)]
    age = rng.normal(loc=45, scale=15, size=n_samples).clip(18, 90).round(1)
    bmi = rng.normal(loc=25, scale=4.5, size=n_samples).clip(15, 45).round(1)
    sex = rng.choice(["F", "M"], size=n_samples)
    smoking_status = rng.choice(
        ["never", "former", "current"], size=n_samples, p=[0.55, 0.25, 0.20]
    )

    # Le statut de la maladie dépend (de manière bruitée) du PRS génétique
    # simulé et de facteurs cliniques, afin que le pipeline ML apprenne un
    # signal réaliste plutôt que du pur bruit.
    dosage_filled = np.nan_to_num(dosage_matrix, nan=0.0)
    genetic_liability = dosage_filled @ effect_sizes
    clinical_liability = (
        0.02 * (age - 45) + 0.05 * (bmi - 25) + (smoking_status == "current") * 0.6
    )
    liability = genetic_liability + clinical_liability + rng.normal(0, 1.0, size=n_samples)
    threshold = np.quantile(liability, 0.70)  # ~30% de cas positifs
    disease_status = (liability > threshold).astype(int)

    data = {
        "sample_id": sample_ids,
        "sex": sex,
        "age": age,
        "bmi": bmi,
        "smoking_status": smoking_status,
        **genotype_columns,
        "disease_status": disease_status,
    }
    variant_matrix_df = pd.DataFrame(data)
    return variant_matrix_df, annotations_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--n-variants", type=int, default=40)
    parser.add_argument("--missing-rate", type=float, default=0.02)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    variant_matrix_df, annotations_df = generate_dataset(
        n_samples=args.n_samples,
        n_variants=args.n_variants,
        missing_rate=args.missing_rate,
        random_state=args.random_state,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_matrix_path = output_dir / "variant_matrix.csv"
    annotations_path = output_dir / "annotations.csv"
    variant_matrix_df.to_csv(variant_matrix_path, index=False)
    annotations_df.to_csv(annotations_path, index=False)

    print(f"Matrice de variants simulée : {variant_matrix_path} "
          f"({variant_matrix_df.shape[0]} échantillons, {variant_matrix_df.shape[1]} colonnes)")
    print(f"Annotations simulées        : {annotations_path} "
          f"({annotations_df.shape[0]} variants)")


if __name__ == "__main__":
    main()

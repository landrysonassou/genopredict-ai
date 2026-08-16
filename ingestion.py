"""Module d'ingestion et de contrôle qualité (QC) des données génomiques.

Ce module fournit des fonctions robustes pour charger des matrices de
variants génétiques (format tabulaire dérivé de VCF), charger les
annotations fonctionnelles associées, et appliquer un contrôle qualité
rigoureux avant toute étape de modélisation.

Format attendu de la matrice de variants (CSV) :
    sample_id, sex, age, bmi, smoking_status, rsXXXX, rsYYYY, ..., disease_status
    où chaque colonne de variant contient un génotype au format VCF
    simplifié : "0/0" (homozygote référence), "0/1" (hétérozygote),
    "1/1" (homozygote alternatif) ou "./." (donnée manquante).

Format attendu du fichier d'annotations (CSV) :
    variant_id, gene, consequence, cadd_score, effect_allele_freq, effect_size
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Génotypes VCF simplifiés valides et leur dosage additif (nombre d'allèles alternatifs)
VALID_GENOTYPES: dict[str, float] = {
    "0/0": 0.0,
    "0|0": 0.0,
    "0/1": 1.0,
    "1/0": 1.0,
    "0|1": 1.0,
    "1|0": 1.0,
    "1/1": 2.0,
    "1|1": 2.0,
}
MISSING_GENOTYPES: frozenset[str] = frozenset({"./.", ".|.", "", "NA", "nan", "NaN"})

REQUIRED_SAMPLE_COLUMN = "sample_id"
REQUIRED_LABEL_COLUMN = "disease_status"


class IngestionError(Exception):
    """Erreur levée lorsqu'un fichier d'entrée est structurellement invalide."""


class GenotypeParsingError(Exception):
    """Erreur levée lorsqu'une valeur de génotype ne peut pas être interprétée."""


@dataclass
class QCReport:
    """Rapport structuré de contrôle qualité.

    Attributes:
        n_samples_input: Nombre d'échantillons avant QC.
        n_samples_output: Nombre d'échantillons après QC.
        n_variants_input: Nombre de variants avant QC.
        n_variants_output: Nombre de variants après QC.
        dropped_samples: Liste des identifiants d'échantillons exclus.
        dropped_variants: Liste des variants exclus avec la raison.
        variant_call_rates: Taux d'appel (1 - taux de données manquantes) par variant.
        variant_maf: Fréquence de l'allèle mineur (MAF) estimée par variant.
    """

    n_samples_input: int
    n_samples_output: int
    n_variants_input: int
    n_variants_output: int
    dropped_samples: list[str] = field(default_factory=list)
    dropped_variants: dict[str, str] = field(default_factory=dict)
    variant_call_rates: dict[str, float] = field(default_factory=dict)
    variant_maf: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Retourne un résumé lisible du rapport de QC."""
        lines = [
            "=== Rapport de Contrôle Qualité (QC) ===",
            f"Échantillons : {self.n_samples_input} -> {self.n_samples_output} "
            f"({len(self.dropped_samples)} exclus)",
            f"Variants     : {self.n_variants_input} -> {self.n_variants_output} "
            f"({len(self.dropped_variants)} exclus)",
        ]
        if self.dropped_variants:
            lines.append("Variants exclus :")
            for variant, reason in self.dropped_variants.items():
                lines.append(f"  - {variant}: {reason}")
        return "\n".join(lines)


def genotype_to_dosage(genotype: str) -> float:
    """Convertit un génotype VCF simplifié en dosage allélique additif.

    Le dosage additif compte le nombre d'allèles alternatifs (0, 1 ou 2),
    convention standard pour l'encodage des variants en apprentissage
    automatique génomique.

    Args:
        genotype: Chaîne de génotype (ex: "0/0", "0/1", "1/1", "./.").

    Returns:
        Le dosage allélique (0.0, 1.0 ou 2.0), ou np.nan si la donnée
        est manquante.

    Raises:
        GenotypeParsingError: Si la chaîne ne correspond à aucun format
            de génotype reconnu (biallélique, diploïde).
    """
    if genotype is None or (isinstance(genotype, float) and np.isnan(genotype)):
        return np.nan
    genotype_str = str(genotype).strip()
    if genotype_str in MISSING_GENOTYPES:
        return np.nan
    if genotype_str in VALID_GENOTYPES:
        return VALID_GENOTYPES[genotype_str]
    raise GenotypeParsingError(
        f"Génotype non reconnu : '{genotype}'. Formats valides : "
        f"{sorted(VALID_GENOTYPES.keys())} ou manquant ({sorted(MISSING_GENOTYPES)})."
    )


def load_variant_matrix(
    filepath: str | Path,
    sample_id_col: str = REQUIRED_SAMPLE_COLUMN,
    label_col: str = REQUIRED_LABEL_COLUMN,
) -> pd.DataFrame:
    """Charge et valide une matrice de variants génétiques au format CSV.

    Args:
        filepath: Chemin vers le fichier CSV de la matrice de variants.
        sample_id_col: Nom de la colonne identifiant chaque échantillon.
        label_col: Nom de la colonne cible (statut de la maladie, 0/1).

    Returns:
        Un DataFrame pandas indexé par `sample_id_col`, contenant les
        colonnes cliniques, les colonnes de variants (génotypes bruts)
        et la colonne cible.

    Raises:
        IngestionError: Si le fichier est introuvable, vide, ou si les
            colonnes obligatoires sont absentes, ou si des identifiants
            d'échantillons sont dupliqués.
    """
    path = Path(filepath)
    if not path.exists():
        raise IngestionError(f"Fichier introuvable : {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise IngestionError(f"Le fichier '{path}' est vide.") from exc
    except pd.errors.ParserError as exc:
        raise IngestionError(f"Erreur de parsing CSV pour '{path}' : {exc}") from exc

    if df.empty:
        raise IngestionError(f"Le fichier '{path}' ne contient aucune ligne.")

    missing_cols = {sample_id_col, label_col} - set(df.columns)
    if missing_cols:
        raise IngestionError(
            f"Colonnes obligatoires manquantes dans '{path}' : {sorted(missing_cols)}"
        )

    if df[sample_id_col].duplicated().any():
        duplicates = df.loc[df[sample_id_col].duplicated(), sample_id_col].tolist()
        raise IngestionError(
            f"Identifiants d'échantillons dupliqués détectés : {duplicates}"
        )

    unique_labels = set(df[label_col].dropna().unique().tolist())
    if not unique_labels.issubset({0, 1, 0.0, 1.0}):
        raise IngestionError(
            f"La colonne cible '{label_col}' doit être binaire (0/1). "
            f"Valeurs trouvées : {sorted(unique_labels)}"
        )

    df = df.set_index(sample_id_col)
    logger.info(
        "Matrice de variants chargée : %d échantillons, %d colonnes.",
        df.shape[0],
        df.shape[1],
    )
    return df


def load_annotations(filepath: str | Path) -> pd.DataFrame:
    """Charge le fichier d'annotations fonctionnelles des variants.

    Args:
        filepath: Chemin vers le fichier CSV d'annotations.

    Returns:
        Un DataFrame indexé par `variant_id`, contenant au minimum les
        colonnes `gene`, `consequence`, `cadd_score`, `effect_allele_freq`
        et `effect_size`.

    Raises:
        IngestionError: Si le fichier est introuvable ou si des colonnes
            obligatoires sont absentes.
    """
    path = Path(filepath)
    if not path.exists():
        raise IngestionError(f"Fichier d'annotations introuvable : {path}")

    df = pd.read_csv(path)
    required = {"variant_id", "gene", "consequence", "cadd_score", "effect_size"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise IngestionError(
            f"Colonnes obligatoires manquantes dans le fichier d'annotations : "
            f"{sorted(missing_cols)}"
        )
    df = df.set_index("variant_id")
    logger.info("Annotations chargées pour %d variants.", df.shape[0])
    return df


def identify_variant_columns(
    df: pd.DataFrame, clinical_columns: list[str], label_col: str = REQUIRED_LABEL_COLUMN
) -> list[str]:
    """Identifie les colonnes de variants génétiques dans la matrice.

    Toute colonne qui n'est ni une colonne clinique déclarée, ni la
    colonne cible, est considérée comme un variant génétique.

    Args:
        df: Matrice de variants chargée via `load_variant_matrix`.
        clinical_columns: Liste des noms de colonnes cliniques/phénotypiques
            (ex: ["sex", "age", "bmi", "smoking_status"]).
        label_col: Nom de la colonne cible.

    Returns:
        La liste ordonnée des noms de colonnes de variants.
    """
    excluded = set(clinical_columns) | {label_col}
    variant_cols = [col for col in df.columns if col not in excluded]
    if not variant_cols:
        raise IngestionError(
            "Aucune colonne de variant identifiée. Vérifiez `clinical_columns`."
        )
    return variant_cols


def quality_control(
    df: pd.DataFrame,
    variant_columns: list[str],
    call_rate_threshold: float = 0.90,
    maf_threshold: float = 0.01,
    sample_missing_threshold: float = 0.10,
) -> tuple[pd.DataFrame, QCReport]:
    """Applique un contrôle qualité (QC) génotypique et phénotypique.

    Étapes appliquées, dans l'ordre :
      1. Calcul du taux d'appel (call rate) par variant ; exclusion des
         variants dont le taux d'appel est inférieur à `call_rate_threshold`.
      2. Calcul de la fréquence de l'allèle mineur (MAF) par variant ;
         exclusion des variants monomorphes ou trop rares
         (MAF < `maf_threshold`).
      3. Exclusion des échantillons dont le taux de données manquantes
         (sur les variants conservés) dépasse `sample_missing_threshold`.

    Args:
        df: Matrice de variants (génotypes bruts en chaînes de caractères).
        variant_columns: Colonnes correspondant aux variants génétiques.
        call_rate_threshold: Taux d'appel minimal requis pour conserver
            un variant (entre 0 et 1).
        maf_threshold: MAF minimale requise pour conserver un variant.
        sample_missing_threshold: Proportion maximale de génotypes
            manquants tolérée par échantillon (entre 0 et 1).

    Returns:
        Un tuple (DataFrame nettoyé avec dosages additifs numériques
        pour les variants conservés, QCReport détaillant les exclusions).

    Raises:
        IngestionError: Si les seuils fournis sont hors de l'intervalle [0, 1].
    """
    for name, value in (
        ("call_rate_threshold", call_rate_threshold),
        ("maf_threshold", maf_threshold),
        ("sample_missing_threshold", sample_missing_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise IngestionError(f"'{name}' doit être compris entre 0 et 1 (reçu {value}).")

    n_samples_input = df.shape[0]
    n_variants_input = len(variant_columns)

    # Étape 1 : décodage des génotypes en dosages additifs (0/1/2/NaN)
    dosage_df = df[variant_columns].apply(
        lambda col: col.map(genotype_to_dosage), axis=0
    )

    dropped_variants: dict[str, str] = {}
    call_rates: dict[str, float] = {}
    mafs: dict[str, float] = {}
    kept_variants: list[str] = []

    for variant in variant_columns:
        series = dosage_df[variant]
        call_rate = 1.0 - series.isna().mean()
        call_rates[variant] = round(float(call_rate), 4)

        if call_rate < call_rate_threshold:
            dropped_variants[variant] = (
                f"call_rate={call_rate:.3f} < seuil {call_rate_threshold}"
            )
            continue

        # MAF estimée à partir du dosage additif moyen (fréquence allélique)
        observed = series.dropna()
        maf = float(observed.mean() / 2.0) if len(observed) > 0 else 0.0
        maf = min(maf, 1.0 - maf)  # ramener à la fréquence de l'allèle MINEUR
        mafs[variant] = round(maf, 4)

        if maf < maf_threshold:
            dropped_variants[variant] = f"MAF={maf:.4f} < seuil {maf_threshold}"
            continue

        kept_variants.append(variant)

    clean_dosage = dosage_df[kept_variants].copy()

    # Étape 2 : exclusion des échantillons trop incomplets
    per_sample_missing_rate = clean_dosage.isna().mean(axis=1)
    samples_to_drop = per_sample_missing_rate[
        per_sample_missing_rate > sample_missing_threshold
    ].index.tolist()

    other_columns = [c for c in df.columns if c not in variant_columns]
    clean_df = pd.concat([df[other_columns], clean_dosage], axis=1)
    clean_df = clean_df.drop(index=samples_to_drop)

    report = QCReport(
        n_samples_input=n_samples_input,
        n_samples_output=clean_df.shape[0],
        n_variants_input=n_variants_input,
        n_variants_output=len(kept_variants),
        dropped_samples=[str(s) for s in samples_to_drop],
        dropped_variants=dropped_variants,
        variant_call_rates=call_rates,
        variant_maf=mafs,
    )
    logger.info(report.summary())
    return clean_df, report

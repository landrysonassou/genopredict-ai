"""Module de médecine personnalisée : recommandations préventives et pharmacogénomiques.

Ce module croise le profil de risque génétique prédit (module `models`)
avec une base de connaissances simplifiée de gènes pharmacogénomiques
d'intérêt clinique reconnu (inspirée des catégories publiques du
consortium CPIC — Clinical Pharmacogenetics Implementation Consortium)
pour générer des recommandations de prise en charge préventive.

AVERTISSEMENT IMPORTANT :
Ce module est un outil de démonstration pédagogique et de recherche.
Il ne constitue EN AUCUN CAS un dispositif médical, un outil de
diagnostic clinique, ni une source de posologie. Toute décision
thérapeutique doit être validée par un professionnel de santé qualifié
sur la base d'un test génétique certifié et d'un examen clinique complet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from models import RiskCategory

logger = logging.getLogger(__name__)

CLINICAL_DISCLAIMER = (
    "Ce rapport est généré à des fins de recherche et de démonstration "
    "technique uniquement. Il ne remplace pas un avis médical, un test "
    "génétique clinique certifié, ni une consultation avec un "
    "professionnel de santé qualifié (généticien clinicien, médecin "
    "traitant, pharmacologue). Aucune décision thérapeutique ne doit être "
    "prise sur la seule base de ce résultat."
)


class TreatmentEngineError(Exception):
    """Erreur levée lors d'une configuration invalide du moteur de recommandation."""


@dataclass(frozen=True)
class PharmacogenomicEntry:
    """Entrée de la base de connaissances pharmacogénomique simplifiée.

    Attributes:
        gene: Symbole du gène (nomenclature HGNC).
        drug_class: Classe thérapeutique concernée par une interaction
            gène-médicament connue.
        guidance: Recommandation générale (niveau catégoriel, non
            posologique) issue de la littérature pharmacogénomique publique.
        cpic_level: Niveau de preuve indicatif (A = preuve forte, B = preuve modérée).
    """

    gene: str
    drug_class: str
    guidance: str
    cpic_level: str


# Base de connaissances simplifiée, à titre illustratif uniquement.
# Références générales : lignes directrices publiques du CPIC (PharmGKB).
PHARMACOGENOMIC_KB: dict[str, list[PharmacogenomicEntry]] = {
    "CYP2C19": [
        PharmacogenomicEntry(
            gene="CYP2C19",
            drug_class="inhibiteurs de la pompe à protons / clopidogrel",
            guidance=(
                "Une variation d'activité de CYP2C19 peut modifier le métabolisme "
                "de certains médicaments courants ; un test pharmacogénétique "
                "préalable est recommandé avant toute prescription concernée."
            ),
            cpic_level="A",
        )
    ],
    "CYP2D6": [
        PharmacogenomicEntry(
            gene="CYP2D6",
            drug_class="antidépresseurs / antalgiques opioïdes",
            guidance=(
                "CYP2D6 influence fortement le métabolisme de nombreux "
                "psychotropes et antalgiques ; une adaptation posologique "
                "individualisée par un pharmacologue clinique est recommandée."
            ),
            cpic_level="A",
        )
    ],
    "TPMT": [
        PharmacogenomicEntry(
            gene="TPMT",
            drug_class="thiopurines (azathioprine, mercaptopurine)",
            guidance=(
                "Un déficit d'activité TPMT expose à une toxicité hématologique "
                "accrue sous thiopurines ; un dosage enzymatique ou génotypique "
                "préalable est fortement recommandé avant traitement."
            ),
            cpic_level="A",
        )
    ],
    "VKORC1": [
        PharmacogenomicEntry(
            gene="VKORC1",
            drug_class="anticoagulants antivitamine K (warfarine)",
            guidance=(
                "Les variants de VKORC1 influencent la sensibilité aux "
                "anticoagulants antivitamine K ; un suivi INR rapproché en "
                "début de traitement est recommandé."
            ),
            cpic_level="A",
        )
    ],
    "APOE": [
        PharmacogenomicEntry(
            gene="APOE",
            drug_class="statines / prise en charge cardiovasculaire",
            guidance=(
                "Le génotype APOE est associé au risque cardiovasculaire et "
                "neurodégénératif ; une prise en charge préventive rapprochée "
                "(bilan lipidique, suivi cognitif) est conseillée."
            ),
            cpic_level="B",
        )
    ],
    "BRCA1": [
        PharmacogenomicEntry(
            gene="BRCA1",
            drug_class="oncologie / dépistage préventif",
            guidance=(
                "Les variants pathogènes de BRCA1 justifient une orientation "
                "prioritaire vers une consultation d'oncogénétique pour "
                "discuter d'un dépistage renforcé et des options préventives."
            ),
            cpic_level="A",
        )
    ],
}

PREVENTIVE_ACTIONS_BY_RISK: dict[RiskCategory, list[str]] = {
    RiskCategory.LOW: [
        "Maintenir un suivi médical de routine standard.",
        "Poursuivre les mesures générales d'hygiène de vie (activité "
        "physique régulière, alimentation équilibrée).",
    ],
    RiskCategory.MODERATE: [
        "Renforcer la fréquence des bilans de dépistage de routine.",
        "Discuter des antécédents familiaux détaillés avec un professionnel de santé.",
        "Réévaluer les facteurs de risque modifiables (tabac, activité physique, alimentation).",
    ],
    RiskCategory.HIGH: [
        "Orienter vers une consultation de conseil génétique spécialisée.",
        "Mettre en place un calendrier de dépistage rapproché et personnalisé.",
        "Discuter des options de prévention primaire disponibles avec un spécialiste.",
    ],
    RiskCategory.VERY_HIGH: [
        "Orientation prioritaire vers une consultation multidisciplinaire "
        "(génétique clinique et spécialité concernée).",
        "Envisager une confirmation par test génétique clinique certifié "
        "(le présent résultat est un score de recherche, non diagnostique).",
        "Établir un plan de surveillance intensifié en concertation avec l'équipe soignante.",
    ],
}


@dataclass
class TherapeuticProfile:
    """Profil de recommandations préventives et pharmacogénomiques d'un patient.

    Attributes:
        sample_id: Identifiant de l'échantillon/patient.
        risk_probability: Probabilité de pathogénicité prédite par le modèle.
        risk_category: Catégorie de risque clinique associée.
        preventive_recommendations: Recommandations de prise en charge préventive.
        pharmacogenomic_alerts: Alertes pharmacogénomiques déclenchées par les
            gènes portant un variant chez ce patient.
        disclaimer: Avertissement clinique obligatoire.
    """

    sample_id: str
    risk_probability: float
    risk_category: RiskCategory
    preventive_recommendations: list[str] = field(default_factory=list)
    pharmacogenomic_alerts: list[PharmacogenomicEntry] = field(default_factory=list)
    disclaimer: str = CLINICAL_DISCLAIMER

    def to_dict(self) -> dict:
        """Sérialise le profil en dictionnaire simple (JSON-compatible)."""
        return {
            "sample_id": self.sample_id,
            "risk_probability": round(self.risk_probability, 4),
            "risk_category": self.risk_category.value,
            "preventive_recommendations": self.preventive_recommendations,
            "pharmacogenomic_alerts": [
                {
                    "gene": entry.gene,
                    "drug_class": entry.drug_class,
                    "guidance": entry.guidance,
                    "cpic_level": entry.cpic_level,
                }
                for entry in self.pharmacogenomic_alerts
            ],
            "disclaimer": self.disclaimer,
        }


def recommend_care(
    sample_id: str,
    risk_probability: float,
    carried_genes: list[str],
    knowledge_base: dict[str, list[PharmacogenomicEntry]] | None = None,
) -> TherapeuticProfile:
    """Génère un profil de recommandations préventives et pharmacogénomiques.

    Args:
        sample_id: Identifiant de l'échantillon/patient concerné.
        risk_probability: Probabilité de pathogénicité prédite (issue de
            `models.predict_risk`), comprise entre 0 et 1.
        carried_genes: Liste des symboles de gènes pour lesquels
            l'échantillon porte au moins un allèle à risque (dosage > 0).
            Typiquement dérivée des annotations de variants (`ingestion.load_annotations`)
            croisées avec les dosages génotypiques du patient.
        knowledge_base: Base de connaissances pharmacogénomique à utiliser
            (par défaut `PHARMACOGENOMIC_KB`). Paramétrable pour faciliter
            les tests unitaires et les extensions futures.

    Returns:
        Un `TherapeuticProfile` complet, prêt à être affiché dans le
        tableau de bord ou exporté.

    Raises:
        ValueError: Si `risk_probability` n'est pas compris entre 0 et 1.
    """
    if not 0.0 <= risk_probability <= 1.0:
        raise ValueError(
            f"risk_probability doit être compris entre 0 et 1 (reçu {risk_probability})."
        )

    kb = knowledge_base if knowledge_base is not None else PHARMACOGENOMIC_KB
    risk_category = RiskCategory.from_probability(risk_probability)

    preventive = list(PREVENTIVE_ACTIONS_BY_RISK[risk_category])

    alerts: list[PharmacogenomicEntry] = []
    for gene in carried_genes:
        gene_upper = gene.upper().strip()
        if gene_upper in kb:
            alerts.extend(kb[gene_upper])

    profile = TherapeuticProfile(
        sample_id=sample_id,
        risk_probability=risk_probability,
        risk_category=risk_category,
        preventive_recommendations=preventive,
        pharmacogenomic_alerts=alerts,
    )
    logger.info(
        "Profil thérapeutique généré pour %s : catégorie=%s, %d alerte(s) pharmacogénomique(s).",
        sample_id,
        risk_category.value,
        len(alerts),
    )
    return profile

"""Tableau de bord interactif GenoPredict-AI (Streamlit).

Ce tableau de bord permet de :
  1. Charger un jeu de données de variants génétiques (ou utiliser le
     jeu de données synthétique de démonstration).
  2. Entraîner un modèle prédictif avec validation croisée.
  3. Sélectionner un échantillon et visualiser son score de risque, les
     explications SHAP associées, et les recommandations préventives.

Lancement local :
    streamlit run src/api_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features, ingestion, models, treatment  # noqa: E402

st.set_page_config(
    page_title="GenoPredict-AI",
    page_icon="🧬",
    layout="wide",
)

CLINICAL_COLUMNS = ["sex", "age", "bmi", "smoking_status"]
NUMERIC_CLINICAL = ["age", "bmi"]
CATEGORICAL_CLINICAL = ["sex", "smoking_status"]
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_data(show_spinner=False)
def _load_default_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge (ou génère à la volée) le jeu de données synthétique par défaut."""
    variant_path = DEFAULT_DATA_DIR / "variant_matrix.csv"
    annotation_path = DEFAULT_DATA_DIR / "annotations.csv"
    if not variant_path.exists() or not annotation_path.exists():
        sys.path.insert(0, str(DEFAULT_DATA_DIR))
        from generate_synthetic_data import generate_dataset  # type: ignore

        variant_df, annotation_df = generate_dataset(n_samples=600, n_variants=40)
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        variant_df.to_csv(variant_path, index=False)
        annotation_df.to_csv(annotation_path, index=False)
    return (
        ingestion.load_variant_matrix(variant_path),
        ingestion.load_annotations(annotation_path),
    )


@st.cache_resource(show_spinner=True)
def _run_pipeline(
    variant_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    model_type: str,
    n_splits: int,
    call_rate_threshold: float,
    maf_threshold: float,
):
    """Exécute le pipeline complet (QC -> features -> entraînement) et le met en cache."""
    variant_cols = ingestion.identify_variant_columns(variant_df, CLINICAL_COLUMNS)
    clean_df, qc_report = ingestion.quality_control(
        variant_df,
        variant_cols,
        call_rate_threshold=call_rate_threshold,
        maf_threshold=maf_threshold,
    )
    kept_variant_cols = [c for c in variant_cols if c not in qc_report.dropped_variants]

    prs = features.compute_polygenic_risk_score(
        clean_df[kept_variant_cols], annotation_df["effect_size"]
    )

    y = clean_df["disease_status"]
    idx_list = list(clean_df.index)
    train_idx, test_idx = train_test_split(
        idx_list, test_size=0.2, stratify=y.to_numpy(), random_state=42
    )

    spec = features.FeatureSpec(
        numeric_clinical_columns=NUMERIC_CLINICAL,
        categorical_clinical_columns=CATEGORICAL_CLINICAL,
        variant_columns=kept_variant_cols,
        include_prs=True,
    )
    builder = features.FeatureBuilder(spec)
    X_train = builder.fit_transform(clean_df.loc[train_idx], prs=prs.loc[train_idx])
    X_test = builder.transform(clean_df.loc[test_idx], prs=prs.loc[test_idx])
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    model, cv_report = models.train_model(
        X_train, y_train, model_type=model_type, n_splits=n_splits
    )
    risk_df = models.predict_risk(model, X_test)
    explanation = models.compute_shap_values(model, X_test)

    return {
        "qc_report": qc_report,
        "cv_report": cv_report,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "risk_df": risk_df,
        "explanation": explanation,
        "clean_df": clean_df,
        "kept_variant_cols": kept_variant_cols,
    }


def _render_shap_bar_chart(shap_df: pd.DataFrame) -> go.Figure:
    """Construit un graphique en barres Plotly des contributions SHAP."""
    shap_df = shap_df.sort_values("shap_value")
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_df["shap_value"]]
    fig = go.Figure(
        go.Bar(
            x=shap_df["shap_value"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in shap_df["shap_value"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Contributions SHAP (rouge = augmente le risque, bleu = diminue)",
        xaxis_title="Valeur SHAP",
        yaxis_title="Caractéristique",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _render_gauge(probability: float) -> go.Figure:
    """Construit une jauge Plotly représentant la probabilité de risque prédite."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2c3e50"},
                "steps": [
                    {"range": [0, 25], "color": "#a8e6a1"},
                    {"range": [25, 50], "color": "#fff3a1"},
                    {"range": [50, 75], "color": "#ffcf9e"},
                    {"range": [75, 100], "color": "#ff9e9e"},
                ],
            },
            title={"text": "Probabilité de pathogénicité prédite"},
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def main() -> None:
    """Point d'entrée de l'application Streamlit."""
    st.title("🧬 GenoPredict-AI")
    st.caption(
        "Plateforme de démonstration : diagnostic prédictif et modélisation "
        "thérapeutique des maladies génétiques (données 100% simulées)."
    )
    st.warning(
        "⚠️ Outil de recherche et de démonstration. Ne constitue pas un "
        "dispositif médical ni un outil de diagnostic clinique.",
        icon="⚠️",
    )

    with st.sidebar:
        st.header("Configuration du pipeline")
        uploaded_variants = st.file_uploader(
            "Matrice de variants (CSV)", type="csv", key="variants_upload"
        )
        uploaded_annotations = st.file_uploader(
            "Annotations de variants (CSV)", type="csv", key="annotations_upload"
        )
        model_type = st.selectbox("Modèle", ["xgboost", "random_forest"], index=0)
        n_splits = st.slider("Nombre de plis (StratifiedKFold)", 3, 10, 5)
        call_rate_threshold = st.slider("Seuil de taux d'appel minimal", 0.5, 1.0, 0.90)
        maf_threshold = st.slider("Seuil MAF minimal", 0.0, 0.10, 0.01)
        run_button = st.button("🚀 Lancer le pipeline", type="primary")

    if uploaded_variants is not None and uploaded_annotations is not None:
        variant_df = pd.read_csv(uploaded_variants).set_index(
            ingestion.REQUIRED_SAMPLE_COLUMN
        )
        annotation_df = pd.read_csv(uploaded_annotations).set_index("variant_id")
        st.info("Utilisation du jeu de données chargé par l'utilisateur.")
    else:
        variant_df, annotation_df = _load_default_data()
        st.info("Utilisation du jeu de données synthétique de démonstration.")

    if "pipeline_result" not in st.session_state or run_button:
        with st.spinner("Exécution du pipeline (QC, features, entraînement)..."):
            st.session_state["pipeline_result"] = _run_pipeline(
                variant_df,
                annotation_df,
                model_type,
                n_splits,
                call_rate_threshold,
                maf_threshold,
            )

    result = st.session_state["pipeline_result"]
    cv_report = result["cv_report"]
    qc_report = result["qc_report"]

    st.subheader("📊 Performance du modèle (validation croisée)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{cv_report.mean_roc_auc:.3f}")
    col2.metric("Sensibilité", f"{cv_report.mean_sensitivity:.3f}")
    col3.metric("Spécificité", f"{cv_report.mean_specificity:.3f}")
    col4.metric("F1-score", f"{cv_report.mean_f1:.3f}")

    with st.expander("Rapport de contrôle qualité (QC)"):
        st.text(qc_report.summary())

    st.divider()
    st.subheader("🔍 Prédiction individuelle et explication clinique")

    sample_ids = result["X_test"].index.tolist()
    selected_sample = st.selectbox("Choisir un échantillon de test", sample_ids)

    sample_position = sample_ids.index(selected_sample)
    probability = float(result["risk_df"].loc[selected_sample, "probability"])
    risk_category = result["risk_df"].loc[selected_sample, "risk_category"]

    col_gauge, col_shap = st.columns([1, 1.4])
    with col_gauge:
        st.plotly_chart(_render_gauge(probability), width='stretch')
        st.metric("Catégorie de risque", risk_category.capitalize())

    with col_shap:
        shap_df = models.top_shap_features(
            result["explanation"], sample_index=sample_position, top_k=8
        )
        st.plotly_chart(_render_shap_bar_chart(shap_df), width='stretch')

    st.divider()
    st.subheader("💊 Recommandations préventives et pharmacogénomiques")

    kept_variant_cols = result["kept_variant_cols"]
    sample_dosages = result["clean_df"].loc[selected_sample, kept_variant_cols]
    carried_variant_ids = sample_dosages[sample_dosages > 0].index.tolist()
    carried_genes = (
        annotation_df.loc[
            annotation_df.index.intersection(carried_variant_ids), "gene"
        ]
        .unique()
        .tolist()
    )

    profile = treatment.recommend_care(
        sample_id=str(selected_sample),
        risk_probability=probability,
        carried_genes=carried_genes,
    )

    st.markdown("**Recommandations de prise en charge préventive :**")
    for rec in profile.preventive_recommendations:
        st.markdown(f"- {rec}")

    if profile.pharmacogenomic_alerts:
        st.markdown("**Alertes pharmacogénomiques :**")
        for alert in profile.pharmacogenomic_alerts:
            st.markdown(
                f"- **{alert.gene}** ({alert.drug_class}, niveau de preuve "
                f"{alert.cpic_level}) : {alert.guidance}"
            )
    else:
        st.markdown("_Aucune alerte pharmacogénomique pour les gènes portés par cet échantillon._")

    st.caption(profile.disclaimer)


if __name__ == "__main__":
    main()

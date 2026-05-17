from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RISK_ORDER = ["Faible", "Moyen", "Eleve", "Critique"]
RISK_COLORS = {
    "Faible": "#2E8B57",
    "Moyen": "#D9A441",
    "Eleve": "#D96C3B",
    "Critique": "#B23A48",
}


st.set_page_config(
    page_title="Credit Risk Scoring & Portfolio Monitoring",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        border: 1px solid #e8ecef;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
    }
    h1, h2, h3 {letter-spacing: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_metrics() -> dict:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {}
    import json

    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def require_pipeline(scoring: pd.DataFrame) -> None:
    if scoring.empty:
        st.warning("Pipeline non execute. Lancez `python -m src.run_pipeline` depuis la racine du projet.")
        st.stop()


def format_money(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


scoring = load_csv("scoring_results.csv")
features = load_csv("features_credit_risk.csv")
dq_anomalies = load_csv("dq_anomalies.csv")
dq_report = load_csv("data_quality_report.csv")
monitoring = load_csv("model_monitoring.csv")
drift_report = load_csv("drift_report.csv")
metrics = load_metrics()

require_pipeline(scoring)

with st.sidebar:
    st.header("Pilotage")
    page = st.radio(
        "Vue",
        [
            "Vue globale",
            "Analyse risque",
            "Scoring clients",
            "Qualite donnees",
            "Monitoring modele",
        ],
        label_visibility="collapsed",
    )

    selected_risk = st.multiselect("Classe de risque", RISK_ORDER, default=RISK_ORDER)
    selected_types = st.multiselect(
        "Type credit",
        sorted(scoring["type_credit"].dropna().unique().tolist()),
        default=sorted(scoring["type_credit"].dropna().unique().tolist()),
    )

filtered = scoring[
    scoring["risk_class"].isin(selected_risk) & scoring["type_credit"].isin(selected_types)
].copy()

high_risk = filtered["risk_class"].isin(["Eleve", "Critique"])
default_rate = filtered["default_flag"].mean() if len(filtered) else 0
avg_score = filtered["probability_default"].mean() if len(filtered) else 0
risk_amount = filtered.loc[high_risk, "montant_credit"].sum()

if page == "Vue globale":
    header(
        "Credit Risk Scoring & Portfolio Monitoring",
        "Portefeuille, scores, risque observe et exposition a risque.",
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Clients", f"{filtered['client_id'].nunique():,}".replace(",", " "))
    col2.metric("Credits", f"{len(filtered):,}".replace(",", " "))
    col3.metric("Montant finance", format_money(filtered["montant_credit"].sum()))
    col4.metric("Taux defaut", f"{default_rate:.1%}")
    col5.metric("Score moyen", f"{avg_score:.1%}")

    left, right = st.columns([1.15, 0.85])
    with left:
        by_month = filtered.copy()
        by_month["date_demande"] = pd.to_datetime(by_month["date_demande"], errors="coerce")
        by_month["mois"] = by_month["date_demande"].dt.to_period("M").astype(str)
        monthly = (
            by_month.groupby("mois")
            .agg(
                credits=("credit_id", "count"),
                avg_score=("probability_default", "mean"),
                observed_default=("default_flag", "mean"),
            )
            .reset_index()
        )
        fig = px.line(
            monthly,
            x="mois",
            y=["avg_score", "observed_default"],
            markers=True,
            labels={"value": "Taux", "mois": "Mois", "variable": "Indicateur"},
            color_discrete_sequence=["#3B78E7", "#B23A48"],
        )
        fig.update_layout(height=390, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        risk_dist = filtered["risk_class"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
        risk_dist.columns = ["risk_class", "credits"]
        fig = px.bar(
            risk_dist,
            x="risk_class",
            y="credits",
            color="risk_class",
            color_discrete_map=RISK_COLORS,
            labels={"risk_class": "Classe", "credits": "Credits"},
        )
        fig.update_layout(height=390, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Exposition prioritaire")
    st.metric("Encours a risque eleve ou critique", format_money(risk_amount))
    st.dataframe(
        filtered.sort_values("probability_default", ascending=False).head(12),
        use_container_width=True,
        hide_index=True,
    )

elif page == "Analyse risque":
    header("Analyse risque", "Lecture metier par type de credit, ville, revenu et endettement.")

    col1, col2 = st.columns(2)
    with col1:
        by_type = (
            filtered.groupby("type_credit")
            .agg(default_rate=("default_flag", "mean"), avg_score=("probability_default", "mean"), credits=("credit_id", "count"))
            .reset_index()
            .sort_values("default_rate", ascending=False)
        )
        fig = px.bar(
            by_type,
            x="type_credit",
            y="default_rate",
            color="avg_score",
            color_continuous_scale="Reds",
            labels={"type_credit": "Type credit", "default_rate": "Taux defaut", "avg_score": "Score moyen"},
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        by_city = (
            filtered.groupby("ville")
            .agg(avg_score=("probability_default", "mean"), credits=("credit_id", "count"))
            .reset_index()
            .sort_values("avg_score", ascending=False)
        )
        fig = px.bar(
            by_city,
            x="avg_score",
            y="ville",
            orientation="h",
            color="avg_score",
            color_continuous_scale="OrRd",
            labels={"avg_score": "Score moyen", "ville": "Ville"},
        )
        fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(
            filtered,
            x="taux_endettement",
            color="risk_class",
            nbins=35,
            color_discrete_map=RISK_COLORS,
            labels={"taux_endettement": "Taux endettement", "risk_class": "Classe"},
        )
        fig.update_layout(height=360, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            filtered,
            x="revenu_mensuel",
            y="montant_credit",
            color="risk_class",
            size="probability_default",
            color_discrete_map=RISK_COLORS,
            labels={"revenu_mensuel": "Revenu mensuel", "montant_credit": "Montant credit"},
        )
        fig.update_layout(height=360, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Scoring clients":
    header("Scoring clients", "Probabilite de defaut, classe de risque et recommandation.")

    min_score, max_score = st.slider(
        "Score PD",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
    )
    table = filtered[
        filtered["probability_default"].between(min_score, max_score, inclusive="both")
    ].sort_values("probability_default", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Dossiers filtres", f"{len(table):,}".replace(",", " "))
    col2.metric("PD moyenne", f"{table['probability_default'].mean() if len(table) else 0:.1%}")
    col3.metric("Critiques", f"{(table['risk_class'] == 'Critique').sum():,}".replace(",", " "))

    st.dataframe(
        table[
            [
                "client_id",
                "credit_id",
                "type_credit",
                "ville",
                "montant_credit",
                "taux_endettement",
                "probability_default",
                "risk_class",
                "recommendation",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

elif page == "Qualite donnees":
    header("Qualite donnees", "Controles, anomalies et score qualite par table.")

    if dq_report.empty:
        st.info("Aucun rapport Data Quality disponible.")
    else:
        cols = st.columns(len(dq_report))
        for col, (_, row) in zip(cols, dq_report.iterrows()):
            col.metric(row["table_name"], f"{row['quality_score']:.1f}/100", f"{row['anomalies_detected']} anomalies")

    col1, col2 = st.columns([0.9, 1.1])
    with col1:
        if not dq_anomalies.empty:
            by_rule = (
                dq_anomalies.groupby(["rule_name", "severity"])
                .size()
                .reset_index(name="anomalies")
                .sort_values("anomalies", ascending=False)
                .head(12)
            )
            fig = px.bar(
                by_rule,
                x="anomalies",
                y="rule_name",
                color="severity",
                orientation="h",
                labels={"rule_name": "Regle", "anomalies": "Anomalies"},
                color_discrete_map={"High": "#B23A48", "Medium": "#D9A441"},
            )
            fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            dq_anomalies.sort_values("detected_at", ascending=False).head(50),
            use_container_width=True,
            hide_index=True,
        )

elif page == "Monitoring modele":
    header("Monitoring modele", "Performance mensuelle, taux observe vs predit et derive PSI.")

    if metrics:
        selected = metrics["selected_model"]
        selected_metrics = metrics["metrics"][selected]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Modele", selected.replace("_", " ").title())
        col2.metric("AUC test", f"{selected_metrics['auc']:.3f}")
        col3.metric("Recall test", f"{selected_metrics['recall']:.3f}")
        col4.metric("F1 test", f"{selected_metrics['f1_score']:.3f}")

    if monitoring.empty:
        st.info("Aucun monitoring disponible.")
    else:
        mon = monitoring.sort_values("scoring_month")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                mon,
                x="scoring_month",
                y=["auc_score", "precision_score", "recall_score", "f1_score"],
                markers=True,
                labels={"value": "Score", "scoring_month": "Mois", "variable": "Metrique"},
            )
            fig.update_layout(height=390, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                mon,
                x="scoring_month",
                y=["default_rate_observed", "default_rate_predicted", "population_stability_index"],
                markers=True,
                labels={"value": "Valeur", "scoring_month": "Mois", "variable": "Indicateur"},
            )
            fig.update_layout(height=390, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

    if not drift_report.empty:
        fig = px.bar(
            drift_report,
            x="scoring_month",
            y="credits",
            color="risk_class",
            color_discrete_map=RISK_COLORS,
            labels={"credits": "Credits", "scoring_month": "Mois", "risk_class": "Classe"},
        )
        fig.update_layout(height=390, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

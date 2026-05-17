"""
API REST de scoring crédit temps-réel.

Lancement :
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Documentation interactive :
    http://localhost:8000/docs
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .config import MODELS_DIR, PROCESSED_DIR, MODEL_VERSION, risk_class, risk_recommendation
from .preprocessing import FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Chargement du modèle (singleton au démarrage)
# ---------------------------------------------------------------------------

_bundle: dict | None = None
_shap_explainer = None
_scoring_cache: pd.DataFrame | None = None


def _load_model() -> None:
    global _bundle, _shap_explainer, _scoring_cache
    model_path = MODELS_DIR / "credit_score_model.pkl"
    if not model_path.exists():
        return  # Le pipeline n'a pas encore tourné

    _bundle = joblib.load(model_path)

    # Pré-charger l'explainer SHAP une seule fois (coûteux à initialiser)
    if _SHAP_AVAILABLE:
        try:
            classifier = _bundle["model"].named_steps["model"]
            _shap_explainer = _shap.TreeExplainer(
                classifier, feature_perturbation="tree_path_dependent"
            )
        except Exception:
            _shap_explainer = None

    # Cache du scoring batch (pour lookup par credit_id)
    scores_path = PROCESSED_DIR / "scoring_results.csv"
    if scores_path.exists():
        _scoring_cache = pd.read_csv(scores_path)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    _load_model()
    yield


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    lifespan=lifespan,
    title="Credit Risk Scoring API",
    description=(
        "API de scoring crédit pour le secteur bancaire et assurantiel marocain. "
        "Fournit une probabilité de défaut (PD), une classe de risque et une "
        "explication SHAP par dossier."
    ),
    version="1.0.0",
    contact={"name": "Risk Data Team"},
    license_info={"name": "Propriétaire"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------

class CreditApplicationInput(BaseModel):
    """Données d'entrée pour scorer un dossier de crédit."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "client_id": "C00042",
                "credit_id": None,
                "age_group": "26-35",
                "revenu_mensuel": 12000.0,
                "ville": "Casablanca",
                "secteur_activite": "Salarie prive",
                "anciennete_emploi": 5.0,
                "montant_credit": 50000.0,
                "duree_mois": 36,
                "mensualite": 1550.0,
                "taux_endettement": 35.0,
                "type_credit": "Conso",
                "nombre_retards_12m": 0,
                "max_jours_retard": 0,
                "nb_impayes": 0,
                "nb_credits_actifs": 1,
                "total_encours": 50000.0,
                "mois_demande": 6,
            }
        }
    )

    client_id: str = Field(..., description="Identifiant unique client")
    credit_id: Optional[str] = Field(None, description="Identifiant dossier (optionnel pour nouveau dossier)")

    # Profil client
    age_group: str = Field(..., description="Tranche d'âge : 18-25 | 26-35 | 36-45 | 46-60 | 60+")
    revenu_mensuel: float = Field(..., gt=0, description="Revenu mensuel net en DH")
    ville: str = Field(..., description="Ville de résidence")
    secteur_activite: str = Field(..., description="Secteur d'activité")
    anciennete_emploi: float = Field(..., ge=0, description="Ancienneté emploi en années")

    # Caractéristiques du crédit
    montant_credit: float = Field(..., gt=0, description="Montant du crédit en DH")
    duree_mois: int = Field(..., gt=0, le=360, description="Durée en mois")
    mensualite: float = Field(..., gt=0, description="Mensualité en DH")
    taux_endettement: float = Field(..., ge=0, le=100, description="Taux d'endettement en %")
    type_credit: str = Field(..., description="Type : Conso | Auto | Immobilier | Equipement | Tresorerie")

    # Historique de paiement (0 pour un nouveau client sans historique)
    nombre_retards_12m: int = Field(0, ge=0, description="Nombre de retards sur 12 mois")
    max_jours_retard: int = Field(0, ge=0, description="Maximum de jours de retard observé")
    nb_impayes: int = Field(0, ge=0, description="Nombre d'impayés")

    # Portefeuille client
    nb_credits_actifs: int = Field(1, ge=0, description="Nombre de crédits actifs")
    total_encours: float = Field(0.0, ge=0, description="Encours total en DH")

    # Contextuel
    mois_demande: Optional[int] = Field(None, ge=1, le=12, description="Mois de la demande (1-12)")


class ShapFactor(BaseModel):
    facteur: str
    impact_shap: float
    sens: str  # hausse_risque | baisse_risque


class ScoreResponse(BaseModel):
    """Réponse du scoring avec explication réglementaire."""

    request_id: str = Field(..., description="Identifiant unique de la requête (audit)")
    client_id: str
    credit_id: Optional[str]
    probability_default: float = Field(..., description="Probabilité de défaut [0.0 - 1.0]")
    risk_class: str = Field(..., description="Classe de risque : Faible | Moyen | Eleve | Critique")
    recommendation: str = Field(..., description="Action recommandée")
    top_risk_factors: list[ShapFactor] = Field(..., description="Top 3 facteurs SHAP explicatifs")
    model_version: str
    scored_at: str
    shap_available: bool


class PortfolioSummary(BaseModel):
    total_credits: int
    montant_total_finance: float
    taux_defaut_prevu: float
    repartition_risque: dict[str, int]
    top5_credits_risques: list[dict]


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _build_feature_dataframe(app_in: CreditApplicationInput) -> pd.DataFrame:
    """Construit le DataFrame feature à partir de l'input API."""
    revenu = max(app_in.revenu_mensuel, 1.0)
    mois = app_in.mois_demande or datetime.now(timezone.utc).month

    row = {
        "age_group": app_in.age_group,
        "revenu_mensuel": app_in.revenu_mensuel,
        "ville": app_in.ville,
        "secteur_activite": app_in.secteur_activite,
        "anciennete_emploi": app_in.anciennete_emploi,
        "montant_credit": app_in.montant_credit,
        "duree_mois": app_in.duree_mois,
        "mensualite": app_in.mensualite,
        "taux_endettement": app_in.taux_endettement,
        "type_credit": app_in.type_credit,
        "nombre_retards_12m": app_in.nombre_retards_12m,
        "max_jours_retard": app_in.max_jours_retard,
        "nb_impayes": app_in.nb_impayes,
        "nb_credits_actifs": app_in.nb_credits_actifs,
        "total_encours": app_in.total_encours,
        # Features dérivées calculées ici
        "ratio_credit_revenu": app_in.montant_credit / revenu,
        "ratio_mensualite_revenu": app_in.mensualite / revenu,
        "encours_sur_revenu": app_in.total_encours / revenu,
        "mois_demande": mois,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def _get_shap_factors(X: pd.DataFrame, model) -> list[ShapFactor]:
    """Retourne les 3 facteurs SHAP les plus importants pour un dossier."""
    if not _SHAP_AVAILABLE or _shap_explainer is None:
        return []

    try:
        preprocessor = model.named_steps["preprocess"]
        X_transformed = preprocessor.transform(X)
        transformed_names = list(preprocessor.get_feature_names_out())

        sv_raw = _shap_explainer.shap_values(X_transformed)
        # Gestion multi-versions SHAP :
        #   - shap < 0.44  → list[ndarray(n, f)]  → sv_raw[1][0]
        #   - shap >= 0.44 → ndarray(n, f, 2)     → sv_raw[0, :, 1]
        if isinstance(sv_raw, list):
            sv_row = sv_raw[1][0]
        elif sv_raw.ndim == 3:
            sv_row = sv_raw[0, :, 1]
        else:
            sv_row = sv_raw[0]

        # Agrégation OHE → features originales
        feat_shap: dict[str, float] = {}
        for feat in FEATURE_COLUMNS:
            idxs = [
                i for i, n in enumerate(transformed_names)
                if n == f"num__{feat}" or n.startswith(f"cat__{feat}_")
            ]
            if idxs:
                feat_shap[feat] = float(np.sum(sv_row[idxs]))

        top3 = sorted(feat_shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        return [
            ShapFactor(
                facteur=feat,
                impact_shap=round(val, 4),
                sens="hausse_risque" if val > 0 else "baisse_risque",
            )
            for feat, val in top3
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Système"])
def health_check() -> dict:
    """Vérifie l'état de santé de l'API et du modèle chargé."""
    return {
        "status": "ok",
        "model_loaded": _bundle is not None,
        "model_version": _bundle.get("model_version") if _bundle else None,
        "shap_available": _SHAP_AVAILABLE and _shap_explainer is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_application(application: CreditApplicationInput) -> ScoreResponse:
    """
    Score un dossier de crédit en temps-réel.

    Retourne :
    - **probability_default** : probabilité de défaut [0.0 – 1.0]
    - **risk_class** : Faible / Moyen / Eleve / Critique
    - **recommendation** : action recommandée au gestionnaire de crédit
    - **top_risk_factors** : top 3 facteurs explicatifs (SHAP) pour conformité RGPD Art. 22
    """
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Lancez d'abord : python -m src.run_pipeline",
        )

    model = _bundle["model"]
    X = _build_feature_dataframe(application)

    prob = float(model.predict_proba(X)[0][1])
    r_class = risk_class(prob)
    rec = risk_recommendation(r_class)
    factors = _get_shap_factors(X, model)

    return ScoreResponse(
        request_id=str(uuid.uuid4()),
        client_id=application.client_id,
        credit_id=application.credit_id,
        probability_default=round(prob, 4),
        risk_class=r_class,
        recommendation=rec,
        top_risk_factors=factors,
        model_version=_bundle.get("model_version", MODEL_VERSION),
        scored_at=datetime.now(timezone.utc).isoformat(),
        shap_available=_SHAP_AVAILABLE and _shap_explainer is not None,
    )


@app.get("/score/{credit_id}", tags=["Scoring"])
def get_existing_score(credit_id: str) -> dict:
    """
    Récupère le score batch existant pour un credit_id donné.
    Utile pour les consultations après scoring nocturne.
    """
    if _scoring_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Aucun scoring batch disponible. Lancez d'abord le pipeline.",
        )

    row = _scoring_cache[_scoring_cache["credit_id"] == credit_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"credit_id '{credit_id}' introuvable.")

    record = row.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in record.items()}


@app.get("/portfolio/summary", response_model=PortfolioSummary, tags=["Portfolio"])
def portfolio_summary() -> PortfolioSummary:
    """
    Résumé du portefeuille de crédit avec répartition par classe de risque.
    Équivalent du tableau de bord Risk pour la Direction des Risques.
    """
    if _scoring_cache is None or _scoring_cache.empty:
        raise HTTPException(status_code=503, detail="Aucun scoring batch disponible.")

    df = _scoring_cache.copy()
    repartition = df["risk_class"].value_counts().to_dict()

    top5 = (
        df.nlargest(5, "probability_default")[
            ["credit_id", "client_id", "type_credit", "montant_credit", "probability_default", "risk_class"]
        ]
        .to_dict(orient="records")
    )

    return PortfolioSummary(
        total_credits=int(len(df)),
        montant_total_finance=float(df["montant_credit"].sum()),
        taux_defaut_prevu=round(float((df["probability_default"] >= 0.25).mean()), 4),
        repartition_risque={str(k): int(v) for k, v in repartition.items()},
        top5_credits_risques=[
            {k: (None if pd.isna(v) else v) for k, v in r.items()} for r in top5
        ],
    )


@app.get("/model/info", tags=["Modèle"])
def model_info() -> dict:
    """Métadonnées du modèle déployé (version, features, seuil de décision)."""
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    metrics_path = MODELS_DIR / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        import json as _json
        with open(metrics_path, encoding="utf-8") as f:
            metrics = _json.load(f)

    fairness_path = MODELS_DIR / "fairness_report.csv"
    fairness = []
    if fairness_path.exists():
        fairness = pd.read_csv(fairness_path).to_dict(orient="records")

    return {
        "model_version": _bundle.get("model_version"),
        "selected_model": _bundle.get("selected_model"),
        "decision_threshold": _bundle.get("decision_threshold"),
        "feature_columns": _bundle.get("feature_columns"),
        "metrics": metrics.get("metrics", {}),
        "fairness_report": fairness,
        "shap_available": _SHAP_AVAILABLE,
    }

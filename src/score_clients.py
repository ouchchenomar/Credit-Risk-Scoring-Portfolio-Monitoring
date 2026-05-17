from __future__ import annotations

import json
import uuid

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, MODEL_VERSION, PROCESSED_DIR, ensure_directories, risk_class, risk_recommendation
from .preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def _get_transformed_feature_names(model) -> list[str]:
    """Retourne les noms de colonnes après OneHotEncoding + scaling."""
    try:
        return list(model.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        return []


def _aggregate_shap_by_original_feature(
    shap_row: np.ndarray,
    transformed_names: list[str],
    original_features: list[str],
) -> dict[str, float]:
    """
    Agrège les valeurs SHAP des colonnes OHE vers les features originales.
    Colonnes numériques : préfixe 'num__<feature>'
    Colonnes OHE       : préfixe 'cat__<feature>_<valeur>'
    """
    result: dict[str, float] = {}
    for feat in original_features:
        indices = [
            i for i, name in enumerate(transformed_names)
            if name == f"num__{feat}" or name.startswith(f"cat__{feat}_")
        ]
        if indices:
            result[feat] = float(np.sum(shap_row[indices]))
    return result


def _compute_shap_explanations(
    model,
    X: pd.DataFrame,
    original_features: list[str],
) -> list[str]:
    """
    Calcule les 3 facteurs de risque les plus impactants par dossier (SHAP).
    Retourne une liste de JSON strings, une par ligne.
    Fallback sur liste vide si SHAP n'est pas disponible.
    """
    if not _SHAP_AVAILABLE:
        return [json.dumps([]) for _ in range(len(X))]

    try:
        preprocessor = model.named_steps["preprocess"]
        classifier = model.named_steps["model"]

        X_transformed = preprocessor.transform(X)
        transformed_names = _get_transformed_feature_names(model)

        explainer = _shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")
        raw = explainer.shap_values(X_transformed)

        # Gestion multi-versions SHAP :
        #   - shap < 0.44  → list[ndarray(n, f)]  → raw[1][i] pour classe défaut
        #   - shap >= 0.44 → ndarray(n, f, 2)     → raw[i, :, 1] pour classe défaut
        def _extract_row(i: int) -> np.ndarray:
            if isinstance(raw, list):
                return raw[1][i]
            if raw.ndim == 3:          # (n_samples, n_features, n_classes)
                return raw[i, :, 1]
            return raw[i]              # (n_samples, n_features) binaire

        explanations: list[str] = []
        for i in range(len(X)):
            agg = _aggregate_shap_by_original_feature(_extract_row(i), transformed_names, original_features)
            top3 = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            factors = [
                {
                    "facteur": feat,
                    "impact_shap": round(val, 4),
                    "sens": "hausse_risque" if val > 0 else "baisse_risque",
                }
                for feat, val in top3
            ]
            explanations.append(json.dumps(factors, ensure_ascii=False))

        return explanations

    except Exception as exc:  # SHAP peut échouer selon la version de scikit-learn
        return [json.dumps([{"erreur": str(exc)}]) for _ in range(len(X))]


def score_clients() -> pd.DataFrame:
    ensure_directories()
    bundle = joblib.load(MODELS_DIR / "credit_score_model.pkl")
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    features = pd.read_csv(PROCESSED_DIR / "features_credit_risk.csv")
    X = features[feature_columns]

    probabilities = model.predict_proba(X)[:, 1]
    risk_classes = [risk_class(float(p)) for p in probabilities]

    # --- Scoring résultats ---
    scoring = features[
        [
            "client_id",
            "credit_id",
            "date_demande",
            "montant_credit",
            "type_credit",
            "ville",
            "revenu_mensuel",
            "taux_endettement",
            "default_flag",
        ]
    ].copy()
    scoring["probability_default"] = probabilities.round(4)
    scoring["risk_class"] = risk_classes
    scoring["recommendation"] = [risk_recommendation(r) for r in risk_classes]
    scoring["score_date"] = pd.Timestamp.utcnow().date().isoformat()
    scoring["model_version"] = bundle.get("model_version", MODEL_VERSION)
    scoring.to_csv(PROCESSED_DIR / "scoring_results.csv", index=False)

    # --- SHAP explicabilité (top 3 facteurs par dossier) ---
    all_features = list(set(FEATURE_COLUMNS + list(feature_columns)))
    shap_explanations = _compute_shap_explanations(model, X, FEATURE_COLUMNS)

    # --- Audit trail réglementaire ---
    # Chaque ligne = snapshot complet de la décision (inputs + outputs + explication)
    audit_rows = []
    for idx, (_, row) in enumerate(features.iterrows()):
        audit_rows.append(
            {
                "audit_id": str(uuid.uuid4()),
                "credit_id": row.get("credit_id"),
                "client_id": row.get("client_id"),
                "model_version": bundle.get("model_version", MODEL_VERSION),
                "decision_threshold": float(bundle.get("decision_threshold", 0.25)),
                "scored_at": pd.Timestamp.utcnow().isoformat(),
                # Snapshot des inputs au moment du score (tracé pour contestation future)
                "input_features": json.dumps(
                    {col: (None if pd.isna(row.get(col)) else row.get(col)) for col in feature_columns},
                    ensure_ascii=False,
                ),
                "probability_default": round(float(probabilities[idx]), 4),
                "risk_class": risk_classes[idx],
                "top_shap_factors": shap_explanations[idx],
                "scored_by": "batch-pipeline",
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(PROCESSED_DIR / "scoring_audit.csv", index=False)

    shap_status = "OK" if _SHAP_AVAILABLE else "non disponible (pip install shap)"
    print(f"  SHAP: {shap_status}")
    print(f"  Audit trail: {len(audit_df)} lignes -> scoring_audit.csv")

    return scoring


def main() -> None:
    scoring = score_clients()
    print(f"Scoring terminé : {len(scoring)} dossiers scorés.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import MODEL_VERSION, MODELS_DIR, PROCESSED_DIR, ensure_directories, risk_class


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) < 2 or len(actual) < 2:
        return 0.0

    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0] = -np.inf
    cuts[-1] = np.inf

    expected_pct = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1)
    actual_pct = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1)
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def monitor_model() -> dict[str, pd.DataFrame]:
    ensure_directories()
    bundle = joblib.load(MODELS_DIR / "credit_score_model.pkl")
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = float(bundle.get("decision_threshold", 0.25))

    features = pd.read_csv(PROCESSED_DIR / "features_credit_risk.csv")
    features["date_demande"] = pd.to_datetime(features["date_demande"], errors="coerce")
    features["scoring_month"] = features["date_demande"].dt.to_period("M").astype(str)

    probabilities = model.predict_proba(features[feature_columns])[:, 1]
    features["probability_default"] = probabilities
    features["risk_class"] = [risk_class(float(probability)) for probability in probabilities]
    features["prediction"] = (features["probability_default"] >= threshold).astype(int)

    baseline_month = features["scoring_month"].min()
    baseline_scores = features.loc[features["scoring_month"] == baseline_month, "probability_default"].to_numpy()

    monitoring_rows = []
    for month, month_df in features.groupby("scoring_month"):
        y_true = month_df["default_flag"].astype(int)
        y_pred = month_df["prediction"].astype(int)
        scores = month_df["probability_default"].astype(float)

        auc_score = None
        if y_true.nunique() == 2:
            auc_score = round(float(roc_auc_score(y_true, scores)), 4)

        monitoring_rows.append(
            {
                "model_version": MODEL_VERSION,
                "scoring_month": month,
                "auc_score": auc_score,
                "precision_score": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall_score": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "default_rate_observed": round(float(y_true.mean()), 4),
                "default_rate_predicted": round(float(y_pred.mean()), 4),
                "population_stability_index": round(_psi(baseline_scores, scores.to_numpy()), 4),
                "created_at": pd.Timestamp.utcnow().isoformat(),
            }
        )

    monitoring = pd.DataFrame(monitoring_rows)
    monitoring.to_csv(PROCESSED_DIR / "model_monitoring.csv", index=False)

    drift_report = (
        features.groupby(["scoring_month", "risk_class"])
        .agg(
            credits=("credit_id", "count"),
            avg_probability_default=("probability_default", "mean"),
            observed_default_rate=("default_flag", "mean"),
            total_amount=("montant_credit", "sum"),
        )
        .reset_index()
    )
    drift_report["avg_probability_default"] = drift_report["avg_probability_default"].round(4)
    drift_report["observed_default_rate"] = drift_report["observed_default_rate"].round(4)
    drift_report.to_csv(PROCESSED_DIR / "drift_report.csv", index=False)

    return {"model_monitoring": monitoring, "drift_report": drift_report}


def main() -> None:
    outputs = monitor_model()
    print(f"Monitoring created: {len(outputs['model_monitoring'])} monthly rows.")


if __name__ == "__main__":
    main()


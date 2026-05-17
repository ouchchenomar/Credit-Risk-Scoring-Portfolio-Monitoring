from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_VERSION, MODELS_DIR, PROCESSED_DIR, ensure_directories
from .preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def _evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.25) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)

    return {
        "auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "pr_auc": round(float(auc(recall_curve, precision_curve)), 4),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
    }


def _feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance = pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": values,
                "direction": "non_linear",
            }
        )
    else:
        values = estimator.coef_[0]
        importance = pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": abs(values),
                "direction": ["positive" if value >= 0 else "negative" for value in values],
            }
        )

    return importance.sort_values("importance", ascending=False).head(30)


def _fairness_report(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features_test: pd.DataFrame,
    threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Calcule les taux de faux positifs / faux négatifs par groupe sensible.
    Conforme aux exigences EBA/RGPD sur la non-discrimination dans le crédit.
    """
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    rows = []
    for group_col in ["secteur_activite", "age_group"]:
        if group_col not in features_test.columns:
            continue
        for group_val in sorted(features_test[group_col].dropna().unique()):
            mask = features_test[group_col] == group_val
            y_g = y_test[mask]
            pred_g = predictions[mask]

            if len(y_g) < 10:  # pas assez de données pour une statistique fiable
                continue

            tn = int(((y_g == 0) & (pred_g == 0)).sum())
            fp = int(((y_g == 0) & (pred_g == 1)).sum())
            fn = int(((y_g == 1) & (pred_g == 0)).sum())
            tp = int(((y_g == 1) & (pred_g == 1)).sum())

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            rows.append(
                {
                    "attribut_sensible": group_col,
                    "groupe": group_val,
                    "n_dossiers": int(len(y_g)),
                    "taux_defaut_observe": round(float(y_g.mean()), 4),
                    "faux_positif_rate": round(float(fpr), 4),
                    "faux_negatif_rate": round(float(fnr), 4),
                    "recall": round(float(recall), 4),
                    "vrai_positifs": tp,
                    "faux_positifs": fp,
                    "vrai_negatifs": tn,
                    "faux_negatifs": fn,
                }
            )

    return pd.DataFrame(rows)


def train_model() -> dict:
    ensure_directories()
    features = pd.read_csv(PROCESSED_DIR / "features_credit_risk.csv")
    X = features[FEATURE_COLUMNS]
    y = features[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=240,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    metrics = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics[name] = _evaluate_model(model, X_test, y_test)

    best_name = max(metrics, key=lambda item: metrics[item]["auc"])
    best_model = candidates[best_name]

    bundle = {
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "model_version": MODEL_VERSION,
        "selected_model": best_name,
        "decision_threshold": 0.25,
    }
    joblib.dump(bundle, MODELS_DIR / "credit_score_model.pkl")

    importance = _feature_importance(best_model)
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    # Rapport d'équité réglementaire (EBA / RGPD Art. 22)
    fairness = _fairness_report(best_model, X_test, y_test, features.loc[X_test.index])
    fairness.to_csv(MODELS_DIR / "fairness_report.csv", index=False)

    payload = {
        "model_version": MODEL_VERSION,
        "selected_model": best_name,
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "target_rate": round(float(y.mean()), 4),
        "metrics": metrics,
    }
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    return payload


def main() -> None:
    metrics = train_model()
    selected = metrics["selected_model"]
    auc_score = metrics["metrics"][selected]["auc"]
    print(f"Model trained: {selected}, test AUC={auc_score}")


if __name__ == "__main__":
    main()


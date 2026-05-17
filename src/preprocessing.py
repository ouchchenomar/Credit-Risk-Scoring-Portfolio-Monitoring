from __future__ import annotations

import pandas as pd

from .config import PROCESSED_DIR, ensure_directories


FEATURE_COLUMNS = [
    "age_group",
    "revenu_mensuel",
    "ville",
    "secteur_activite",
    "anciennete_emploi",
    "montant_credit",
    "duree_mois",
    "mensualite",
    "taux_endettement",
    "type_credit",
    "nombre_retards_12m",
    "max_jours_retard",
    "nb_impayes",
    "nb_credits_actifs",
    "total_encours",
    "ratio_credit_revenu",
    "ratio_mensualite_revenu",
    "encours_sur_revenu",
    "mois_demande",
]

CATEGORICAL_FEATURES = ["age_group", "ville", "secteur_activite", "type_credit"]
NUMERIC_FEATURES = [feature for feature in FEATURE_COLUMNS if feature not in CATEGORICAL_FEATURES]
TARGET_COLUMN = "default_flag"


def build_features() -> pd.DataFrame:
    ensure_directories()
    clients = pd.read_csv(PROCESSED_DIR / "clients_clean.csv")
    credits = pd.read_csv(PROCESSED_DIR / "credit_applications_clean.csv")
    payments = pd.read_csv(PROCESSED_DIR / "payments_clean.csv")

    credits["date_demande"] = pd.to_datetime(credits["date_demande"], errors="coerce")
    payments["date_echeance"] = pd.to_datetime(payments["date_echeance"], errors="coerce")
    payments["date_paiement"] = pd.to_datetime(payments["date_paiement"], errors="coerce")
    payments["jours_retard"] = pd.to_numeric(payments["jours_retard"], errors="coerce").fillna(0)

    payments["is_late"] = payments["statut_paiement"].isin(["RETARD", "IMPAYE"]) | (payments["jours_retard"] > 0)
    payments["is_unpaid"] = payments["statut_paiement"].eq("IMPAYE")

    payment_features = (
        payments.groupby("credit_id")
        .agg(
            nombre_retards_12m=("is_late", "sum"),
            max_jours_retard=("jours_retard", "max"),
            nb_impayes=("is_unpaid", "sum"),
            montant_du_total=("montant_du", "sum"),
            montant_paye_total=("montant_paye", "sum"),
        )
        .reset_index()
    )

    client_portfolio = (
        credits.groupby("client_id")
        .agg(nb_credits_actifs=("credit_id", "count"), total_encours=("montant_credit", "sum"))
        .reset_index()
    )

    features = credits.merge(clients, on="client_id", how="left")
    features = features.merge(payment_features, on="credit_id", how="left")
    features = features.merge(client_portfolio, on="client_id", how="left")

    for column in [
        "nombre_retards_12m",
        "max_jours_retard",
        "nb_impayes",
        "montant_du_total",
        "montant_paye_total",
        "nb_credits_actifs",
        "total_encours",
    ]:
        features[column] = features[column].fillna(0)

    features["revenu_mensuel"] = pd.to_numeric(features["revenu_mensuel"], errors="coerce").fillna(0)
    features["ratio_credit_revenu"] = features["montant_credit"] / features["revenu_mensuel"].clip(lower=1)
    features["ratio_mensualite_revenu"] = features["mensualite"] / features["revenu_mensuel"].clip(lower=1)
    features["encours_sur_revenu"] = features["total_encours"] / features["revenu_mensuel"].clip(lower=1)
    features["mois_demande"] = features["date_demande"].dt.month.fillna(0).astype(int)

    output_columns = [
        "client_id",
        "credit_id",
        "date_demande",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]
    features = features[output_columns].copy()
    features.to_csv(PROCESSED_DIR / "features_credit_risk.csv", index=False)
    return features


def main() -> None:
    features = build_features()
    print(f"Feature table created: {len(features)} rows, {len(FEATURE_COLUMNS)} features.")


if __name__ == "__main__":
    main()


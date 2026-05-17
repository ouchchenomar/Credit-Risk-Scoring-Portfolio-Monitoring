from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RANDOM_SEED, RAW_DIR, SAMPLE_DIR, ensure_directories


@dataclass(frozen=True)
class GenerationConfig:
    clients: int = 1800
    credits: int = 2600
    seed: int = RANDOM_SEED


def _date_series(rng: np.random.Generator, size: int) -> pd.Series:
    start = pd.Timestamp("2024-01-01")
    offsets = rng.integers(0, 730, size=size)
    return pd.Series(start + pd.to_timedelta(offsets, unit="D")).dt.normalize()


def generate_clients(rng: np.random.Generator, n_clients: int) -> pd.DataFrame:
    age_groups = ["18-25", "26-35", "36-45", "46-60", "60+"]
    villes = ["Casablanca", "Rabat", "Marrakech", "Fes", "Tanger", "Agadir", "Oujda"]
    secteurs = [
        "Salarie prive",
        "Fonction publique",
        "Commerce",
        "Independant",
        "Industrie",
        "Services",
        "Agriculture",
        "Etudiant",
    ]

    revenus = rng.lognormal(mean=8.72, sigma=0.55, size=n_clients).round(0)
    revenus = np.clip(revenus, 2200, 52000)
    anciennete = np.maximum(0, rng.normal(loc=5.5, scale=4.0, size=n_clients)).round(1)

    clients = pd.DataFrame(
        {
            "client_id": [f"C{i:05d}" for i in range(1, n_clients + 1)],
            "age_group": rng.choice(age_groups, n_clients, p=[0.12, 0.30, 0.27, 0.22, 0.09]),
            "revenu_mensuel": revenus.astype(int),
            "ville": rng.choice(villes, n_clients, p=[0.33, 0.18, 0.12, 0.10, 0.12, 0.09, 0.06]),
            "secteur_activite": rng.choice(
                secteurs,
                n_clients,
                p=[0.30, 0.17, 0.13, 0.13, 0.09, 0.10, 0.04, 0.04],
            ),
            "anciennete_emploi": anciennete,
        }
    )

    anomaly_idx = rng.choice(clients.index, size=max(3, n_clients // 300), replace=False)
    clients.loc[anomaly_idx[: len(anomaly_idx) // 2], "client_id"] = np.nan
    duplicate_rows = clients.sample(4, random_state=RANDOM_SEED)
    clients = pd.concat([clients, duplicate_rows], ignore_index=True)
    return clients


def generate_credits(
    rng: np.random.Generator,
    clients: pd.DataFrame,
    n_credits: int,
) -> pd.DataFrame:
    valid_clients = clients.dropna(subset=["client_id"]).drop_duplicates("client_id")
    selected_clients = valid_clients.sample(n=n_credits, replace=True, random_state=RANDOM_SEED)
    selected_clients = selected_clients.reset_index(drop=True)

    type_credit = rng.choice(
        ["Conso", "Auto", "Immobilier", "Equipement", "Tresorerie"],
        size=n_credits,
        p=[0.32, 0.24, 0.18, 0.16, 0.10],
    )

    amount_ranges = {
        "Conso": (5000, 85000),
        "Auto": (30000, 240000),
        "Immobilier": (180000, 1200000),
        "Equipement": (15000, 160000),
        "Tresorerie": (8000, 120000),
    }
    duration_ranges = {
        "Conso": (6, 72),
        "Auto": (12, 84),
        "Immobilier": (60, 120),
        "Equipement": (12, 84),
        "Tresorerie": (6, 48),
    }

    amounts = []
    durations = []
    for credit_type in type_credit:
        low, high = amount_ranges[credit_type]
        d_low, d_high = duration_ranges[credit_type]
        amounts.append(rng.integers(low, high + 1))
        durations.append(rng.integers(d_low, d_high + 1))

    amount_array = np.array(amounts, dtype=float)
    duration_array = np.array(durations, dtype=float)
    mensualite = (amount_array / duration_array) * rng.uniform(1.05, 1.22, size=n_credits)
    revenu = selected_clients["revenu_mensuel"].to_numpy(dtype=float)
    extra_debt = rng.uniform(0, 0.18, size=n_credits) * revenu
    taux_endettement = ((mensualite + extra_debt) / revenu * 100).round(2)
    taux_endettement = np.clip(taux_endettement, 5, 92).round(2)

    amount_income_ratio = amount_array / np.maximum(revenu, 1)
    sector_risk = selected_clients["secteur_activite"].isin(["Independant", "Commerce", "Agriculture"]).astype(float)
    young_or_senior = selected_clients["age_group"].isin(["18-25", "60+"]).astype(float)
    debt_pressure = np.maximum(0, taux_endettement - 35) / 20

    # Intercept calibré pour un taux de défaut ~7%, proche des portefeuilles retail réels
    # (shift de -1.80 par rapport à l'intercept initial -3.25 qui produisait ~31%)
    logits = (
        -5.05
        + 1.05 * debt_pressure
        + 0.45 * (duration_array > 72)
        + 0.45 * (type_credit == "Conso")
        + 0.55 * (revenu < 4500)
        + 0.38 * sector_risk
        + 0.32 * young_or_senior
        + 0.18 * np.minimum(amount_income_ratio / 10, 2)
        + rng.normal(0, 0.38, size=n_credits)
    )
    pd_default = 1 / (1 + np.exp(-logits))
    default_flag = rng.binomial(1, pd_default)

    credits = pd.DataFrame(
        {
            "credit_id": [f"CR{i:06d}" for i in range(1, n_credits + 1)],
            "client_id": selected_clients["client_id"].to_numpy(),
            "montant_credit": amount_array.round(0).astype(int),
            "duree_mois": duration_array.astype(int),
            "mensualite": mensualite.round(2),
            "taux_endettement": taux_endettement,
            "type_credit": type_credit,
            "date_demande": _date_series(rng, n_credits),
            "default_flag": default_flag,
        }
    )

    idx = rng.choice(credits.index, size=28, replace=False)
    credits.loc[idx[:6], "montant_credit"] = -credits.loc[idx[:6], "montant_credit"].abs()
    credits.loc[idx[6:12], "taux_endettement"] = rng.uniform(105, 180, size=6).round(2)
    credits.loc[idx[12:18], "duree_mois"] = rng.choice([2, 3, 4, 140], size=6)
    credits.loc[idx[18:23], "client_id"] = "UNKNOWN_CLIENT"
    credits.loc[idx[23:28], "mensualite"] = -credits.loc[idx[23:28], "mensualite"].abs()
    duplicate_rows = credits.sample(6, random_state=RANDOM_SEED + 1)
    credits = pd.concat([credits, duplicate_rows], ignore_index=True)
    return credits


def generate_payments(rng: np.random.Generator, credits: pd.DataFrame) -> pd.DataFrame:
    payment_rows = []
    payment_id = 1
    usable_credits = credits.drop_duplicates("credit_id").copy()

    for _, credit in usable_credits.iterrows():
        months = int(min(max(credit.get("duree_mois", 6), 3), rng.integers(4, 15)))
        date_start = pd.to_datetime(credit["date_demande"], errors="coerce")
        if pd.isna(date_start):
            date_start = pd.Timestamp("2024-01-01")

        base_late_prob = min(max(float(credit.get("taux_endettement", 30)) / 140, 0.05), 0.55)
        if int(credit.get("default_flag", 0)) == 1:
            status_probs = [max(0.25, 1 - base_late_prob - 0.25), min(0.48, base_late_prob), 0.20]
        else:
            status_probs = [max(0.70, 1 - base_late_prob), min(0.25, base_late_prob), 0.03]
        status_probs = np.array(status_probs) / np.sum(status_probs)

        for month in range(1, months + 1):
            due_date = date_start + pd.DateOffset(months=month)
            status = rng.choice(["A_JOUR", "RETARD", "IMPAYE"], p=status_probs)
            if status == "A_JOUR":
                days_late = 0
                amount_paid = float(credit["mensualite"])
                pay_date = due_date
            elif status == "RETARD":
                days_late = int(rng.integers(5, 61))
                amount_paid = float(credit["mensualite"])
                pay_date = due_date + pd.to_timedelta(days_late, unit="D")
            else:
                days_late = int(rng.integers(35, 121))
                amount_paid = 0.0
                pay_date = pd.NaT

            payment_rows.append(
                {
                    "payment_id": f"P{payment_id:08d}",
                    "credit_id": credit["credit_id"],
                    "date_echeance": due_date.normalize(),
                    "date_paiement": pay_date.normalize() if not pd.isna(pay_date) else pd.NaT,
                    "montant_du": round(float(credit["mensualite"]), 2),
                    "montant_paye": round(amount_paid, 2),
                    "jours_retard": days_late,
                    "statut_paiement": status,
                }
            )
            payment_id += 1

    payments = pd.DataFrame(payment_rows)
    idx = rng.choice(payments.index, size=24, replace=False)
    payments.loc[idx[:8], "statut_paiement"] = "STATUT_INVALIDE"
    payments.loc[idx[8:16], "date_paiement"] = pd.to_datetime(payments.loc[idx[8:16], "date_echeance"]) - pd.Timedelta(days=5)
    payments.loc[idx[16:24], "montant_du"] = -payments.loc[idx[16:24], "montant_du"].abs()
    duplicate_rows = payments.sample(8, random_state=RANDOM_SEED + 2)
    payments = pd.concat([payments, duplicate_rows], ignore_index=True)
    return payments


def generate_data(config: GenerationConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or GenerationConfig()
    ensure_directories()
    rng = np.random.default_rng(config.seed)

    clients = generate_clients(rng, config.clients)
    credits = generate_credits(rng, clients, config.credits)
    payments = generate_payments(rng, credits)

    clients.to_csv(RAW_DIR / "clients.csv", index=False)
    credits.to_csv(RAW_DIR / "credit_applications.csv", index=False)
    payments.to_csv(RAW_DIR / "payments.csv", index=False)

    clients.head(100).to_csv(SAMPLE_DIR / "clients_sample.csv", index=False)
    credits.head(100).to_csv(SAMPLE_DIR / "credit_applications_sample.csv", index=False)
    payments.head(100).to_csv(SAMPLE_DIR / "payments_sample.csv", index=False)

    return {"clients": clients, "credits": credits, "payments": payments}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fictive credit risk data.")
    parser.add_argument("--clients", type=int, default=GenerationConfig.clients)
    parser.add_argument("--credits", type=int, default=GenerationConfig.credits)
    args = parser.parse_args()
    generate_data(GenerationConfig(clients=args.clients, credits=args.credits))
    print("Raw and sample data generated.")


if __name__ == "__main__":
    main()

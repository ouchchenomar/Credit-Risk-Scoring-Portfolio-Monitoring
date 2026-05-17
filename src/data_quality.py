from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR, ensure_directories


ANOMALY_COLUMNS = [
    "run_id",
    "table_name",
    "business_key",
    "rule_name",
    "anomaly_type",
    "severity",
    "detected_at",
]


def _append_anomalies(
    rows: list[dict[str, str]],
    *,
    run_id: str,
    table_name: str,
    keys: pd.Series,
    mask: pd.Series,
    rule_name: str,
    anomaly_type: str,
    severity: str,
    detected_at: pd.Timestamp,
) -> None:
    bad_keys = keys[mask.fillna(True)]
    for key in bad_keys.astype(str).tolist():
        rows.append(
            {
                "run_id": run_id,
                "table_name": table_name,
                "business_key": key,
                "rule_name": rule_name,
                "anomaly_type": anomaly_type,
                "severity": severity,
                "detected_at": detected_at.isoformat(),
            }
        )


def run_data_quality() -> dict[str, pd.DataFrame]:
    ensure_directories()
    run_id = str(uuid4())[:8]
    detected_at = pd.Timestamp.utcnow()

    clients = pd.read_csv(RAW_DIR / "clients.csv")
    credits = pd.read_csv(RAW_DIR / "credit_applications.csv")
    payments = pd.read_csv(RAW_DIR / "payments.csv")

    rows: list[dict[str, str]] = []

    client_id_missing = clients["client_id"].isna() | (clients["client_id"].astype(str).str.strip() == "")
    client_id_duplicate = clients["client_id"].notna() & clients["client_id"].duplicated(keep=False)
    _append_anomalies(
        rows,
        run_id=run_id,
        table_name="clients",
        keys=clients["client_id"].fillna("MISSING_CLIENT_ID"),
        mask=client_id_missing,
        rule_name="client_id_obligatoire",
        anomaly_type="missing_key",
        severity="High",
        detected_at=detected_at,
    )
    _append_anomalies(
        rows,
        run_id=run_id,
        table_name="clients",
        keys=clients["client_id"].fillna("MISSING_CLIENT_ID"),
        mask=client_id_duplicate,
        rule_name="client_id_unique",
        anomaly_type="duplicate_key",
        severity="High",
        detected_at=detected_at,
    )

    clean_clients = clients.loc[~client_id_missing].drop_duplicates("client_id", keep="first").copy()
    valid_client_ids = set(clean_clients["client_id"].astype(str))

    credit_id_missing = credits["credit_id"].isna() | (credits["credit_id"].astype(str).str.strip() == "")
    credit_id_duplicate = credits["credit_id"].notna() & credits["credit_id"].duplicated(keep=False)
    client_missing = ~credits["client_id"].astype(str).isin(valid_client_ids)
    amount_invalid = pd.to_numeric(credits["montant_credit"], errors="coerce").fillna(-1) <= 0
    monthly_invalid = pd.to_numeric(credits["mensualite"], errors="coerce").fillna(-1) <= 0
    debt_invalid = ~pd.to_numeric(credits["taux_endettement"], errors="coerce").between(0, 100, inclusive="both")
    duration_invalid = ~pd.to_numeric(credits["duree_mois"], errors="coerce").between(6, 120, inclusive="both")
    type_invalid = ~credits["type_credit"].isin(["Conso", "Auto", "Immobilier", "Equipement", "Tresorerie"])

    credit_rules = [
        (credit_id_missing, "credit_id_obligatoire", "missing_key", "High"),
        (credit_id_duplicate, "credit_id_unique", "duplicate_key", "High"),
        (client_missing, "client_id_present_dans_clients", "foreign_key", "High"),
        (amount_invalid, "montant_credit_positif", "invalid_amount", "High"),
        (monthly_invalid, "mensualite_positive", "invalid_amount", "High"),
        (debt_invalid, "taux_endettement_coherent", "invalid_ratio", "High"),
        (duration_invalid, "duree_valide", "invalid_duration", "Medium"),
        (type_invalid, "type_credit_autorise", "invalid_category", "Medium"),
    ]
    for mask, rule_name, anomaly_type, severity in credit_rules:
        _append_anomalies(
            rows,
            run_id=run_id,
            table_name="credit_applications",
            keys=credits["credit_id"].fillna("MISSING_CREDIT_ID"),
            mask=mask,
            rule_name=rule_name,
            anomaly_type=anomaly_type,
            severity=severity,
            detected_at=detected_at,
        )

    invalid_credits = np.logical_or.reduce(
        [
            credit_id_missing.to_numpy(),
            credit_id_duplicate.to_numpy(),
            client_missing.to_numpy(),
            amount_invalid.to_numpy(),
            monthly_invalid.to_numpy(),
            debt_invalid.to_numpy(),
            duration_invalid.to_numpy(),
            type_invalid.to_numpy(),
        ]
    )
    clean_credits = credits.loc[~invalid_credits].drop_duplicates("credit_id", keep="first").copy()
    valid_credit_ids = set(clean_credits["credit_id"].astype(str))

    payment_id_missing = payments["payment_id"].isna() | (payments["payment_id"].astype(str).str.strip() == "")
    payment_id_duplicate = payments["payment_id"].notna() & payments["payment_id"].duplicated(keep=False)
    credit_missing = ~payments["credit_id"].astype(str).isin(valid_credit_ids)
    payment_status_invalid = ~payments["statut_paiement"].isin(["A_JOUR", "RETARD", "IMPAYE"])
    due_amount_invalid = pd.to_numeric(payments["montant_du"], errors="coerce").fillna(-1) <= 0
    paid_amount_invalid = pd.to_numeric(payments["montant_paye"], errors="coerce").fillna(-1) < 0
    late_days_invalid = pd.to_numeric(payments["jours_retard"], errors="coerce").fillna(-1) < 0

    date_due = pd.to_datetime(payments["date_echeance"], errors="coerce")
    date_paid = pd.to_datetime(payments["date_paiement"], errors="coerce")
    paid_rows = payments["date_paiement"].notna() & (payments["date_paiement"].astype(str).str.strip() != "")
    date_invalid = date_due.isna() | (paid_rows & (date_paid < date_due))

    payment_rules = [
        (payment_id_missing, "payment_id_obligatoire", "missing_key", "High"),
        (payment_id_duplicate, "payment_id_unique", "duplicate_key", "High"),
        (credit_missing, "credit_id_present_dans_credits", "foreign_key", "High"),
        (payment_status_invalid, "statut_paiement_autorise", "invalid_category", "Medium"),
        (due_amount_invalid, "montant_du_positif", "invalid_amount", "High"),
        (paid_amount_invalid, "montant_paye_non_negatif", "invalid_amount", "High"),
        (late_days_invalid, "jours_retard_non_negatif", "invalid_delay", "High"),
        (date_invalid, "dates_paiement_coherentes", "invalid_date", "Medium"),
    ]
    for mask, rule_name, anomaly_type, severity in payment_rules:
        _append_anomalies(
            rows,
            run_id=run_id,
            table_name="payments",
            keys=payments["payment_id"].fillna("MISSING_PAYMENT_ID"),
            mask=mask,
            rule_name=rule_name,
            anomaly_type=anomaly_type,
            severity=severity,
            detected_at=detected_at,
        )

    invalid_payments = np.logical_or.reduce(
        [
            payment_id_missing.to_numpy(),
            payment_id_duplicate.to_numpy(),
            credit_missing.to_numpy(),
            payment_status_invalid.to_numpy(),
            due_amount_invalid.to_numpy(),
            paid_amount_invalid.to_numpy(),
            late_days_invalid.to_numpy(),
            date_invalid.to_numpy(),
        ]
    )
    clean_payments = payments.loc[~invalid_payments].drop_duplicates("payment_id", keep="first").copy()

    anomalies = pd.DataFrame(rows, columns=ANOMALY_COLUMNS)
    anomalies.to_csv(PROCESSED_DIR / "dq_anomalies.csv", index=False)

    clean_clients.to_csv(PROCESSED_DIR / "clients_clean.csv", index=False)
    clean_credits.to_csv(PROCESSED_DIR / "credit_applications_clean.csv", index=False)
    clean_payments.to_csv(PROCESSED_DIR / "payments_clean.csv", index=False)

    report_rows = []
    source_tables = {
        "clients": clients,
        "credit_applications": credits,
        "payments": payments,
    }
    for table_name, table_df in source_tables.items():
        table_anomalies = anomalies.loc[anomalies["table_name"] == table_name]
        anomaly_rate = len(table_anomalies) / max(len(table_df), 1)
        report_rows.append(
            {
                "table_name": table_name,
                "rows_checked": len(table_df),
                "anomalies_detected": len(table_anomalies),
                "anomaly_rate": round(anomaly_rate, 4),
                "quality_score": round(max(0, 100 * (1 - anomaly_rate)), 2),
            }
        )
    report = pd.DataFrame(report_rows)
    report.to_csv(PROCESSED_DIR / "data_quality_report.csv", index=False)

    return {
        "clients_clean": clean_clients,
        "credit_applications_clean": clean_credits,
        "payments_clean": clean_payments,
        "dq_anomalies": anomalies,
        "data_quality_report": report,
    }


def main() -> None:
    outputs = run_data_quality()
    print(
        "Data quality completed: "
        f"{len(outputs['dq_anomalies'])} anomalies, "
        f"{len(outputs['credit_applications_clean'])} clean credits."
    )


if __name__ == "__main__":
    main()


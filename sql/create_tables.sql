CREATE TABLE clients (
    client_id VARCHAR(20) PRIMARY KEY,
    age_group VARCHAR(20),
    revenu_mensuel NUMERIC(12, 2),
    ville VARCHAR(80),
    secteur_activite VARCHAR(80),
    anciennete_emploi NUMERIC(6, 2)
);

CREATE TABLE credit_applications (
    credit_id VARCHAR(20) PRIMARY KEY,
    client_id VARCHAR(20) REFERENCES clients(client_id),
    montant_credit NUMERIC(14, 2),
    duree_mois INTEGER,
    mensualite NUMERIC(12, 2),
    taux_endettement NUMERIC(8, 2),
    type_credit VARCHAR(50),
    date_demande DATE,
    default_flag INTEGER
);

CREATE TABLE payments (
    payment_id VARCHAR(20) PRIMARY KEY,
    credit_id VARCHAR(20) REFERENCES credit_applications(credit_id),
    date_echeance DATE,
    date_paiement DATE,
    montant_du NUMERIC(12, 2),
    montant_paye NUMERIC(12, 2),
    jours_retard INTEGER,
    statut_paiement VARCHAR(30)
);

CREATE TABLE dq_anomalies (
    run_id VARCHAR(40),
    table_name VARCHAR(80),
    business_key VARCHAR(120),
    rule_name VARCHAR(120),
    anomaly_type VARCHAR(80),
    severity VARCHAR(20),
    detected_at TIMESTAMP
);

CREATE TABLE scoring_results (
    client_id VARCHAR(20),
    credit_id VARCHAR(20),
    probability_default NUMERIC(8, 4),
    risk_class VARCHAR(20),
    recommendation VARCHAR(120),
    score_date DATE,
    model_version VARCHAR(40)
);

CREATE TABLE model_monitoring (
    model_version VARCHAR(40),
    scoring_month VARCHAR(7),
    auc_score NUMERIC(8, 4),
    precision_score NUMERIC(8, 4),
    recall_score NUMERIC(8, 4),
    f1_score NUMERIC(8, 4),
    default_rate_observed NUMERIC(8, 4),
    default_rate_predicted NUMERIC(8, 4),
    population_stability_index NUMERIC(8, 4),
    created_at TIMESTAMP
);


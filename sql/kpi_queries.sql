-- Taux de defaut global
SELECT
    AVG(default_flag) AS default_rate
FROM credit_applications;

-- Encours a risque eleve ou critique
SELECT
    SUM(ca.montant_credit) AS encours_a_risque
FROM credit_applications ca
JOIN scoring_results sr ON sr.credit_id = ca.credit_id
WHERE sr.risk_class IN ('Eleve', 'Critique');

-- Score moyen par type de credit
SELECT
    ca.type_credit,
    AVG(sr.probability_default) AS score_moyen,
    COUNT(*) AS credits
FROM credit_applications ca
JOIN scoring_results sr ON sr.credit_id = ca.credit_id
GROUP BY ca.type_credit
ORDER BY score_moyen DESC;

-- Anomalies par regle
SELECT
    rule_name,
    severity,
    COUNT(*) AS anomalies
FROM dq_anomalies
GROUP BY rule_name, severity
ORDER BY anomalies DESC;

-- Monitoring mensuel du modele
SELECT
    scoring_month,
    auc_score,
    precision_score,
    recall_score,
    default_rate_observed,
    default_rate_predicted,
    population_stability_index
FROM model_monitoring
ORDER BY scoring_month;


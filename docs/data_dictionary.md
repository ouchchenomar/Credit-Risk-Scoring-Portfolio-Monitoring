# Data Dictionary

## clients

| Colonne | Description |
| --- | --- |
| `client_id` | Identifiant fictif du client |
| `age_group` | Tranche d'age |
| `revenu_mensuel` | Revenu mensuel simule |
| `ville` | Ville du client |
| `secteur_activite` | Secteur professionnel |
| `anciennete_emploi` | Anciennete d'emploi en annees |

## credit_applications

| Colonne | Description |
| --- | --- |
| `credit_id` | Identifiant fictif du credit |
| `client_id` | Identifiant client |
| `montant_credit` | Montant finance |
| `duree_mois` | Duree du credit en mois |
| `mensualite` | Mensualite simulee |
| `taux_endettement` | Charge de dette estimee |
| `type_credit` | Type de credit |
| `date_demande` | Date de demande |
| `default_flag` | Cible de defaut simulee |

## payments

| Colonne | Description |
| --- | --- |
| `payment_id` | Identifiant du paiement |
| `credit_id` | Identifiant credit |
| `date_echeance` | Date d'echeance |
| `date_paiement` | Date de paiement |
| `montant_du` | Montant attendu |
| `montant_paye` | Montant paye |
| `jours_retard` | Nombre de jours de retard |
| `statut_paiement` | `A_JOUR`, `RETARD` ou `IMPAYE` |

## scoring_results

| Colonne | Description |
| --- | --- |
| `probability_default` | Probabilite estimee de defaut |
| `risk_class` | Classe de risque |
| `recommendation` | Action metier recommandee |
| `score_date` | Date de scoring |
| `model_version` | Version du modele |

## model_monitoring

| Colonne | Description |
| --- | --- |
| `auc_score` | Pouvoir discriminant mensuel |
| `precision_score` | Precision mensuelle |
| `recall_score` | Recall mensuel |
| `f1_score` | F1-score mensuel |
| `default_rate_observed` | Taux de defaut observe |
| `default_rate_predicted` | Taux de defaut predit |
| `population_stability_index` | Derive de population simplifiee |


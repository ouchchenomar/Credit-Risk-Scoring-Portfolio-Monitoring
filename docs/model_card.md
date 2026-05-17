# Model Card

## Objectif

Estimer une probabilite de defaut pour chaque demande de credit fictive, puis classer le dossier en risque faible, moyen, eleve ou critique.

## Donnees

Les donnees sont entierement fictives. Elles simulent des profils clients, des demandes de credit et des historiques de paiement. Aucune donnee personnelle reelle n'est utilisee.

## Cible

`default_flag = 1` si le credit presente un profil de defaut simule. `default_flag = 0` sinon.

## Modeles compares

- Logistic Regression: baseline interpretable.
- Random Forest: modele non lineaire pour comparaison de performance.

## Metriques

- AUC.
- Precision.
- Recall.
- F1-score.
- PR-AUC.
- Matrice de confusion.

## Utilisation prevue

Le modele sert d'outil d'aide a l'analyse et au pilotage. Il ne doit pas etre utilise pour une decision automatique reelle.

## Limites

- Donnees synthetiques.
- Pas de validation externe.
- Pas de fairness testing complet.
- Les seuils de risque sont pedagogiques.
- Une validation humaine reste obligatoire pour tout usage metier.

## Monitoring

Le monitoring mensuel suit la performance, le taux observe vs predit et un PSI simplifie pour reperer une derive de population.


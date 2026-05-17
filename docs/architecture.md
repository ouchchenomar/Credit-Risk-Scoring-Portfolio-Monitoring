# Architecture

Le projet suit un flux simple et reproductible:

1. `src/data_generation.py` cree des donnees fictives.
2. `src/data_quality.py` controle les donnees et produit `dq_anomalies.csv`.
3. `src/preprocessing.py` construit la table `features_credit_risk.csv`.
4. `src/train_model.py` entraine et sauvegarde le modele.
5. `src/score_clients.py` calcule le score et la classe de risque.
6. `src/monitoring.py` suit la performance mensuelle.
7. `dashboard/app.py` expose les KPI et les analyses.

## Couche data

- `data/raw`: donnees brutes fictives.
- `data/processed`: donnees nettoyees, features, scoring, monitoring.
- `data/sample`: echantillons partageables.

## Couche modele

Le modele sauvegarde dans `models/credit_score_model.pkl` contient:

- le pipeline de preprocessing;
- le classifieur retenu;
- les colonnes de features;
- la version du modele;
- le seuil de decision utilise pour le monitoring.

## Couche pilotage

Le dashboard est volontairement separe de la logique ML. Il lit les fichiers produits par le pipeline pour garder une separation claire entre production des donnees et restitution metier.


# Notebooks

La V1 est script-first pour rester reproductible avec `python -m src.run_pipeline`.

Ordre recommande pour creer les notebooks portfolio:

1. `01_data_exploration.ipynb`: exploration du portefeuille, distributions et taux de defaut.
2. `02_feature_engineering.ipynb`: construction et justification des variables.
3. `03_model_training.ipynb`: comparaison Logistic Regression vs Random Forest.
4. `04_model_monitoring.ipynb`: performance mensuelle, taux observe vs predit et PSI.

Les scripts `src/` contiennent deja la logique executable qui peut etre reprise dans ces notebooks.


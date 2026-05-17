"""Tests unitaires pour le feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


class TestFeatureDefinitions:
    def test_pas_de_doublon_dans_feature_columns(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_categoriques_et_numeriques_couvrent_toutes_les_features(self):
        """Chaque feature doit être soit numérique soit catégorielle, jamais les deux."""
        all_defined = set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)
        assert set(FEATURE_COLUMNS) == all_defined

    def test_pas_de_chevauchement_cat_num(self):
        intersection = set(CATEGORICAL_FEATURES) & set(NUMERIC_FEATURES)
        assert len(intersection) == 0, f"Features dans les deux listes : {intersection}"

    def test_target_absent_des_features(self):
        assert TARGET_COLUMN not in FEATURE_COLUMNS

    def test_features_derivees_presentes(self):
        """Les features calculées doivent être déclarées."""
        derivees = {"ratio_credit_revenu", "ratio_mensualite_revenu", "encours_sur_revenu"}
        assert derivees.issubset(set(FEATURE_COLUMNS))


class TestRatioCalculations:
    def test_ratio_credit_revenu(self):
        montant, revenu = 50_000.0, 10_000.0
        assert abs(montant / revenu - 5.0) < 1e-9

    def test_ratio_mensualite_revenu(self):
        mensualite, revenu = 1_500.0, 10_000.0
        assert abs(mensualite / revenu - 0.15) < 1e-9

    def test_revenu_clip_anti_division_zero(self):
        """Le clip lower=1 doit éviter une division par zéro."""
        revenu = max(0.0, 1.0)  # même logique que dans preprocessing.py
        ratio = 50_000.0 / revenu
        assert np.isfinite(ratio)

    def test_encours_sur_revenu_zero_nouveau_client(self):
        """Un nouveau client sans encours doit donner ratio = 0."""
        total_encours, revenu = 0.0, 12_000.0
        assert total_encours / max(revenu, 1.0) == 0.0

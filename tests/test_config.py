"""Tests unitaires pour les fonctions de configuration et de classification du risque."""
import pytest
from src.config import risk_class, risk_recommendation, RISK_THRESHOLDS


class TestRiskClass:
    def test_faible(self):
        assert risk_class(0.0) == "Faible"
        assert risk_class(0.05) == "Faible"
        assert risk_class(0.09) == "Faible"

    def test_moyen(self):
        assert risk_class(RISK_THRESHOLDS["faible"]) == "Moyen"   # frontière basse
        assert risk_class(0.15) == "Moyen"
        assert risk_class(0.24) == "Moyen"

    def test_eleve(self):
        assert risk_class(RISK_THRESHOLDS["moyen"]) == "Eleve"    # frontière basse
        assert risk_class(0.35) == "Eleve"
        assert risk_class(0.49) == "Eleve"

    def test_critique(self):
        assert risk_class(RISK_THRESHOLDS["eleve"]) == "Critique"  # frontière basse
        assert risk_class(0.75) == "Critique"
        assert risk_class(1.0) == "Critique"

    def test_toutes_valeurs_couvertes(self):
        """Aucune valeur de PD entre 0 et 1 ne doit tomber hors des 4 classes."""
        import numpy as np
        classes_valides = {"Faible", "Moyen", "Eleve", "Critique"}
        for pd_val in np.linspace(0.0, 1.0, 50):
            assert risk_class(float(pd_val)) in classes_valides

    def test_frontiere_faible_moyen(self):
        """La frontière 0.10 appartient à Moyen (borne inférieure inclusive)."""
        assert risk_class(0.10) == "Moyen"

    def test_frontiere_moyen_eleve(self):
        assert risk_class(0.25) == "Eleve"

    def test_frontiere_eleve_critique(self):
        assert risk_class(0.50) == "Critique"


class TestRiskRecommendation:
    def test_mapping_complet(self):
        assert "standard" in risk_recommendation("Faible").lower()
        assert "surveiller" in risk_recommendation("Moyen").lower()
        assert "manuel" in risk_recommendation("Eleve").lower()
        assert "escalade" in risk_recommendation("Critique").lower()

    def test_classe_inconnue(self):
        """Un classe non reconnue doit retourner un fallback non vide."""
        result = risk_recommendation("INEXISTANT")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_toutes_classes_ont_recommendation(self):
        for classe in ["Faible", "Moyen", "Eleve", "Critique"]:
            rec = risk_recommendation(classe)
            assert isinstance(rec, str) and len(rec) > 3

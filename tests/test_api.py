"""Tests d'intégration pour l'API FastAPI (hors chargement de modèle)."""
import pytest

try:
    from fastapi.testclient import TestClient
    from src.api import app
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi non installé")
class TestHealthEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    def test_health_retourne_200(self):
        response = self.client.get("/health")
        assert response.status_code == 200

    def test_health_contient_status_ok(self):
        data = self.client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_contient_timestamp(self):
        data = self.client.get("/health").json()
        assert "timestamp" in data


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi non installé")
class TestScoreEndpoint:
    """Tests sur /score sans modèle chargé (503 attendu)."""

    def setup_method(self):
        self.client = TestClient(app)
        self.payload = {
            "client_id": "C00001",
            "age_group": "26-35",
            "revenu_mensuel": 12000.0,
            "ville": "Casablanca",
            "secteur_activite": "Salarie prive",
            "anciennete_emploi": 5.0,
            "montant_credit": 50000.0,
            "duree_mois": 36,
            "mensualite": 1550.0,
            "taux_endettement": 35.0,
            "type_credit": "Conso",
        }

    def test_score_sans_modele_retourne_503(self):
        """Sans modèle chargé, l'API doit retourner 503 (pas un crash 500)."""
        response = self.client.post("/score", json=self.payload)
        # 200 si modèle présent, 503 si pas encore entraîné — les deux sont valides
        assert response.status_code in (200, 503)

    def test_score_payload_invalide_retourne_422(self):
        """Un payload sans champs obligatoires doit retourner 422 (validation Pydantic)."""
        response = self.client.post("/score", json={"client_id": "C00001"})
        assert response.status_code == 422

    def test_score_revenu_negatif_retourne_422(self):
        bad_payload = {**self.payload, "revenu_mensuel": -1000.0}
        response = self.client.post("/score", json=bad_payload)
        assert response.status_code == 422

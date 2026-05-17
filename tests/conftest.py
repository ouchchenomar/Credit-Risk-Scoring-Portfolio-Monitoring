"""Configuration pytest partagée entre tous les modules de test."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marque les tests lents (pipeline complet, entraînement modèle)",
    )

"""Tests unitaires pour la génération de données synthétiques."""
import numpy as np
import pandas as pd
import pytest

from src.data_generation import GenerationConfig, generate_clients, generate_credits, generate_payments


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def clients_df(rng):
    return generate_clients(rng, 200)


@pytest.fixture(scope="module")
def credits_df(rng, clients_df):
    return generate_credits(rng, clients_df, 300)


class TestClients:
    def test_colonnes_attendues(self, clients_df):
        expected = {"client_id", "age_group", "revenu_mensuel", "ville", "secteur_activite", "anciennete_emploi"}
        assert expected.issubset(set(clients_df.columns))

    def test_revenus_dans_les_bornes(self, clients_df):
        valid = clients_df["revenu_mensuel"].dropna()
        assert (valid >= 2200).all(), "Revenu < 2 200 DH détecté"
        assert (valid <= 52000).all(), "Revenu > 52 000 DH détecté"

    def test_anciennete_non_negative(self, clients_df):
        assert (clients_df["anciennete_emploi"].dropna() >= 0).all()

    def test_age_groups_valides(self, clients_df):
        valides = {"18-25", "26-35", "36-45", "46-60", "60+"}
        assert clients_df["age_group"].dropna().isin(valides).all()

    def test_villes_valides(self, clients_df):
        valides = {"Casablanca", "Rabat", "Marrakech", "Fes", "Tanger", "Agadir", "Oujda"}
        assert clients_df["ville"].dropna().isin(valides).all()


class TestCredits:
    def test_colonnes_attendues(self, credits_df):
        expected = {
            "credit_id", "client_id", "montant_credit", "duree_mois",
            "mensualite", "taux_endettement", "type_credit", "date_demande", "default_flag",
        }
        assert expected.issubset(set(credits_df.columns))

    def test_types_credit_valides(self, credits_df):
        valides = {"Conso", "Auto", "Immobilier", "Equipement", "Tresorerie"}
        clean = credits_df["type_credit"].dropna()
        assert clean.isin(valides).all()

    def test_default_flag_binaire(self, credits_df):
        valid_flags = credits_df["default_flag"].dropna()
        assert set(valid_flags.unique()).issubset({0, 1})

    def test_taux_defaut_realiste(self, credits_df):
        """Après fix intercept, le taux doit être ≤ 20% (cible ~7%)."""
        clean = credits_df.drop_duplicates("credit_id").dropna(subset=["default_flag"])
        rate = clean["default_flag"].mean()
        assert rate <= 0.20, f"Taux de défaut trop élevé : {rate:.1%} (max attendu 20%)"

    def test_anomalies_injectees_presentes(self, credits_df):
        """Les anomalies intentionnelles doivent exister pour valider le module DQ."""
        assert (credits_df["montant_credit"] < 0).any(), "Aucun montant négatif trouvé"
        assert (credits_df["taux_endettement"] > 100).any(), "Aucun taux > 100% trouvé"


class TestPayments:
    def test_colonnes_attendues(self, rng, credits_df):
        payments = generate_payments(rng, credits_df)
        expected = {
            "payment_id", "credit_id", "date_echeance", "montant_du",
            "montant_paye", "jours_retard", "statut_paiement",
        }
        assert expected.issubset(set(payments.columns))

    def test_statuts_valides_et_invalides(self, rng, credits_df):
        payments = generate_payments(rng, credits_df)
        """Les anomalies injectent des statuts invalides volontairement."""
        all_statuts = set(payments["statut_paiement"].dropna().unique())
        valides = {"A_JOUR", "RETARD", "IMPAYE"}
        # Il doit y avoir au moins un statut invalide (anomalie injectée)
        assert not all_statuts.issubset(valides), "Aucun statut invalide trouvé — anomalies non injectées"

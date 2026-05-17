from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

RANDOM_SEED = 42
MODEL_VERSION = "credit-risk-v1.0"

RISK_THRESHOLDS = {
    "faible": 0.10,
    "moyen": 0.25,
    "eleve": 0.50,
}


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def risk_class(probability_default: float) -> str:
    if probability_default < RISK_THRESHOLDS["faible"]:
        return "Faible"
    if probability_default < RISK_THRESHOLDS["moyen"]:
        return "Moyen"
    if probability_default < RISK_THRESHOLDS["eleve"]:
        return "Eleve"
    return "Critique"


def risk_recommendation(risk: str) -> str:
    recommendations = {
        "Faible": "Suivi standard",
        "Moyen": "Dossier a surveiller",
        "Eleve": "Controle manuel recommande",
        "Critique": "Escalade risque prioritaire",
    }
    return recommendations.get(risk, "Analyse manuelle")


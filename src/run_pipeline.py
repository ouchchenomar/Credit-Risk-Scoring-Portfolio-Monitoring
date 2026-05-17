from __future__ import annotations

from .data_generation import generate_data
from .data_quality import run_data_quality
from .monitoring import monitor_model
from .preprocessing import build_features
from .score_clients import score_clients
from .train_model import train_model


def main() -> None:
    print("1/6 Generating fictive data...")
    generate_data()
    print("2/6 Running data quality controls...")
    run_data_quality()
    print("3/6 Building model features...")
    build_features()
    print("4/6 Training scoring model...")
    train_model()
    print("5/6 Scoring credit applications...")
    score_clients()
    print("6/6 Creating model monitoring tables...")
    monitor_model()
    print("Pipeline completed.")


if __name__ == "__main__":
    main()


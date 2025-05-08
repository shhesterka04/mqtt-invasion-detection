"""Centralised configuration using pydantic.BaseSettings."""
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Global settings (overridable via env vars or .env file).
    model_path can be passed to Trainer/CLI.
    """

    # Dataset root containing *.csv generated from pcap
    data_root: Path = Path("data")

    # Default output directory for results / checkpoints
    results_root: Path = Path("results")

    # Random seed to make experiments reproducible
    seed: int = 42

    # Feature level: packet | uniflow | biflow
    mode: str = "biflow"

    # List of ML/DL models to train
    models: List[str] = ["lr", "rf", "mlp"]

    # CV folds (0 = train‑test split)
    cv_folds: int = 0

    # DL hyper‑params
    epochs: int = 20
    batch_size: int = 64

    model_config = SettingsConfigDict(env_prefix="MQTTIDS_", env_file=".env")


cfg = Config()  # singleton style

"""MQTT‑IDS package.
Collects public API and lazy‑loads heavy modules only when accessed.
"""
from importlib import import_module as _imp

__all__ = [
    "Config",
    "load_dataset",
    "preprocess",
    "build_pipeline",
    "Trainer",
    "save_metrics",
    "evaluate",
]

from .config import Config  # noqa: E402
from .data.loader import load_dataset  # noqa: E402
from .preprocessing import preprocess  # noqa: E402
from .pipeline import build_pipeline  # noqa: E402
from .train import Trainer  # noqa: E402
from .metrics import save_metrics, evaluate  # noqa: E402

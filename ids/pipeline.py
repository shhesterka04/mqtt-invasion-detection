"""Build sklearn Pipeline combining preprocessor and estimator.
Supports classical ML models only (mlp via sklearn)."""
from __future__ import annotations

from typing import Dict
from sklearn.pipeline import Pipeline

from .preprocessing import make_preprocessor
from .models.classical import get_classical_models


def build_pipeline(model_key: str, X_sample) -> Pipeline:
    """Return a Pipeline for the given model_key and sample data."""
    # Preprocessor based on sample
    prep = make_preprocessor(X_sample)

    model_dict: Dict[str, object] = get_classical_models()
    if model_key not in model_dict:
        raise ValueError(f"Unknown model key {model_key}")

    estimator = model_dict[model_key]
    return Pipeline([("prep", prep), ("clf", estimator)])
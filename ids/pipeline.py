# ids/pipeline.py
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np, scipy.sparse as sp

from .preprocessing     import make_preprocessor
from .models.classical  import get_classical_models

_DENSE_MODELS = {"nb"}        

def _to_dense(x):
    """CSR → numpy.ndarray (для NB, etc.)."""
    return x.toarray() if sp.issparse(x) else x

def build_pipeline(
    model_key: str,
    X_sample : pd.DataFrame,
    y_sample : Optional[np.ndarray] = None,
) -> Pipeline:
    prep = make_preprocessor(X_sample.copy())
    model_dict: Dict[str, object] = get_classical_models()

    if model_key not in model_dict:
        raise ValueError(f"Unknown model key '{model_key}'")

    estimator = model_dict[model_key]

    if callable(estimator) and y_sample is not None:
        pos_w = (y_sample == 0).sum() / max((y_sample == 1).sum(), 1)
        estimator = estimator(pos_w)

    steps = [("prep", prep)]

    if model_key in _DENSE_MODELS:
        steps.append(("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)))

    steps.append(("clf", estimator))
    return Pipeline(steps)

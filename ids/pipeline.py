from __future__ import annotations
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .preprocessing     import make_preprocessor
from .models.classical  import get_classical_models
from .models.dl         import get_dl_models
from .config            import cfg

_DENSE_MODELS = {"nb"}
_DL_MODEL_KEYS = {"mlp_dl", "cnn_1d", "bilstm"}

def _to_dense(x):
    return x.toarray() if sp.issparse(x) else x

def build_pipeline(
    model_key: str,
    X_sample: pd.DataFrame,
    y_sample: Optional[np.ndarray] = None,
) -> Union[Pipeline, tf.keras.Model]:
    """Returns sklearn Pipeline or tf.keras.Model depending on model_key."""

    if model_key in _DL_MODEL_KEYS:
        input_dim = X_sample.shape[1]
        return get_dl_models(input_dim)[model_key]

    prep = make_preprocessor(X_sample.copy())
    steps = [("prep", prep)]

    model_dict: Dict[str, object] = get_classical_models()
    if model_key not in model_dict:
        raise ValueError(f"Unknown model key '{model_key}'")

    estimator = model_dict[model_key]

    if callable(estimator) and y_sample is not None:
        pos_w = (y_sample == 0).sum() / max((y_sample == 1).sum(), 1)
        estimator = estimator(pos_w)

    if model_key in _DENSE_MODELS:
        steps.append(("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)))

    steps.append(("clf", estimator))
    return Pipeline(steps)

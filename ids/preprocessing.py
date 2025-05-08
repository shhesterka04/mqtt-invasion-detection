"""Dynamic preprocessing: One‑Hot categorical + StandardScaler numeric."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    cat_cols = df.select_dtypes(include=["object", "category"], exclude=["bool"]).columns
    num_cols = df.select_dtypes(exclude=["object", "category"], include=[float, int]).columns
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )


def preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, pd.DataFrame]:
    prep = make_preprocessor(X)
    X_proc = prep.fit_transform(X)
    return prep, X_proc
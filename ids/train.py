"""Trainer class (supports CV or simple split with GroupKFold)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from .config import cfg, Config
from .metrics import save_metrics
from .pipeline import build_pipeline
from .utils import LOGGER, seed_everything, timing

class Trainer:
    def __init__(self, conf: Config | None = None):
        self.conf = conf or cfg
        seed_everything(self.conf.seed)

    def run(self, X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray] = None) -> None:
        """Train & evaluate. Use train-test split if cv_folds=0, else GroupKFold by flow."""
        out_root = self.conf.results_root
        out_root.mkdir(parents=True, exist_ok=True)
        model_keys: List[str] = self.conf.models

        if self.conf.cv_folds <= 0:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=self.conf.seed
            )
            for mk in model_keys:
                self._fit_save(mk, X_tr, y_tr, X_te, y_te, out_root, suffix="_fold0")
        else:
            if groups is None:
                raise ValueError("`groups` must be provided for GroupKFold CV.")
            gkf = GroupKFold(n_splits=self.conf.cv_folds)
            for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
                X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                for mk in model_keys:
                    self._fit_save(mk, X_tr, y_tr, X_te, y_te, out_root, suffix=f"_fold{fold_idx}")

    def _fit_save(
        self,
        model_key: str,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_te: pd.DataFrame,
        y_te: np.ndarray,
        out_dir: Path,
        suffix: str,
    ) -> None:
        """Train single model, predict, and save metrics."""
        LOGGER.info("[%s%s] training…", model_key, suffix)
        with timing(model_key):
            pipe = build_pipeline(model_key, X_tr)
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
            save_metrics(y_te, preds, out_dir / f"{model_key}{suffix}")
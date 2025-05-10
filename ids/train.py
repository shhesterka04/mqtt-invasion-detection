"""Trainer class (CV / split by StratifiedGroupKFold)."""
from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

from .config   import cfg, Config
from .metrics  import save_metrics
from .pipeline import build_pipeline
from .utils    import LOGGER, seed_everything, timing


class Trainer:
    def __init__(self, conf: Config | None = None):
        self.conf = conf or cfg
        seed_everything(self.conf.seed)

    def run(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        flow_groups:   np.ndarray | None = None, 
        attack_groups: np.ndarray | None = None, 
    ) -> None:
        """Train / eval with simple split or StratifiedGroupKFold."""
        out_root = self.conf.results_root
        out_root.mkdir(parents=True, exist_ok=True)
        model_keys: List[str] = self.conf.models

        if self.conf.cv == 0:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=self.conf.seed
            )
            self._loop(model_keys, X_tr, y_tr, X_te, y_te, out_root, suffix="_fold0")
        elif self.conf.cv == "loso":
            if attack_groups is None or flow_groups is None:
                raise ValueError("Need attack_groups and flow_groups for LOSO")
        else:
            if flow_groups is None:
                raise ValueError("flow_groups required for k‑fold")
            sgkf = StratifiedGroupKFold(
                n_splits=int(self.conf.cv), shuffle=True,
                random_state=self.conf.seed
            )
            for fold_idx, (tr_idx, te_idx) in enumerate(
                    sgkf.split(X, y, groups=flow_groups)):
                X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                self._loop(
                    model_keys, X_tr, y_tr, X_te, y_te, out_root,
                    suffix=f"_fold{fold_idx}"
                )

    def _loop(
        self,
        model_keys: List[str],
        X_tr: pd.DataFrame, y_tr: np.ndarray,
        X_te: pd.DataFrame, y_te: np.ndarray,
        out_dir: Path, suffix: str,
    ):
        for mk in model_keys:
            self._fit_save(mk, X_tr, y_tr, X_te, y_te, out_dir, suffix)

    def _fit_save(
        self,
        model_key: str,
        X_tr: pd.DataFrame, y_tr: np.ndarray,
        X_te: pd.DataFrame, y_te: np.ndarray,
        out_dir: Path, suffix: str,
    ) -> None:
        LOGGER.info("[%s%s] training…", model_key, suffix)
        with timing(model_key):
            pipe = build_pipeline(model_key, X_tr, y_tr)
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
            save_metrics(y_te, preds, out_dir / f"{model_key}{suffix}")

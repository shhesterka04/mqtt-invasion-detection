"""Compute and save metrics + quick `evaluate()` helper for notebooks/CLI."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Dict

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

__all__ = ["save_metrics", "evaluate"]

def _metric_dict(y_true, y_pred) -> Dict[str, float]:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": prec,
        "recall": rec,
        "f1_macro": f1,
    }


def save_metrics(y_true, y_pred, out_base: Path):
    """Save confusion‑matrix CSV + JSON report with aggregated metrics."""
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(out_base.with_name(out_base.name + "_cm.csv"), cm, fmt="%d", delimiter=",")
    report = _metric_dict(y_true, y_pred)
    out_base.with_name(out_base.name + "_report.json").write_text(json.dumps(report, indent=2))


def evaluate(y_true, y_pred) -> Dict[str, float]:
    """Return metrics dict without touching filesystem (useful in notebooks)."""
    return _metric_dict(y_true, y_pred)
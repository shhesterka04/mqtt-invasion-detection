"""Data loading utilities with optional grouping support.

This loader supports both single CSV files and directories of CSVs.
It concatenates all CSVs and returns features, labels, and flow groups:

* If a file contains an `is_attack` column, that column is popped and used as `y`.
* Otherwise labels are derived from the filename: `normal` → 0, else → 1.

Grouping:
* Computes `flow_id` by factorizing the tuple of (ip_src, ip_dst, prt_src, prt_dst).
* Drops identifier columns afterwards to prevent leakage.

Returns
-------
X : pd.DataFrame
    Concatenated features without identifier or `is_attack` columns.
y : np.ndarray
    1-D array of labels (0=benign,1=attack).
groups : np.ndarray
    1-D array of integer flow IDs for each record.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, List, Union

import pandas as pd
import numpy as np

__all__ = ["load_dataset"]

_ATTACK_RE = re.compile(r"attack|bruteforce|scan|sparta", re.I)
_NORMAL_RE = re.compile(r"normal", re.I)


def _derive_label_from_name(name: str) -> int:
    if _NORMAL_RE.search(name):
        return 0
    if _ATTACK_RE.search(name):
        return 1
    return 1


def _read_single_csv(path: Path) -> pd.DataFrame:
    """Read one CSV into a DataFrame, retaining identifier columns and `is_attack` if present."""
    df = pd.read_csv(path)
    return df


def load_dataset(source: Union[str, Path]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load data and return (X, y, groups)."""
    source = Path(source)
    # Collect CSV files
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.glob("*.csv"))
    else:
        raise FileNotFoundError(f"Source {source} is not a file or directory")

    if not paths:
        raise ValueError(f"No CSV files found in {source}")

    df_list: List[pd.DataFrame] = []
    y_list: List[np.ndarray] = []
    scenario_list: List[np.ndarray] = []
    for idx, path in enumerate(paths):
        df = _read_single_csv(path)
        # Extract labels
        if 'is_attack' in df.columns:
            y = df.pop('is_attack').astype(int).values
        else:
            y = np.full(len(df), _derive_label_from_name(path.stem), dtype=int)
        df_list.append(df)
        y_list.append(y)
        # Scenario grouping: each file gets a unique group id
        scenario_list.append(np.full(len(df), idx, dtype=int))

    # Concatenate all frames and labels
    df_all = pd.concat(df_list, ignore_index=True)
    y_all = np.concatenate(y_list, axis=0)
    groups = np.concatenate(scenario_list, axis=0)

    # Drop identifier columns to prevent leakage
    id_cols = ['ip_src', 'ip_dst', 'prt_src', 'prt_dst']
    X = df_all.drop(columns=id_cols, errors='ignore')

    return X, y_all, groups
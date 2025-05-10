# ids/data/loader.py
from __future__ import annotations
import re, numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple, Union, List

__all__ = ["load_dataset"]

_ATTACK_RE  = re.compile(r"attack|bruteforce|scan|sparta", re.I)
_NORMAL_RE  = re.compile(r"normal", re.I)

def _derive_label_from_name(name: str) -> int:
    return 0 if _NORMAL_RE.search(name) else 1

def _read_single_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def load_dataset(source: Union[str, Path], add_noise: float = 0.0, flip_prob: float = 0.10,
) -> Tuple[pd.DataFrame,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X                : DataFrame (без id‑колонок)
    y                : ndarray  (0/1)
    flow_groups      : ndarray  (id потока)
    scenario_groups  : ndarray  (id файла‑сценария)
    attack_groups    : ndarray  (scenario_id для attack, −1 для benign)
    """
    source = Path(source)
    paths  = [source] if source.is_file() else sorted(source.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {source}")

    df_list, y_list, scen_list = [], [], []

    for idx, p in enumerate(paths):
        df = _read_single_csv(p)

        if "is_attack" in df.columns:
            y_raw = pd.to_numeric(df.pop("is_attack"), errors="coerce")
            mask  = y_raw.notna()
            y     = y_raw[mask].astype(int).values
            df    = df[mask].reset_index(drop=True)
        else:
            y = np.full(len(df), _derive_label_from_name(p.stem), dtype=int)

        df_list.append(df)
        y_list.append(y)
        scen_list.append(np.full(len(df), idx, dtype=int))

    df_all           = pd.concat(df_list, ignore_index=True)
    y_all            = np.concatenate(y_list,   axis=0)
    scenario_groups  = np.concatenate(scen_list, axis=0)

    flow_cols = ["ip_src", "ip_dst", "prt_src", "prt_dst"]
    tuples    = pd.Series(list(zip(*(df_all[c] for c in flow_cols))))
    df_all["flow_id"] = tuples.factorize()[0]
    flow_groups = df_all["flow_id"].to_numpy()
    attack_groups = np.where(y_all == 1, scenario_groups, -1)

    X = df_all.drop(columns=flow_cols + ["flow_id"], errors="ignore")

    # 1) Label‑noise  ─────────────────────────────────────────
    if flip_prob > 0:
        flip = np.random.rand(len(y_all)) < flip_prob

        y_all[flip] = 1 - y_all[flip]
    # 2) Numeric noise  σ = add_noise · std(col) ─────────────
    if add_noise > 0:
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                v      = pd.to_numeric(X[c], errors="coerce")
                std    = v.std(ddof=0)
                noise  = np.random.normal(0, add_noise * std, size=len(v))
                v      = v.add(noise, fill_value=np.nan)
                X[c]   = v

    # 3) Drop «strong» features  (регулярку можно расширить) ─
    DROP = [col for col in X.columns if
            col.endswith("_sum") or col.endswith("_bytes")]
    X.drop(columns=DROP, errors="ignore", inplace=True)

    # 4) Огрубляем значения  → целые (исчезают «мелкие» различия)
    num_cols = X.select_dtypes("number").columns
    X[num_cols] = X[num_cols].round(0)

    return X, y_all, flow_groups, scenario_groups, attack_groups

#!/usr/bin/env python3
# ids/cli/classification.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence

from ids.config          import Config
from ids.data.loader     import load_dataset
from ids.train           import Trainer
from ids.models.classical import get_classical_models

MODE_MAP = {0: "packet", 1: "uniflow", 2: "biflow"}

DL_MODEL_KEYS = ["mlp_dl", "cnn_1d", "bilstm"]   
ALL_MODELS = set(get_classical_models().keys()) | set(DL_MODEL_KEYS)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="classification",
        description="Training / evaluation for MQTT‑IDS",
    )
    ap.add_argument("--data", default="data",
        help="CSV‑файл или директория с CSV‑файлами")
    ap.add_argument("--mode", type=int, choices=[0, 1, 2], default=2,
        help="0 packet, 1 uniflow, 2 biflow")
    ap.add_argument("--models", nargs="+", default=["lr"],
        help=f"модели: {', '.join(sorted(ALL_MODELS))}")
    ap.add_argument("--cv", default="0",
        help="k (целое) для StratifiedGroupKFold или 'loso'")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", type=int, default=1)
    return ap.parse_args(argv)

def main() -> None:
    args = parse_args()

    bad = [m for m in args.models if m not in ALL_MODELS]
    if bad:
        raise ValueError(f"Неизвестные модели: {', '.join(bad)}")

    X, y, flow_g, scen_g, atk_g = load_dataset(Path(args.data), add_noise=0.1)

    conf = Config(
        data_root   = Path(args.data),
        results_root= Path(args.output),
        mode        = MODE_MAP[args.mode],
        models      = args.models,
        cv          = args.cv if args.cv == "loso" else int(args.cv),
        epochs      = args.epochs,
        seed        = args.seed,
    )

    Trainer(conf).run(X, y, flow_groups=flow_g, attack_groups=atk_g)

    print("\n[DEBUG] Files in", conf.results_root)
    for p in sorted(conf.results_root.rglob("*")):
        if p.is_file():
            print("  └─", p.relative_to(conf.results_root))

if __name__ == "__main__":
    main()

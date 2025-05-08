"""CLI entry‑point compatible with original script.
Now supports `--data DIR` so you may pass a directory containing many
CSV‑files (e.g. `biflow_normal.csv`, `biflow_scan_A.csv`, …). The loader
concatenates them automatically.

Example
-------
    python3.10 classification.py \
        --data biflow_features \
        --mode 2 \
        --models lr rf mlp \
        --cv 3 --epochs 15 --verbose 1 --output results

Note: *не пишите комментарий после обратного слеша* – иначе zsh посчитает
строку новой командой.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ids.config import Config
from ids.data.loader import load_dataset
from ids.train import Trainer


MODE_MAP = {0: "packet", 1: "uniflow", 2: "biflow"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="CSV file or directory with many CSVs")
    ap.add_argument("--mode", type=int, choices=[0, 1, 2], default=2,
                    help="0 packet, 1 uniflow, 2 biflow")
    ap.add_argument("--models", nargs="*", default=["lr"], help="lr rf knn svm mlp …")
    ap.add_argument("--cv", type=int, default=0, help="k‑folds (0 = simple split)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output", default="results")
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1) load data (directory or single CSV)
    X, y, groups = load_dataset(Path(args.data))

    # 2) build runtime config for Trainer
    conf = Config(
        data_root=Path(args.data),
        results_root=Path(args.output),
        mode=MODE_MAP[args.mode],
        models=args.models,
        cv_folds=args.cv,
        epochs=args.epochs,
        seed=args.seed,
    )

    # 3) train & evaluate
    trainer = Trainer(conf=conf)
    trainer.run(X, y, groups)

    # Debug: list result files and print first lines of each
    print("[DEBUG] Listing all files in results directory:")
    results_path = Path(args.output)
    if results_path.exists():
        for path in sorted(results_path.rglob('*')):
            if path.is_file():
                rel = path.relative_to(results_path)
                print(f"--- {rel} ---")
                try:
                    lines = path.read_text().splitlines()
                    for ln in lines[:5]:
                        print(ln)
                    if len(lines) > 5:
                        print("...")
                except Exception as e:
                    print(f"[ERROR] Cannot read {rel}: {e}")
    else:
        print(f"[DEBUG] No results directory found at {results_path}")


if __name__ == "__main__":
    main()

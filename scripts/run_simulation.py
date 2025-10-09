# scripts/run_simulation.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from biocircuitai.config import CONFIG
from biocircuitai.simulate.experiments import sweep, default_sweep

def parse_args():
    p = argparse.ArgumentParser(description="Run parameter sweep and write CSV.")
    p.add_argument("--out", type=str, default=str(CONFIG.paths["processed"] / "toggle_dataset.csv"),
                   help="Output CSV path.")
    p.add_argument("--use-defaults", action="store_true",
                   help="Use CONFIG.grid as-is (recommended).")
    p.add_argument("--rows", type=int, default=0,
                   help="If >0, take only the first N rows (debug).")
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.use_defaults:
        df = default_sweep()
    else:
        # Example: you could tweak CONFIG.grid here if desired, but default is best for v0.
        df = default_sweep()

    if args.rows and args.rows > 0:
        df = df.head(args.rows)

    df.to_csv(out, index=False)
    print(f"[OK] Wrote sweep with {len(df):,} rows → {out}")

if __name__ == "__main__":
    main()

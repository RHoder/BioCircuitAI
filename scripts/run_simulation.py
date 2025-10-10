# scripts/run_simulation.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from biocircuitai.config import CONFIG
from biocircuitai.simulate.experiments import default_sweep

def parse_args():
    p = argparse.ArgumentParser(description="Run parameter sweep and write CSV.")
    p.add_argument("--out", type=str, default=str(CONFIG.paths["processed"] / "toggle_dataset.csv"),
                   help="Output CSV path.")
    p.add_argument("--rows", type=int, default=0, help="If >0, keep only first N rows (debug).")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes.")
    p.add_argument("--chunksize", type=int, default=1000, help="Job submission batch size.")
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = default_sweep(workers=args.workers, chunksize=args.chunksize)

    if args.rows and args.rows > 0:
        df = df.head(args.rows)

    df.to_csv(out, index=False)
    print(f"[OK] Wrote sweep with {len(df):,} rows → {out}")

if __name__ == "__main__":
    main()

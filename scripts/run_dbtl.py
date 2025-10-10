# scripts/run_dbtl.py
from __future__ import annotations
import argparse
from biocircuitai.control.scheduler import DBTLConfig, run_dbtl_sync

def parse_args():
    p = argparse.ArgumentParser(description="Run full DBTL orchestration.")
    p.add_argument("--outdir", type=str, default="data/dbtl/")
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--targetA", type=float, default=10.0)
    p.add_argument("--targetB", type=float, default=8.0)
    p.add_argument("--hidden", type=str, default="", help="optional '256,256'; blank = infer")
    return p.parse_args()

def main():
    a = parse_args()
    hidden = tuple(int(x) for x in a.hidden.split(",") if x.strip()) if a.hidden else None
    cfg = DBTLConfig(outdir=a.outdir, trials=a.trials, targetA=a.targetA, targetB=a.targetB, hidden=hidden)
    outdir = run_dbtl_sync(cfg)
    print(f"[OK] DBTL artifacts in: {outdir}")

if __name__ == "__main__":
    main()

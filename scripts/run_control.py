# scripts/run_control.py
from __future__ import annotations
import argparse
import json
import os

from biocircuitai.config import CONFIG
from biocircuitai.control.controllers import (
    ClosedLoopConfig, PIDGains, run_closed_loop, plot_closed_loop
)

def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop control demo for the toggle switch.")
    p.add_argument("--targetA", type=float, default=CONFIG.target["A"])
    p.add_argument("--targetB", type=float, default=CONFIG.target["B"])
    p.add_argument("--T", type=float, default=200.0, help="Total horizon")
    p.add_argument("--dt_ctrl", type=float, default=2.0, help="Control update period")
    p.add_argument("--KpA", type=float, default=0.2)
    p.add_argument("--KiA", type=float, default=0.01)
    p.add_argument("--KdA", type=float, default=0.0)
    p.add_argument("--KpB", type=float, default=0.2)
    p.add_argument("--KiB", type=float, default=0.01)
    p.add_argument("--KdB", type=float, default=0.0)
    p.add_argument("--umin", type=float, default=-0.9)
    p.add_argument("--umax", type=float, default=3.0)
    p.add_argument("--outdir", type=str, default="data/control/")
    return p.parse_args()

def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cfg = ClosedLoopConfig(
        params=CONFIG.defaults,
        targetA=a.targetA,
        targetB=a.targetB,
        T=a.T,
        dt_ctrl=a.dt_ctrl,
        dt_ode=a.dt_ctrl,
        pidA=PIDGains(a.KpA, a.KiA, a.KdA),
        pidB=PIDGains(a.KpB, a.KiB, a.KdB),
        umin=a.umin, umax=a.umax,
        y0=(0.1, 0.1),
    )
    res = run_closed_loop(cfg)
    # save artifacts
    import numpy as np
    import pandas as pd
    traj_csv = os.path.join(a.outdir, "closed_loop_trajectory.csv")
    pd.DataFrame({"t": res["t"], "A": res["Y"][:,0], "B": res["Y"][:,1]}).to_csv(traj_csv, index=False)
    u_csv = os.path.join(a.outdir, "closed_loop_controls.csv")
    pd.DataFrame(res["U"], columns=["uA","uB"]).to_csv(u_csv, index=False)
    fig_png = os.path.join(a.outdir, "closed_loop.png")
    plot_closed_loop(res, fig_png)

    # simple summary
    finalA = float(res["Y"][-1,0]); finalB = float(res["Y"][-1,1])
    summary = {
        "target": {"A": a.targetA, "B": a.targetB},
        "final": {"A": finalA, "B": finalB},
        "abs_error": {"A": abs(finalA - a.targetA), "B": abs(finalB - a.targetB)},
        "gains": {"A": [a.KpA, a.KiA, a.KdA], "B": [a.KpB, a.KiB, a.KdB]},
        "umin": a.umin, "umax": a.umax, "dt_ctrl": a.dt_ctrl, "T": a.T,
    }
    sum_json = os.path.join(a.outdir, "summary.json")
    with open(sum_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Closed-loop complete → {a.outdir}")
    print(f" Final A={finalA:.3f}, B={finalB:.3f} vs targets A={a.targetA}, B={a.targetB}")

if __name__ == "__main__":
    main()

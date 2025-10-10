# scripts/optimize_circuit.py
from __future__ import annotations
import argparse, json, os
import numpy as np
from biocircuitai.config import CONFIG
from biocircuitai.surrogate.evaluate import load_model
from biocircuitai.surrogate.dataset import PARAM_COLS
from biocircuitai.control.optimizer import (
    SurrogateWrapper, BOConfig, bo_optimize, validate_on_ode,
    plot_convergence, plot_timeseries
)

def parse_args():
    p = argparse.ArgumentParser(description="Optimize circuit parameters via surrogate BO, then validate on ODE.")
    p.add_argument("--model", type=str, default="data/models/surrogate.pt")
    p.add_argument("--csv", type=str, default="data/processed/toggle_dataset.csv")
    p.add_argument("--targetA", type=float, default=CONFIG.target["A"])
    p.add_argument("--targetB", type=float, default=CONFIG.target["B"])
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--outdir", type=str, default="data/opt/")
    p.add_argument("--hidden", type=str, default="", help="optional, e.g., '256,256'; if omitted we'll infer from checkpoint")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # load trained model (hidden inferred if not provided)
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip()) if args.hidden else None
    model, device, used_hidden = load_model(args.model, hidden=hidden)
    print(f"[INFO] Loaded model on {device} with hidden={used_hidden}")

    # wrap surrogate
    sur = SurrogateWrapper(model, device)

    # run BO
    cfg = BOConfig(target={"A": args.targetA, "B": args.targetB}, bounds=CONFIG.bounds, n_trials=args.trials)
    best, history = bo_optimize(sur, cfg)
    x_best = np.array([best.params[k] if k in best.params else best.params[PARAM_COLS[i]] for i,k in enumerate(PARAM_COLS)], dtype=float)
    print(f"[OK] BO best loss={best.value:.4f}")
    print("[OK] Best params:", {k: float(v) for k, v in zip(PARAM_COLS, x_best)})

    # validate on true ODE
    val = validate_on_ode(x_best)
    print(f"[VAL] true steady_A={val['steady_A']:.3f}, steady_B={val['steady_B']:.3f}  (targets: A={args.targetA}, B={args.targetB})")

    # save artifacts
    with open(os.path.join(args.outdir, "best_result.json"), "w") as f:
        json.dump({
            "best_loss": float(best.value),
            "best_params": {k: float(v) for k, v in zip(PARAM_COLS, x_best)},
            "true_steady_A": float(val["steady_A"]),
            "true_steady_B": float(val["steady_B"]),
            "target": {"A": args.targetA, "B": args.targetB},
            "hidden": used_hidden
        }, f, indent=2)

    plot_convergence(history, os.path.join(args.outdir, "bo_convergence.png"))
    plot_timeseries(val["t"], val["Y"], os.path.join(args.outdir, "true_trajectory.png"))
    print(f"[OK] Artifacts → {args.outdir}")

if __name__ == "__main__":
    main()

# scripts/run_dbtl.py
from __future__ import annotations

import argparse
import json
import os
import uuid
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from biocircuitai.config import CONFIG
from biocircuitai.io.logging import get_logger
from biocircuitai.io.storage import RunRecord, register
from biocircuitai.simulate.experiments import default_sweep
from biocircuitai.surrogate.train import TrainConfig, train_surrogate
from biocircuitai.surrogate.evaluate import load_model
from biocircuitai.surrogate.dataset import PARAM_COLS
from biocircuitai.control.optimizer import (
    SurrogateWrapper,
    BOConfig,
    bo_optimize,
    validate_on_ode,
    plot_convergence,
    plot_timeseries,
)

log = get_logger("biocircuitai.dbtl")

def _parse_targets_inline(s: str) -> List[Dict[str, float]]:
    """
    Parse --targets "A:B:label, A:B, ..." (label optional).
    Returns list of dicts: {"A": float, "B": float, "label": str}
    """
    out = []
    if not s or not s.strip():
        return out
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(":")]
        if len(parts) == 2:
            A, B = float(parts[0]), float(parts[1])
            out.append({"A": A, "B": B, "label": f"A{A}_B{B}"})
        elif len(parts) == 3:
            A, B = float(parts[0]), float(parts[1])
            label = parts[2]
            out.append({"A": A, "B": B, "label": label})
        else:
            raise ValueError(f"Bad target spec '{chunk}'. Use 'A:B' or 'A:B:label'.")
    return out

def _load_targets_json(path: str) -> List[Dict[str, float]]:
    with open(path, "r") as f:
        raw = json.load(f)
    out = []
    for item in raw:
        out.append({
            "A": float(item["A"]),
            "B": float(item["B"]),
            "label": item.get("label", f"A{item['A']}_B{item['B']}"),
        })
    return out

def parse_args():
    p = argparse.ArgumentParser(
        description="Full DBTL: sweep → train → multi-target BO → true ODE validate."
    )
    # Output layout
    p.add_argument("--outdir", type=str, default="data/dbtl/", help="Top-level output directory")
    # Sweep controls
    p.add_argument("--csv", type=str, default="", help="Use existing dataset CSV (skip sweep) if provided")
    p.add_argument("--workers", type=int, default=1, help="Parallel sweep workers (Windows-safe)")
    p.add_argument("--chunksize", type=int, default=1000, help="Sweep job submission batch size")
    # Training controls
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=str, default="", help="MLP sizes, e.g. '256,256'. Blank: infer or default.")
    # BO controls (shared for all targets)
    p.add_argument("--trials", type=int, default=80, help="Optuna trials per target")
    # Multi-target spec
    p.add_argument("--targets", type=str, default="", help="Inline list 'A:B[:label], A:B[:label], ...'")
    p.add_argument("--targets_json", type=str, default="", help="Path to JSON list of {A,B,label}")
    # Single-target fallback (used only if no multi-targets specified)
    p.add_argument("--targetA", type=float, default=CONFIG.target["A"])
    p.add_argument("--targetB", type=float, default=CONFIG.target["B"])
    return p.parse_args()

def main():
    args = parse_args()
    run_id = uuid.uuid4().hex[:8]
    outdir = os.path.join(args.outdir, run_id)
    os.makedirs(outdir, exist_ok=True)
    log.info(f"DBTL run_id={run_id} → {outdir}")

    # ------------- Targets -------------
    targets: List[Dict[str, float]] = []
    targets += _parse_targets_inline(args.targets)
    if args.targets_json:
        targets += _load_targets_json(args.targets_json)
    if not targets:
        # single-target fallback
        targets = [{"A": float(args.targetA), "B": float(args.targetB), "label": f"A{args.targetA}_B{args.targetB}"}]
    log.info(f"Targets: {', '.join([t['label'] for t in targets])}")

    # ------------- 1) Sweep (or reuse) -------------
    try:
        if args.csv:
            csv_path = args.csv
            log.info(f"Phase: sweep (skipped) using existing CSV → {csv_path}")
        else:
            log.info(f"Phase: sweep (workers={args.workers}, chunksize={args.chunksize})")
            df = default_sweep(workers=args.workers, chunksize=args.chunksize)
            csv_path = os.path.join(outdir, "toggle_dataset.csv")
            df.to_csv(csv_path, index=False)
            log.info(f"Saved dataset → {csv_path} ({len(df)} rows)")
        register(RunRecord(run_id, "sweep", "OK", {"csv": csv_path}, "dataset ready"))
    except Exception as e:
        register(RunRecord(run_id, "sweep", "FAIL", {}, str(e)))
        raise

    # ------------- 2) Train (normalized) -------------
    try:
        log.info(f"Phase: train (epochs={args.epochs}, bs={args.bs}, lr={args.lr})")
        model_path = os.path.join(outdir, "surrogate.pt")
        scalers_path = os.path.join(outdir, "scalers.joblib")
        hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip()) if args.hidden else (256, 256)

        tcfg = TrainConfig(
            csv_path=csv_path,
            batch_size=args.bs,
            lr=args.lr,
            epochs=args.epochs,
            val_split=0.2,
            hidden=hidden,
            dropout=0.0,
            model_out=model_path,
            scalers_out=scalers_path,
            normalize=True,
        )
        model, best_val = train_surrogate(tcfg)
        register(RunRecord(run_id, "train", "OK",
                           {"model": model_path, "scalers": scalers_path, "val_mse": round(float(best_val), 6)},
                           "training completed"))
        log.info(f"Saved model → {model_path}; scalers → {scalers_path}; best val MSE={best_val:.6f}")
    except Exception as e:
        register(RunRecord(run_id, "train", "FAIL", {}, str(e)))
        raise

    # Load model once; wrapper re-used across targets
    model, device, used_hidden = load_model(model_path, hidden=hidden)
    scalers_for_wrapper = scalers_path if os.path.exists(scalers_path) else None
    sur = SurrogateWrapper(model, device, scalers_for_wrapper)

    # Combined results table across targets
    rows = []

    # ------------- 3) BO + 4) Validate for each target -------------
    for t in targets:
        label = t["label"]
        tdir = os.path.join(outdir, label)
        os.makedirs(tdir, exist_ok=True)
        log.info(f"Phase: optimize+validate for target '{label}' (A={t['A']}, B={t['B']})")

        # --- BO ---
        try:
            bocfg = BOConfig(target={"A": t["A"], "B": t["B"]},
                             bounds=CONFIG.bounds, n_trials=args.trials)
            best, history = bo_optimize(sur, bocfg)
            x_best = np.array([best.params[k] for k in PARAM_COLS], dtype=float)

            # Save BO artifacts
            with open(os.path.join(tdir, "bo_best.json"), "w") as f:
                json.dump({
                    "label": label,
                    "target": {"A": t["A"], "B": t["B"]},
                    "best_loss": float(best.value),
                    "best_params": {k: float(v) for k, v in zip(PARAM_COLS, x_best)},
                    "hidden": used_hidden
                }, f, indent=2)
            plot_convergence(history, os.path.join(tdir, "bo_convergence.png"))
            register(RunRecord(run_id, f"bo:{label}", "OK",
                               {"best_loss": float(best.value)},
                               "bayesian optimization completed"))
            log.info(f"[{label}] BO best loss={best.value:.4f}")
        except Exception as e:
            register(RunRecord(run_id, f"bo:{label}", "FAIL", {}, str(e)))
            raise

        # --- Validate on true ODE ---
        try:
            val = validate_on_ode(x_best)
            plot_timeseries(val["t"], val["Y"], os.path.join(tdir, "true_trajectory.png"))
            with open(os.path.join(tdir, "validate_result.json"), "w") as f:
                json.dump({
                    "label": label,
                    "target": {"A": t["A"], "B": t["B"]},
                    "true_steady_A": float(val["steady_A"]),
                    "true_steady_B": float(val["steady_B"])
                }, f, indent=2)
            register(RunRecord(run_id, f"validate:{label}", "OK",
                               {"true_A": float(val["steady_A"]), "true_B": float(val["steady_B"])},
                               "validation completed"))
            log.info(f"[{label}] Validation: A={val['steady_A']:.3f}, B={val['steady_B']:.3f}")
        except Exception as e:
            register(RunRecord(run_id, f"validate:{label}", "FAIL", {}, str(e)))
            raise

        # Row for combined summary
        rows.append({
            "label": label,
            "target_A": t["A"],
            "target_B": t["B"],
            "best_loss": float(best.value),
            "true_steady_A": float(val["steady_A"]),
            "true_steady_B": float(val["steady_B"]),
            **{f"param_{k}": float(v) for k, v in zip(PARAM_COLS, x_best)},
        })

    # Write combined summary at run root
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "summary.csv"), index=False)

    log.info(f"DBTL complete → {outdir}")
    print(f"[OK] DBTL artifacts in: {outdir}")
    print(f"[OK] Targets: {', '.join([r['label'] for r in rows])}")

if __name__ == "__main__":
    main()

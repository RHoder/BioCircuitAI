# src/biocircuitai/control/multitarget.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import numpy as np
import torch

from ..config import CONFIG
from ..surrogate.evaluate import load_model
from ..surrogate.dataset import PARAM_COLS
from .optimizer import (
    SurrogateWrapper,
    BOConfig,
    bo_optimize,
    validate_on_ode,
    plot_convergence,
    plot_timeseries,
)

@dataclass
class TargetSpec:
    A: float
    B: float
    label: str | None = None  # optional name for folders/rows

def run_multi_target(
    model_path: str = "data/models/surrogate.pt",
    outdir: str = "data/multitarget/",
    targets: Iterable[TargetSpec] = (TargetSpec(10.0, 8.0, "default"),),
    trials: int = 80,
    hidden: Tuple[int, ...] | None = None,
) -> List[Dict]:
    """
    Runs BO+validation for each target in `targets`.
    Returns a list of result dicts; also writes artifacts under outdir/<label>/.
    """
    os.makedirs(outdir, exist_ok=True)

    # Load model once; wrap surrogate (auto-uses scalers if present)
    model, device, used_hidden = load_model(model_path, hidden=hidden)
    scalers_path = os.path.join(os.path.dirname(model_path), "scalers.joblib")
    sur = SurrogateWrapper(model, device, scalers_path if os.path.exists(scalers_path) else None)

    results: List[Dict] = []
    for t in targets:
        label = t.label or f"A{t.A}_B{t.B}"
        t_out = os.path.join(outdir, label)
        os.makedirs(t_out, exist_ok=True)

        # Run BO on the surrogate
        bocfg = BOConfig(target={"A": t.A, "B": t.B}, bounds=CONFIG.bounds, n_trials=trials)
        best, history = bo_optimize(sur, bocfg)

        # Extract best x in correct order
        x_best = np.array([best.params[k] for k in PARAM_COLS], dtype=float)

        # Validate on the true ODE
        val = validate_on_ode(x_best)

        # Save JSON + plots
        with open(os.path.join(t_out, "result.json"), "w") as f:
            json.dump({
                "label": label,
                "target": {"A": t.A, "B": t.B},
                "best_loss": float(best.value),
                "best_params": {k: float(v) for k, v in zip(PARAM_COLS, x_best)},
                "true_steady_A": float(val["steady_A"]),
                "true_steady_B": float(val["steady_B"]),
                "hidden": used_hidden,
            }, f, indent=2)

        plot_convergence(history, os.path.join(t_out, "bo_convergence.png"))
        plot_timeseries(val["t"], val["Y"], os.path.join(t_out, "true_trajectory.png"))

        results.append({
            "label": label,
            "target_A": t.A,
            "target_B": t.B,
            "best_loss": float(best.value),
            "true_steady_A": float(val["steady_A"]),
            "true_steady_B": float(val["steady_B"]),
            **{f"param_{k}": float(v) for k, v in zip(PARAM_COLS, x_best)},
        })

    # Write a combined CSV summary
    import pandas as pd
    pd.DataFrame(results).to_csv(os.path.join(outdir, "summary.csv"), index=False)

    return results

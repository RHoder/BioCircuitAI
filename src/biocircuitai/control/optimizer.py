# src/biocircuitai/control/optimizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import json, os
import numpy as np
import optuna
import torch
import matplotlib.pyplot as plt

from ..config import CONFIG
from ..simulate.models import toggle_switch_ode
from ..simulate.ode_solver import integrate, summarize_timeseries
from ..surrogate.dataset import PARAM_COLS, TARGET_COLS
from ..surrogate.evaluate import load_model  # uses hidden-size inference

# ---------- Surrogate wrapper ----------

class SurrogateWrapper:
    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device

    def predict(self, x_vec: np.ndarray) -> np.ndarray:
        """x_vec: shape (8,) in PARAM_COLS order → returns (2,) steady_A, steady_B"""
        x = torch.tensor(x_vec[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.model(x).detach().cpu().numpy()[0]
        return y


# ---------- Optimization ----------

@dataclass
class BOConfig:
    target: Dict[str, float] = None
    bounds: Dict[str, Tuple[float, float]] = None
    n_trials: int = 80
    seed: int = 42

def bo_optimize(sur: SurrogateWrapper, cfg: BOConfig):
    """Bayesian optimization on the surrogate to hit target steady states."""
    if cfg.target is None:
        cfg.target = CONFIG.target
    if cfg.bounds is None:
        cfg.bounds = CONFIG.bounds

    rng = np.random.RandomState(cfg.seed)

    def objective(trial: optuna.trial.Trial):
        x = np.array([trial.suggest_float(k, *cfg.bounds[k]) for k in PARAM_COLS], dtype=np.float32)
        y = sur.predict(x)
        # L1 loss toward target steady states (equal weights)
        loss = abs(y[0] - cfg.target["A"]) + abs(y[1] - cfg.target["B"])
        # tiny regularizer to avoid extreme parameters
        loss += 1e-4 * np.linalg.norm(x, 2)
        return float(loss)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
    best = study.best_trial

    hist = [(t.number, t.value) for t in study.trials if t.value is not None]
    return best, hist


# ---------- Validation on the true ODE ----------

def validate_on_ode(params_vec: np.ndarray, t_span=(0.0, 200.0)):
    """Run the real ODE with best params; return steady stats + full trajectory."""
    params = {k: float(v) for k, v in zip(PARAM_COLS, params_vec)}
    t, Y = integrate(toggle_switch_ode, y0=(0.1, 0.1), t_span=t_span, params=params)
    summ = summarize_timeseries(t, Y, tail_frac=0.2)
    A_steady, B_steady = map(float, summ["steady"])
    return dict(t=t, Y=Y, steady_A=A_steady, steady_B=B_steady, summary=summ)


# ---------- Plot helpers ----------

def plot_convergence(history, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    it = [h[0] for h in history]
    val = [h[1] for h in history]
    plt.figure()
    plt.plot(it, val, marker='o', linewidth=1)
    plt.xlabel("trial")
    plt.ylabel("BO objective (|A-tA| + |B-tB|)")
    plt.title("Bayesian optimization convergence")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)

def plot_timeseries(t: np.ndarray, Y: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7,4))
    plt.plot(t, Y[:,0], label="A")
    plt.plot(t, Y[:,1], label="B")
    plt.xlabel("time")
    plt.ylabel("concentration (arb.)")
    plt.title("Toggle-switch trajectory (true ODE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)

# src/biocircuitai/control/optimizer.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import joblib
import numpy as np
import optuna
import torch
import matplotlib.pyplot as plt

from ..config import CONFIG
from ..simulate.models import toggle_switch_ode
from ..simulate.ode_solver import integrate, summarize_timeseries
from ..surrogate.dataset import PARAM_COLS, TARGET_COLS
from ..surrogate.evaluate import load_model  # model loader with hidden-size inference
from ..surrogate.preprocess import transform_xy, invert_y, DEFAULT_LOG_COLS


# --------------------------- Surrogate wrapper ---------------------------

class SurrogateWrapper:
    """
    Wraps a trained Torch model and optional scalers for normalized I/O.
    If a scalers.joblib is provided, inputs are log/standardized before
    prediction and outputs are inverse-transformed to original units.
    """
    def __init__(self, model: torch.nn.Module, device: str, scalers_path: str | None = None):
        self.model = model
        self.device = device
        self.xsc = None
        self.ysc = None
        self.log_cols = DEFAULT_LOG_COLS
        if scalers_path and os.path.exists(scalers_path):
            pack = joblib.load(scalers_path)
            self.xsc = pack.get("x_scaler", None)
            self.ysc = pack.get("y_scaler", None)
            self.log_cols = pack.get("log_cols", DEFAULT_LOG_COLS)

    def predict(self, x_vec: np.ndarray) -> np.ndarray:
        """
        x_vec: shape (len(PARAM_COLS),) in ORIGINAL units and order PARAM_COLS.
        Returns steady-state predictions (A, B) in ORIGINAL units.
        """
        import pandas as pd
        df = pd.DataFrame([dict(zip(PARAM_COLS, map(float, x_vec)))])
        if self.xsc is not None and self.ysc is not None:
            Xn, _ = transform_xy(df, PARAM_COLS, [], self.xsc, None, self.log_cols)
            x = torch.tensor(Xn, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                y_n = self.model(x).detach().cpu().numpy()
            y = invert_y(y_n, self.ysc)[0]
        else:
            x = torch.tensor(np.asarray(x_vec, dtype=np.float32)[None, :], device=self.device)
            with torch.no_grad():
                y = self.model(x).detach().cpu().numpy()[0]
        return y


# --------------------------- BO configuration ---------------------------

@dataclass
class BOConfig:
    target: Dict[str, float] | None = None
    bounds: Dict[str, Tuple[float, float]] | None = None
    n_trials: int = 80
    seed: int = 42

def bo_optimize(sur: SurrogateWrapper, cfg: BOConfig):
    """
    Bayesian optimization over PARAM_COLS bounds to minimize
    |A - tA| + |B - tB| (+ tiny L2 reg on params).
    Returns (best_trial, history).
    """
    if cfg.target is None:
        cfg.target = CONFIG.target
    if cfg.bounds is None:
        cfg.bounds = CONFIG.bounds

    def objective(trial: optuna.trial.Trial) -> float:
        x = np.array([trial.suggest_float(k, *cfg.bounds[k]) for k in PARAM_COLS], dtype=np.float32)
        y = sur.predict(x)  # (A, B) in original units
        loss = abs(y[0] - cfg.target["A"]) + abs(y[1] - cfg.target["B"])
        loss += 1e-4 * float(np.linalg.norm(x, ord=2))
        return float(loss)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
    )
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

    best = study.best_trial
    history: List[Tuple[int, float]] = [(t.number, t.value) for t in study.trials if t.value is not None]
    return best, history


# --------------------------- True ODE validation ---------------------------

def validate_on_ode(params_vec: np.ndarray, t_span: Tuple[float, float] = (0.0, 200.0)):
    """
    Run the real toggle-switch ODE with the chosen params.
    Returns dict with time series and steady-state summary.
    """
    params = {k: float(v) for k, v in zip(PARAM_COLS, params_vec)}
    t, Y = integrate(toggle_switch_ode, y0=(0.1, 0.1), t_span=t_span, params=params)
    summ = summarize_timeseries(t, Y, tail_frac=0.2)
    A_steady, B_steady = map(float, summ["steady"])
    return {"t": t, "Y": Y, "steady_A": A_steady, "steady_B": B_steady, "summary": summ}


# --------------------------- Plot helpers ---------------------------

def plot_convergence(history: List[Tuple[int, float]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    it = [h[0] for h in history]
    val = [h[1] for h in history]
    plt.figure()
    plt.plot(it, val, marker="o", linewidth=1)
    plt.xlabel("trial")
    plt.ylabel("BO objective (|A - tA| + |B - tB|)")
    plt.title("Bayesian optimization convergence")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)

def plot_timeseries(t: np.ndarray, Y: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(t, Y[:, 0], label="A")
    plt.plot(t, Y[:, 1], label="B")
    plt.xlabel("time")
    plt.ylabel("concentration (arb.)")
    plt.title("Toggle-switch trajectory (true ODE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)

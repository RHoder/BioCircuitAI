# src/biocircuitai/simulate/experiments.py
"""
Parameter sweeps → tabular datasets (CSV-ready).
"""
from __future__ import annotations
import itertools
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd

from ..config import CONFIG
from .models import toggle_switch_ode
from .ode_solver import integrate, summarize_timeseries


def _as_list(x) -> List:
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def sweep(param_grid: Dict[str, Iterable], seeds: Iterable[int] = (0, 1, 2),
          model_fn=toggle_switch_ode,
          y0=(0.1, 0.1),
          t_span=(0.0, 200.0)) -> pd.DataFrame:
    """
    Exhaustive grid sweep across param_grid (Cartesian product) × seeds.
    For each run: integrate ODE → summarize → one row.

    Returns:
        Pandas DataFrame with columns: params..., steady_A, steady_B, var_A, var_B
    """
    keys = sorted(param_grid.keys())
    grids = [list(param_grid[k]) for k in keys]
    rows = []

    for vals in itertools.product(*grids):
        params = dict(zip(keys, vals))
        for s in seeds:
            # small randomization of initial condition per seed
            y0_seeded = np.asarray(y0, dtype=float) + 0.01 * np.array([s, -s], dtype=float)
            t, Y = integrate(model_fn, y0_seeded, t_span=t_span, params=params)
            summary = summarize_timeseries(t, Y, tail_frac=0.2)
            A_steady, B_steady = summary["steady"]
            A_var, B_var = summary["var"]
            rows.append({
                **params,
                "seed": s,
                "steady_A": float(A_steady),
                "steady_B": float(B_steady),
                "var_A": float(A_var),
                "var_B": float(B_var),
            })

    return pd.DataFrame(rows)


def default_sweep() -> pd.DataFrame:
    """
    Convenience wrapper: use CONFIG.grid and CONFIG.defaults to produce a dataset.
    """
    return sweep(CONFIG.grid, seeds=[0, 1, 2])

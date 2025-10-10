# src/biocircuitai/simulate/experiments.py
"""
Parameter sweeps → tabular datasets (CSV-ready).
Now supports parallel execution via ProcessPoolExecutor.
"""
from __future__ import annotations
import itertools, os
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..config import CONFIG
from .models import toggle_switch_ode
from .ode_solver import integrate, summarize_timeseries


def _as_list(x) -> List:
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def _simulate_one(job: Tuple[Dict[str, float], int, Tuple[float,float], Tuple[float,float]]):
    """
    A single simulation job.
    Args:
      job: (params, seed, y0, t_span)
    Returns:
      dict row with params + steady/var metrics + seed
    """
    params, seed, y0, t_span = job
    y0_seeded = np.asarray(y0, dtype=float) + 0.01 * np.array([seed, -seed], dtype=float)
    t, Y = integrate(toggle_switch_ode, y0_seeded, t_span=t_span, params=params)
    summary = summarize_timeseries(t, Y, tail_frac=0.2)
    A_steady, B_steady = summary["steady"]
    A_var, B_var = summary["var"]
    return {
        **params,
        "seed": seed,
        "steady_A": float(A_steady),
        "steady_B": float(B_steady),
        "var_A": float(A_var),
        "var_B": float(B_var),
    }


def sweep(param_grid: Dict[str, Iterable],
          seeds: Iterable[int] = (0, 1, 2),
          model_fn=toggle_switch_ode,
          y0: Tuple[float,float] = (0.1, 0.1),
          t_span: Tuple[float,float] = (0.0, 200.0),
          workers: int = 1,
          chunksize: int = 1000) -> pd.DataFrame:
    """
    Exhaustive grid sweep across param_grid (Cartesian product) × seeds.

    Args:
      workers: 1 = serial. >1 = run with ProcessPoolExecutor(workers).
      chunksize: how many jobs to submit per batch to reduce overhead.
    """
    keys = sorted(param_grid.keys())
    grids = [list(param_grid[k]) for k in keys]
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grids)]
    seed_list = list(seeds)

    # Build job list
    jobs = [(params, s, y0, t_span) for params in combos for s in seed_list]

    if workers <= 1:
        # Serial path (easier to debug)
        rows = [_simulate_one(j) for j in jobs]
        return pd.DataFrame(rows)

    # Parallel path
    rows: List[dict] = []
    # On Windows, need the __main__ guard in scripts (we have it in run_simulation.py)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # Submit in chunks to reduce memory pressure
        for i in range(0, len(jobs), chunksize):
            batch = jobs[i:i+chunksize]
            futures = [ex.submit(_simulate_one, j) for j in batch]
            for fut in as_completed(futures):
                rows.append(fut.result())
    return pd.DataFrame(rows)


def default_sweep(workers: int = 1, chunksize: int = 1000) -> pd.DataFrame:
    """
    Convenience wrapper: use CONFIG.grid and CONFIG.defaults to produce a dataset.
    """
    return sweep(CONFIG.grid, seeds=[0, 1, 2], workers=workers, chunksize=chunksize)

# src/biocircuitai/simulate/ode_solver.py
"""
Thin wrappers around scipy.integrate.solve_ivp + helpers to summarize trajectories.
"""
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
from scipy.integrate import solve_ivp

def integrate(model_fn, y0, t_span=(0.0, 200.0), params: dict | None = None,
              t_eval=None, method="RK45", rtol=1e-6, atol=1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate an ODE model.
    Returns:
        t: (T,) time points
        Y: (T, D) state matrix
    """
    if params is None:
        params = {}
    if t_eval is None:
        # adaptive dense output; we’ll just use solver's internal steps
        sol = solve_ivp(lambda t, y: model_fn(t, y, params),
                        t_span=t_span, y0=np.asarray(y0, dtype=float),
                        method=method, rtol=rtol, atol=atol, dense_output=False)
        t = sol.t
        Y = sol.y.T
    else:
        sol = solve_ivp(lambda t, y: model_fn(t, y, params),
                        t_span=t_span, y0=np.asarray(y0, dtype=float),
                        t_eval=np.asarray(t_eval, dtype=float),
                        method=method, rtol=rtol, atol=atol)
        t = sol.t
        Y = sol.y.T
    return t, Y


def summarize_timeseries(t: np.ndarray, Y: np.ndarray, tail_frac: float = 0.2) -> Dict[str, np.ndarray]:
    """
    Compute simple summary statistics over the last 'tail_frac' of the trajectory.
    Returns:
        dict with "steady" (mean over tail) and "var" (variance over tail)
    """
    assert 0.0 < tail_frac <= 1.0, "tail_frac must be in (0,1]"
    n = len(t)
    start = max(0, int((1.0 - tail_frac) * n))
    tail = Y[start:, :]
    steady = np.mean(tail, axis=0)
    var = np.var(tail, axis=0)
    return {"steady": steady, "var": var}

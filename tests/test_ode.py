# tests/test_ode.py
from __future__ import annotations
import numpy as np

from biocircuitai.simulate.models import toggle_switch_ode
from biocircuitai.simulate.ode_solver import integrate, summarize_timeseries
from biocircuitai.config import CONFIG

def test_toggle_switch_integration_runs():
    params = CONFIG.defaults.copy()
    y0 = np.array([0.1, 0.1], dtype=float)
    t, Y = integrate(toggle_switch_ode, y0, t_span=(0.0, 50.0), params=params)
    assert np.isfinite(Y).all(), "State contains NaNs or infs."
    summary = summarize_timeseries(t, Y, tail_frac=0.2)
    steady = summary["steady"]
    assert steady.shape == (2,), "Expected steady state vector of length 2."

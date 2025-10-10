# tests/test_control.py
from __future__ import annotations
import json, os
from biocircuitai.control.optimizer import BOConfig, SurrogateWrapper, bo_optimize, validate_on_ode
from biocircuitai.surrogate.evaluate import load_model

def test_bo_runs():
    model, device, _ = load_model("data/models/surrogate.pt")
    sur = SurrogateWrapper(model, device)
    cfg = BOConfig(n_trials=10)
    best, hist = bo_optimize(sur, cfg)
    assert len(hist) >= 1

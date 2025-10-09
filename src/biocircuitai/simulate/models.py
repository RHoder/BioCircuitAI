# src/biocircuitai/simulate/models.py
"""
Deterministic ODE models for simple genetic circuits.
Currently: Toggle switch (mutual repression). Repressilator stub included.
"""
from __future__ import annotations
import numpy as np

def toggle_switch_ode(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """
    Classic Gardner toggle switch:
      dA/dt = alpha_A / (1 + (B/K_B)^n_B) - dA * A
      dB/dt = alpha_B / (1 + (A/K_A)^n_A) - dB * B

    Args:
        t: time (unused; ODE signature)
        y: state vector [A, B]
        params: dict with alpha_A, alpha_B, K_A, K_B, n_A, n_B, dA, dB
    Returns:
        np.ndarray([dA_dt, dB_dt])
    """
    A, B = y
    alpha_A = params["alpha_A"]
    alpha_B = params["alpha_B"]
    K_A = params["K_A"]
    K_B = params["K_B"]
    n_A = params["n_A"]
    n_B = params["n_B"]
    dA_ = params["dA"]
    dB_ = params["dB"]

    dA_dt = alpha_A / (1.0 + (B / K_B) ** n_B) - dA_ * A
    dB_dt = alpha_B / (1.0 + (A / K_A) ** n_A) - dB_ * B
    return np.array([dA_dt, dB_dt], dtype=float)


# Optional second model (placeholder for later)
def repressilator_ode(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """
    Stub for the Elowitz repressilator (3-node ring). Fill later if needed.
    y = [X1, X2, X3]
    """
    X1, X2, X3 = y
    alpha = params.get("alpha", 50.0)
    K = params.get("K", 1.0)
    n = params.get("n", 2.0)
    d = params.get("d", 1.0)

    dX1 = alpha / (1.0 + (X3 / K) ** n) - d * X1
    dX2 = alpha / (1.0 + (X1 / K) ** n) - d * X2
    dX3 = alpha / (1.0 + (X2 / K) ** n) - d * X3
    return np.array([dX1, dX2, dX3], dtype=float)

# Pseudocode
import numpy as np

def toggle_switch_ode(t, y, params):
    # y = [A, B]; A represses B and vice versa (Hill repression)
    A, B = y
    alpha_A, alpha_B = params["alpha_A"], params["alpha_B"]
    K_A, K_B = params["K_A"], params["K_B"]
    n_A, n_B = params["n_A"], params["n_B"]
    dA, dB = params["dA"], params["dB"]

    dA_dt = alpha_A / (1 + (B / K_B)**n_B) - dA * A
    dB_dt = alpha_B / (1 + (A / K_A)**n_A) - dB * B
    return np.array([dA_dt, dB_dt])

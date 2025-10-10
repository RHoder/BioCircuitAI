# src/biocircuitai/control/controllers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, List
import numpy as np
import matplotlib.pyplot as plt

from ..simulate.models import toggle_switch_ode
from ..simulate.ode_solver import integrate
from ..surrogate.dataset import TARGET_COLS

# -------------------- Control-aware ODE wrapper --------------------

def _controlled_toggle_ode(u: Tuple[float, float]) -> Callable:
    """
    Returns an ODE fn f(t,y,params) that multiplies alpha_A/alpha_B
    by (1+u_A) and (1+u_B), respectively. u in [-0.9, +3.0] typical.
    """
    uA, uB = u
    scaleA = max(0.0, 1.0 + float(uA))
    scaleB = max(0.0, 1.0 + float(uB))

    def f(t: float, y: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        params2 = dict(params)
        params2["alpha_A"] = params["alpha_A"] * scaleA
        params2["alpha_B"] = params["alpha_B"] * scaleB
        return toggle_switch_ode(t, y, params2)
    return f

# -------------------- PID controller --------------------

@dataclass
class PIDGains:
    Kp: float
    Ki: float
    Kd: float

@dataclass
class PIDState:
    integ: float = 0.0
    prev_err: float = 0.0

def _pid_step(err: float, dt: float, gains: PIDGains, st: PIDState,
              integ_limits: Tuple[float, float] = (-10.0, 10.0)) -> Tuple[float, PIDState]:
    st.integ = np.clip(st.integ + err * dt, *integ_limits)
    deriv = (err - st.prev_err) / dt if dt > 0 else 0.0
    out = gains.Kp * err + gains.Ki * st.integ + gains.Kd * deriv
    st.prev_err = err
    return out, st

# -------------------- Closed-loop configuration --------------------

@dataclass
class ClosedLoopConfig:
    # plant (uncontrolled) parameters
    params: Dict[str, float]
    # target steady values
    targetA: float
    targetB: float
    # simulation horizon and control rate
    T: float = 200.0         # total time
    dt_ctrl: float = 2.0     # control update period (s)
    # integrator step horizon per control interval
    dt_ode: float = 2.0
    # PID gains (independent for uA and uB channels)
    pidA: PIDGains = PIDGains(0.2, 0.01, 0.0)
    pidB: PIDGains = PIDGains(0.2, 0.01, 0.0)
    # input saturation
    umin: float = -0.9
    umax: float = 3.0
    # initial conditions
    y0: Tuple[float, float] = (0.1, 0.1)

# -------------------- Runner --------------------

def run_closed_loop(cfg: ClosedLoopConfig):
    """
    Piecewise-constant control: every dt_ctrl, compute u from PID on error
    (target - current), integrate plant over next interval with that u.
    Returns dict with time, states, control, and per-interval data.
    """
    n_steps = int(np.ceil(cfg.T / cfg.dt_ctrl))
    t_all: List[float] = []
    Y_all: List[np.ndarray] = []
    U_all: List[Tuple[float, float]] = []
    seg_times: List[np.ndarray] = []

    y = np.array(cfg.y0, dtype=float)
    t_cur = 0.0
    stA, stB = PIDState(), PIDState()

    for k in range(n_steps):
        # Measure error at interval start
        errA = cfg.targetA - y[0]
        errB = cfg.targetB - y[1]

        # PID updates → control action
        uA, stA = _pid_step(errA, cfg.dt_ctrl, cfg.pidA, stA)
        uB, stB = _pid_step(errB, cfg.dt_ctrl, cfg.pidB, stB)
        uA = float(np.clip(uA, cfg.umin, cfg.umax))
        uB = float(np.clip(uB, cfg.umin, cfg.umax))
        U_all.append((uA, uB))

        # Integrate plant over [t_cur, t_cur + dt_ctrl] with constant u
        ode = _controlled_toggle_ode((uA, uB))
        t_span = (t_cur, min(cfg.T, t_cur + cfg.dt_ctrl))
        t_seg, Y_seg = integrate(ode, y0=y, t_span=t_span, params=cfg.params, t_eval=None, rtol=1e-6, atol=1e-8)

        # Append segment (drop duplicate time at join)
        if k == 0:
            t_all.extend(t_seg.tolist())
            Y_all.append(Y_seg)
        else:
            t_all.extend(t_seg[1:].tolist())
            Y_all.append(Y_seg[1:, :])
        seg_times.append(t_seg)

        # Prepare for next interval
        y = Y_seg[-1, :]
        t_cur = t_span[1]

        if t_cur >= cfg.T:
            break

    # Stack results
    Y = np.vstack(Y_all)
    t = np.asarray(t_all)

    return {
        "t": t,
        "Y": Y,                   # columns: A, B
        "U": np.asarray(U_all),   # rows per interval: [uA, uB]
        "target": (cfg.targetA, cfg.targetB),
        "dt_ctrl": cfg.dt_ctrl,
        "segments": seg_times,
        "params": cfg.params,
    }

# -------------------- Plot helpers --------------------

def plot_closed_loop(res: dict, out_path: str | None = None):
    t, Y, U = res["t"], res["Y"], res["U"]
    targetA, targetB = res["target"]
    plt.figure(figsize=(8, 5))
    # states
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t, Y[:,0], label="A")
    ax1.plot(t, Y[:,1], label="B")
    ax1.axhline(targetA, linestyle="--", linewidth=1, label=f"target A={targetA}")
    ax1.axhline(targetB, linestyle="--", linewidth=1, label=f"target B={targetB}")
    ax1.set_ylabel("concentration")
    ax1.set_title("Closed-loop trajectory")
    ax1.legend(loc="best")
    # controls (staircase)
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    t_ctrl = np.arange(len(U)) * res["dt_ctrl"]
    ax2.step(t_ctrl, U[:,0], where="post", label="uA")
    ax2.step(t_ctrl, U[:,1], where="post", label="uB")
    ax2.set_xlabel("time")
    ax2.set_ylabel("control u")
    ax2.legend(loc="best")
    plt.tight_layout()
    if out_path:
        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=160)
    return plt.gcf()

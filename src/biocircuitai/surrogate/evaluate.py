# src/biocircuitai/surrogate/evaluate.py
from __future__ import annotations
import os, joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from .dataset import CircuitDataset, PARAM_COLS, TARGET_COLS
from .model import MLP
from .preprocess import transform_xy, invert_y, DEFAULT_LOG_COLS

def _infer_hidden_from_state(state_dict):
    linear_keys = [k for k in state_dict.keys() if k.endswith(".weight") and k.startswith("net.")]
    if not linear_keys: return None
    linear_keys = sorted(linear_keys, key=lambda k: int(k.split(".")[1]))
    if len(linear_keys) >= 2:
        return tuple(state_dict[k].shape[0] for k in linear_keys[:-1])
    return None

def load_model(model_path: str, hidden: tuple[int, ...] | None = None, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(model_path, map_location=device)
    if hidden is None:
        hidden = _infer_hidden_from_state(state) or (128,128)
    model = MLP(len(PARAM_COLS), len(TARGET_COLS), hidden=hidden).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, device, hidden

def _load_scalers_if_any(model_path: str):
    # expects scalers alongside model by default (same folder)
    scalers_path = os.path.join(os.path.dirname(model_path), "scalers.joblib")
    if os.path.exists(scalers_path):
        pack = joblib.load(scalers_path)
        return pack["x_scaler"], pack["y_scaler"], pack.get("log_cols", DEFAULT_LOG_COLS)
    return None, None, DEFAULT_LOG_COLS

def eval_scatter(csv_path: str, model_path: str, out_png: str = "data/models/pred_vs_true.png", hidden: tuple[int, ...] | None = None):
    model, device, used_hidden = load_model(model_path, hidden=hidden)
    xsc, ysc, log_cols = _load_scalers_if_any(model_path)

    # Load raw dataset for plotting; transform X for model; invert preds to original units
    import pandas as pd
    df = pd.read_csv(csv_path)
    if xsc is not None and ysc is not None:
        Xn, _ = transform_xy(df, PARAM_COLS, TARGET_COLS, xsc, None, log_cols)
    else:
        Xn = df[PARAM_COLS].values

    X = torch.tensor(Xn, dtype=torch.float32, device=device)
    with torch.no_grad():
        Pn = model(X).cpu().numpy()

    if ysc is not None:
        P = invert_y(Pn, ysc)
    else:
        P = Pn

    y = df[TARGET_COLS].values

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig = plt.figure(figsize=(8,4))
    for i, label in enumerate(TARGET_COLS):
        ax = fig.add_subplot(1,2,i+1)
        ax.scatter(y[:, i], P[:, i], s=3, alpha=0.3)
        lo, hi = float(y[:, i].min()), float(y[:, i].max())
        ax.plot([lo, hi], [lo, hi], linewidth=1)
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"pred {label}")
        ax.set_title(f"{label} — pred vs true (hidden={used_hidden}, norm={'on' if xsc is not None else 'off'})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"[OK] Wrote {out_png}")

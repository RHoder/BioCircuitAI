# src/biocircuitai/surrogate/evaluate.py
from __future__ import annotations
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from .dataset import CircuitDataset, PARAM_COLS, TARGET_COLS
from .model import MLP

def _infer_hidden_from_state(state_dict):
    """
    Try to infer hidden layer widths from state_dict keys produced by MLP(...).
    Assumes layers laid out as: [Linear, ReLU, (Dropout)] repeated ... then final Linear.
    """
    # common keys: 'net.0.weight' (first Linear), 'net.2.weight' (second Linear) if dropout absent
    # safer: collect all 'net.<idx>.weight' except the LAST one (output layer)
    linear_keys = [k for k in state_dict.keys() if k.endswith(".weight") and k.startswith("net.")]
    # sort by index
    def _idx(k): return int(k.split(".")[1])
    linear_keys = sorted(linear_keys, key=_idx)
    if not linear_keys:
        return None
    # last linear is output; hidden = all previous out_features
    if len(linear_keys) >= 2:
        hidden = [ state_dict[k].shape[0] for k in linear_keys[:-1] ]
        return tuple(hidden) if hidden else None
    else:
        # only one Linear means no hidden layers (unlikely here) — fall back
        return None

def load_model(model_path: str, hidden: tuple[int, ...] | None = None, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(model_path, map_location=device)
    if hidden is None:
        inferred = _infer_hidden_from_state(state)
        hidden = inferred if inferred is not None else (128, 128)
    model = MLP(len(PARAM_COLS), len(TARGET_COLS), hidden=hidden).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, device, hidden

def eval_scatter(csv_path: str, model_path: str, out_png: str = "data/models/pred_vs_true.png", hidden: tuple[int, ...] | None = None):
    model, device, used_hidden = load_model(model_path, hidden=hidden)
    ds = CircuitDataset(csv_path, PARAM_COLS, TARGET_COLS)
    X = ds.X.to(device)
    with torch.no_grad():
        P = model(X).cpu().numpy()
    y = ds.y.numpy()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig = plt.figure(figsize=(8,4))
    for i, label in enumerate(TARGET_COLS):
        ax = fig.add_subplot(1,2,i+1)
        ax.scatter(y[:, i], P[:, i], s=3, alpha=0.3)
        lo, hi = float(y[:, i].min()), float(y[:, i].max())
        ax.plot([lo, hi], [lo, hi], linewidth=1)
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"pred {label}")
        ax.set_title(f"{label} — pred vs true (hidden={used_hidden})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"[OK] Wrote {out_png} (hidden={used_hidden})")

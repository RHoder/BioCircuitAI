# src/biocircuitai/surrogate/train.py
from __future__ import annotations
from dataclasses import dataclass
import os, json, joblib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from .dataset import PARAM_COLS, TARGET_COLS
from .model import MLP
from .preprocess import fit_scalers, transform_xy

@dataclass
class TrainConfig:
    csv_path: str
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 60
    val_split: float = 0.2
    hidden: tuple = (128,128)
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_out: str = "data/models/surrogate.pt"
    scalers_out: str = "data/models/scalers.joblib"
    normalize: bool = True

def train_surrogate(cfg: TrainConfig):
    os.makedirs(os.path.dirname(cfg.model_out), exist_ok=True)

    # Load data
    df = pd.read_csv(cfg.csv_path)

    # Train/val split indices (stable)
    N = len(df)
    n_val = max(1, int(cfg.val_split * N))
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(N, generator=gen).numpy()
    idx_tr, idx_va = idx[:-n_val], idx[-n_val:]
    df_tr = df.iloc[idx_tr].reset_index(drop=True)
    df_va = df.iloc[idx_va].reset_index(drop=True)

    # Preprocess / scalers
    if cfg.normalize:
        x_scaler, y_scaler, log_cols = fit_scalers(df_tr, PARAM_COLS, TARGET_COLS)
        Xtr, Ytr = transform_xy(df_tr, PARAM_COLS, TARGET_COLS, x_scaler, y_scaler, log_cols)
        Xva, Yva = transform_xy(df_va, PARAM_COLS, TARGET_COLS, x_scaler, y_scaler, log_cols)
        # Save scalers
        joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "log_cols": log_cols}, cfg.scalers_out)
        with open(cfg.scalers_out.replace(".joblib", ".meta.json"), "w") as f:
            json.dump({"PARAM_COLS": PARAM_COLS, "TARGET_COLS": TARGET_COLS}, f)
    else:
        # raw mode
        Xtr = df_tr[PARAM_COLS].values
        Ytr = df_tr[TARGET_COLS].values
        Xva = df_va[PARAM_COLS].values
        Yva = df_va[TARGET_COLS].values
        x_scaler = y_scaler = None

    # Datasets
    tl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                  torch.tensor(Ytr, dtype=torch.float32)),
                    batch_size=cfg.batch_size, shuffle=True)
    vl = DataLoader(TensorDataset(torch.tensor(Xva, dtype=torch.float32),
                                  torch.tensor(Yva, dtype=torch.float32)),
                    batch_size=cfg.batch_size, shuffle=False)

    # Model
    model = MLP(len(PARAM_COLS), len(TARGET_COLS), hidden=cfg.hidden, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for ep in range(1, cfg.epochs+1):
        model.train(); tr_loss = 0.0; n_tr = 0
        for X, y in tl:
            X, y = X.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward(); opt.step()
            tr_loss += loss.item()*X.size(0); n_tr += X.size(0)
        tr_loss /= max(1, n_tr)

        model.eval(); va_loss = 0.0; n_va = 0
        with torch.no_grad():
            for X, y in vl:
                X, y = X.to(cfg.device), y.to(cfg.device)
                pred = model(X)
                loss = loss_fn(pred, y)
                va_loss += loss.item()*X.size(0); n_va += X.size(0)
        va_loss /= max(1, n_va)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}

        print(f"[epoch {ep:03d}] train_mse={tr_loss:.6f}  val_mse={va_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), cfg.model_out)
    print(f"[OK] Saved model → {cfg.model_out} (best val MSE={best_val:.6f})")
    if cfg.normalize:
        print(f"[OK] Saved scalers → {cfg.scalers_out}")
    return model, best_val

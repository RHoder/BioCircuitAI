# src/biocircuitai/surrogate/train.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from .dataset import CircuitDataset, PARAM_COLS, TARGET_COLS
from .model import MLP

@dataclass
class TrainConfig:
    csv_path: str
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 50
    val_split: float = 0.2
    hidden: tuple = (128,128)
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_out: str = "data/models/surrogate.pt"

def train_surrogate(cfg: TrainConfig):
    ds = CircuitDataset(cfg.csv_path, PARAM_COLS, TARGET_COLS)
    n_val = max(1, int(cfg.val_split * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val], generator=torch.Generator().manual_seed(42))

    tl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MLP(len(PARAM_COLS), len(TARGET_COLS), hidden=cfg.hidden, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for ep in range(1, cfg.epochs+1):
        # train
        model.train()
        tr_loss, n_tr = 0.0, 0
        for X, y in tl:
            X = X.to(cfg.device); y = y.to(cfg.device)
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * X.size(0)
            n_tr += X.size(0)
        tr_loss /= max(1, n_tr)

        # val
        model.eval()
        va_loss, n_va = 0.0, 0
        with torch.no_grad():
            for X, y in vl:
                X = X.to(cfg.device); y = y.to(cfg.device)
                pred = model(X)
                loss = loss_fn(pred, y)
                va_loss += loss.item() * X.size(0)
                n_va += X.size(0)
        va_loss /= max(1, n_va)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[epoch {ep:03d}] train_mse={tr_loss:.6f}  val_mse={va_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    import os
    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), cfg.model_out)
    print(f"[OK] Saved best model → {cfg.model_out} (val_mse={best_val:.6f})")
    return model, best_val

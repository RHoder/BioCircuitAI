# src/biocircuitai/surrogate/dataset.py
from __future__ import annotations
import pandas as pd
import torch
from torch.utils.data import Dataset

PARAM_COLS = ["alpha_A","alpha_B","K_A","K_B","n_A","n_B","dA","dB"]  # exclude 'seed' on purpose
TARGET_COLS = ["steady_A","steady_B"]

class CircuitDataset(Dataset):
    def __init__(self, csv_path: str, x_cols=PARAM_COLS, y_cols=TARGET_COLS):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df[x_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[y_cols].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

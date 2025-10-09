import pandas as pd, torch
from torch.utils.data import Dataset

class CircuitDataset(Dataset):
    def __init__(self, csv_path, x_cols, y_cols):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df[x_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[y_cols].values, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

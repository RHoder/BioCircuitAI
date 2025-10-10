# src/biocircuitai/surrogate/model.py
from __future__ import annotations
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(128,128), dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# tests/test_surrogate.py
from __future__ import annotations
import torch
from biocircuitai.surrogate.train import TrainConfig, train_surrogate

def test_train_small_runs():
    # Train on a tiny slice to ensure the loop works
    cfg = TrainConfig(csv_path="data/processed/toggle_dataset.csv", epochs=2, batch_size=512)
    model, val = train_surrogate(cfg)
    assert isinstance(val, float)
    assert hasattr(model, "state_dict")

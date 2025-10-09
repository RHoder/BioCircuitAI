"""
Global configuration for BioCircuitAI.
Edit here instead of hard-coding in scripts.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
LOG_DIR = ROOT / "logs"

# ensure folders exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# simulation defaults
DEFAULT_PARAMS = dict(
    alpha_A=40.0,
    alpha_B=35.0,
    K_A=1.0,
    K_B=1.0,
    n_A=2.0,
    n_B=2.0,
    dA=1.0,
    dB=1.0,
)

DEFAULT_GRID = {
    "alpha_A": np.linspace(20, 60, 5),
    "alpha_B": np.linspace(20, 60, 5),
    "K_A": [0.5, 1.0, 2.0],
    "K_B": [0.5, 1.0, 2.0],
    "n_A": [1, 2, 3],
    "n_B": [1, 2, 3],
    "dA": [0.5, 1.0, 1.5],
    "dB": [0.5, 1.0, 1.5],
}

# ML / training defaults
TRAINING = dict(batch_size=64, lr=1e-3, epochs=200, val_split=0.2)

# optimizer defaults
OPT_BOUNDS = {
    "alpha_A": (10, 70),
    "alpha_B": (10, 70),
    "K_A": (0.3, 3.0),
    "K_B": (0.3, 3.0),
    "n_A": (1, 4),
    "n_B": (1, 4),
    "dA": (0.3, 2.0),
    "dB": (0.3, 2.0),
}

TARGET_PHENOTYPE = {"A": 10.0, "B": 8.0}

# random seeds
SEED = 42
np.random.seed(SEED)

class _Config:
    def __init__(self):
        self.root = ROOT
        self.paths = dict(
            data=DATA_DIR,
            processed=PROCESSED_DIR,
            models=MODEL_DIR,
            logs=LOG_DIR,
        )
        self.defaults = DEFAULT_PARAMS
        self.grid = DEFAULT_GRID
        self.training = TRAINING
        self.bounds = OPT_BOUNDS
        self.target = TARGET_PHENOTYPE
        self.seed = SEED

CONFIG = _Config()

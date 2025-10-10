# src/biocircuitai/surrogate/preprocess.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_LOG_COLS = ["K_A", "K_B", "dA", "dB"]  # rate/affinity-like -> log1p helps

def apply_log(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        # guard against nonpositive values (shouldn't happen in your grid)
        df[c] = np.log1p(np.clip(df[c].values, a_min=1e-12, a_max=None))
    return df

def fit_scalers(
    df_train: pd.DataFrame,
    x_cols: List[str],
    y_cols: List[str],
    log_cols: List[str] = DEFAULT_LOG_COLS,
) -> Tuple[StandardScaler, StandardScaler, List[str]]:
    X_train = apply_log(df_train, log_cols)[x_cols].values
    Y_train = df_train[y_cols].values
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)
    return x_scaler, y_scaler, log_cols

def transform_xy(
    df: pd.DataFrame,
    x_cols: List[str],
    y_cols: List[str],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler | None,
    log_cols: List[str],
):
    X = apply_log(df, log_cols)[x_cols].values
    Xn = x_scaler.transform(X)
    Yn = None
    if y_scaler is not None and len(y_cols) > 0:
        Y = df[y_cols].values
        Yn = y_scaler.transform(Y)
    return Xn, Yn

def invert_y(Yn: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    return y_scaler.inverse_transform(Yn)

"""Preprocessing: Min-Max normalization (train-fit only), sliding-window
sequence construction, and chronological train/val/test splitting.

Works for both the multivariate (F=6) and univariate (F=1) experiments by simply
passing a different ``feature_cols`` list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config


def split_row(n_rows: int) -> int:
    """Index of the first test row: the last ``TEST_SIZE`` fraction of rows are test targets."""
    n_test = round(n_rows * config.TEST_SIZE)
    return n_rows - n_test


def _make_windows(features: np.ndarray, target: np.ndarray, lookback: int):
    """Build (samples, lookback, n_features) inputs, aligned targets, and the
    target row index for each sample (target row = i + lookback)."""
    n = len(target)
    X, y, rows = [], [], []
    for i in range(n - lookback):
        X.append(features[i : i + lookback])
        y.append(target[i + lookback])
        rows.append(i + lookback)
    return np.asarray(X), np.asarray(y), np.asarray(rows)


def build_supervised(df: pd.DataFrame, feature_cols: list[str], lookback: int) -> dict:
    """Return a fully split, normalized supervised-learning bundle for one experiment.

    Scalers are fit on the training rows only (rows before the test region) to
    prevent look-ahead bias, then applied to the whole series before windowing.
    """
    n_rows = len(df)
    s = split_row(n_rows)

    feat_raw = df[feature_cols].to_numpy(dtype="float64")
    tgt_raw = df[[config.TARGET_COL]].to_numpy(dtype="float64")

    feat_scaler = MinMaxScaler().fit(feat_raw[:s])
    tgt_scaler = MinMaxScaler().fit(tgt_raw[:s])

    feat_scaled = feat_scaler.transform(feat_raw)
    tgt_scaled = tgt_scaler.transform(tgt_raw).ravel()

    X, y, rows = _make_windows(feat_scaled, tgt_scaled, lookback)

    is_test = rows >= s
    X_test, y_test, rows_test = X[is_test], y[is_test], rows[is_test]

    X_tv, y_tv = X[~is_test], y[~is_test]
    n_val = round(len(X_tv) * config.VAL_FRACTION)
    n_val = max(1, n_val)
    X_train, y_train = X_tv[:-n_val], y_tv[:-n_val]
    X_val, y_val = X_tv[-n_val:], y_tv[-n_val:]

    # Representative original-scale feature values per test sample (last input
    # timestep = row t-1) for SHAP dependence / beeswarm colouring.
    feat_last_orig = feat_raw[rows_test - 1]

    tgt_min = float(tgt_scaler.data_min_[0])
    tgt_max = float(tgt_scaler.data_max_[0])

    weeks = [df[config.DATE_COL].iloc[int(t)].strftime("%Y-%m-%d") for t in rows_test]
    y_test_orig = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    return {
        "lookback": lookback,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_orig": y_test_orig,
        "rows_test": rows_test,
        "feat_last_orig": feat_last_orig,
        "feat_scaler": feat_scaler,
        "tgt_scaler": tgt_scaler,
        "tgt_min": tgt_min,
        "tgt_max": tgt_max,
        "tgt_range": tgt_max - tgt_min,
        "weeks": weeks,
    }


def inverse_target(scaled: np.ndarray, tgt_scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transform normalized predictions back to PHP/kg."""
    return tgt_scaler.inverse_transform(np.asarray(scaled).reshape(-1, 1)).ravel()

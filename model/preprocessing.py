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


def _make_windows(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    origins: np.ndarray,
    include_targets: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build lookback inputs and 12 consecutive target values for each origin."""
    horizon = config.FORECAST_HORIZON
    X = np.asarray([features[o - lookback:o] for o in origins])
    y = (
        np.asarray([target[o:o + horizon] for o in origins])
        if include_targets
        else np.empty((len(origins), horizon))
    )
    return X, y, np.asarray(origins, dtype=int)


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
    if config.CLIP_INPUTS_TO_TRAIN_RANGE:
        # Out-of-range (shock) values are held at the training min/max.
        feat_scaled = np.clip(feat_scaled, 0.0, 1.0)
    tgt_scaled = tgt_scaler.transform(tgt_raw).ravel()

    horizon = config.FORECAST_HORIZON

    # Every training/validation target must be strictly before the first test row.
    eligible_origins = np.arange(lookback, s - horizon + 1)
    if not len(eligible_origins):
        raise ValueError("Not enough pre-test rows for the requested lookback and horizon.")
    n_val = max(1, round(len(eligible_origins) * config.VAL_FRACTION))
    val_start = len(eligible_origins) - n_val
    train_end = val_start - (horizon - 1)
    if train_end <= 0:
        raise ValueError("Not enough windows remain for training after the validation gap.")

    train_origins = eligible_origins[:train_end]
    val_origins = eligible_origins[val_start:]
    X_train, y_train, _ = _make_windows(feat_scaled, tgt_scaled, lookback, train_origins)
    X_val, y_val, _ = _make_windows(feat_scaled, tgt_scaled, lookback, val_origins)
    target_offsets = np.arange(horizon)

    # Each horizon gets an input window ending immediately before its target.
    rows_test = np.arange(s, n_rows)
    weeks = [df[config.DATE_COL].iloc[int(t)].strftime("%Y-%m-%d") for t in rows_test]
    X_test_by_horizon: dict[str, np.ndarray] = {}
    origins_test_by_horizon: dict[str, np.ndarray] = {}
    feat_last_orig_by_horizon: dict[str, np.ndarray] = {}
    for h in range(1, horizon + 1):
        origins = rows_test - h + 1
        X_test_by_horizon[str(h)], _, origins_test_by_horizon[str(h)] = _make_windows(
            feat_scaled, tgt_scaled, lookback, origins, include_targets=False
        )
        feat_last_orig_by_horizon[str(h)] = feat_raw[origins - 1]

    tgt_min = float(tgt_scaler.data_min_[0])
    tgt_max = float(tgt_scaler.data_max_[0])

    y_test_orig = tgt_raw[rows_test, 0]
    y_test_scaled = tgt_scaled[rows_test, None]

    return {
        "horizon": horizon,
        "lookback": lookback,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "y_train": y_train,
        "train_target_rows": train_origins[:, None] + target_offsets,
        "X_val": X_val,
        "y_val": y_val,
        "val_target_rows": val_origins[:, None] + target_offsets,
        "X_test_by_horizon": X_test_by_horizon,
        "origins_test_by_horizon": origins_test_by_horizon,
        "y_test": np.repeat(y_test_scaled, horizon, axis=1),
        "y_test_orig": y_test_orig,
        "y_test_orig_by_horizon": {
            str(h): y_test_orig.copy() for h in range(1, horizon + 1)
        },
        "rows_test": rows_test,
        "weeks_by_horizon": {
            str(h): list(weeks) for h in range(1, horizon + 1)
        },
        "feat_last_orig": feat_last_orig_by_horizon,
        "feat_last_orig_by_horizon": feat_last_orig_by_horizon,
        "feat_scaler": feat_scaler,
        "tgt_scaler": tgt_scaler,
        "tgt_min": tgt_min,
        "tgt_max": tgt_max,
        "tgt_range": tgt_max - tgt_min,
        "weeks": weeks,
    }


def inverse_target(scaled: np.ndarray, tgt_scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transform normalized predictions back to PHP/kg."""
    scaled = np.asarray(scaled)
    return tgt_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape)

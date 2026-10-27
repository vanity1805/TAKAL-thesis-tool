# =============================================================================
# Program      : TAKAL preprocessing (model/preprocessing.py)
# Description  : Min-Max normalization, sliding-window sequence construction and
#                the chronological train / validation / test split for the
#                weekly NCR rice price dataset.
# Author       : TAKAL Thesis Team (contact: chavezearljustineb@gmail.com)
# System Fit   : Second stage of the offline pipeline (python -m model.run_all).
#                It receives the cleaned 330-week table from data_prep.py and
#                hands a "bundle" of ready-to-train arrays to train.py,
#                explain.py, predict_future.py and export_results.py.
# Date Written : 2026
# Date Revised : 2026-10-27
# Purpose      : Turn the weekly table into supervised-learning tensors without
#                leaking future information into training: scalers are fitted on
#                pre-test rows only, and every training / validation target lies
#                strictly before the first test row.
# Algorithms   : Min-Max scaling (scikit-learn); sliding-window segmentation
#                (each window of L weeks predicts the next 12 weekly prices);
#                chronological 80/20 split with an 11-origin purge between the
#                training and validation windows.
# Data         : NumPy arrays shaped (samples, timesteps, features) for model
# Structures     inputs and (samples, 12) for targets; one dictionary (the
#                "bundle") holding those arrays, the fitted scalers and the
#                horizon-keyed ("1" .. "12") test windows; plain lists of
#                "YYYY-MM-DD" week labels.
# =============================================================================
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
    x_windows = np.asarray([features[origin - lookback:origin] for origin in origins])
    y_windows = (
        np.asarray([target[origin:origin + horizon] for origin in origins])
        if include_targets
        else np.empty((len(origins), horizon))
    )
    return x_windows, y_windows, np.asarray(origins, dtype=int)


def build_supervised(weekly_data: pd.DataFrame, feature_cols: list[str], lookback: int) -> dict:
    """Return a fully split, normalized supervised-learning bundle for one experiment.

    Scalers are fit on the training rows only (rows before the test region) to
    prevent look-ahead bias, then applied to the whole series before windowing.
    """
    n_rows = len(weekly_data)
    split_index = split_row(n_rows)

    feature_raw = weekly_data[feature_cols].to_numpy(dtype="float64")
    target_raw = weekly_data[[config.TARGET_COL]].to_numpy(dtype="float64")

    feature_scaler = MinMaxScaler().fit(feature_raw[:split_index])
    target_scaler = MinMaxScaler().fit(target_raw[:split_index])

    feature_scaled = feature_scaler.transform(feature_raw)
    if config.CLIP_INPUTS_TO_TRAIN_RANGE:
        # Out-of-range (shock) values are held at the training min/max.
        feature_scaled = np.clip(feature_scaled, 0.0, 1.0)
    target_scaled = target_scaler.transform(target_raw).ravel()

    horizon = config.FORECAST_HORIZON

    # Every training/validation target must be strictly before the first test row.
    eligible_origins = np.arange(lookback, split_index - horizon + 1)
    if not len(eligible_origins):
        raise ValueError("Not enough pre-test rows for the requested lookback and horizon.")
    n_validation = max(1, round(len(eligible_origins) * config.VAL_FRACTION))
    validation_start = len(eligible_origins) - n_validation
    train_end = validation_start - (horizon - 1)
    if train_end <= 0:
        raise ValueError("Not enough windows remain for training after the validation gap.")

    train_origins = eligible_origins[:train_end]
    validation_origins = eligible_origins[validation_start:]
    x_train, y_train, _ = _make_windows(feature_scaled, target_scaled, lookback, train_origins)
    x_validation, y_validation, _ = _make_windows(
        feature_scaled, target_scaled, lookback, validation_origins
    )
    target_offsets = np.arange(horizon)

    # Each horizon gets an input window ending immediately before its target.
    test_rows = np.arange(split_index, n_rows)
    weeks = [weekly_data[config.DATE_COL].iloc[int(row)].strftime("%Y-%m-%d") for row in test_rows]
    x_test_by_horizon: dict[str, np.ndarray] = {}
    origins_test_by_horizon: dict[str, np.ndarray] = {}
    feature_last_original_by_horizon: dict[str, np.ndarray] = {}
    for step in range(1, horizon + 1):
        origins = test_rows - step + 1
        x_test_by_horizon[str(step)], _, origins_test_by_horizon[str(step)] = _make_windows(
            feature_scaled, target_scaled, lookback, origins, include_targets=False
        )
        feature_last_original_by_horizon[str(step)] = feature_raw[origins - 1]

    target_min = float(target_scaler.data_min_[0])
    target_max = float(target_scaler.data_max_[0])

    y_test_original = target_raw[test_rows, 0]
    y_test_scaled = target_scaled[test_rows, None]

    # The dictionary keys below are the data contract shared with train.py,
    # explain.py, predict_future.py, export_results.py and leakage_audit.py.
    return {
        "horizon": horizon,
        "lookback": lookback,
        "feature_cols": feature_cols,
        "X_train": x_train,
        "y_train": y_train,
        "train_target_rows": train_origins[:, None] + target_offsets,
        "X_val": x_validation,
        "y_val": y_validation,
        "val_target_rows": validation_origins[:, None] + target_offsets,
        "X_test_by_horizon": x_test_by_horizon,
        "origins_test_by_horizon": origins_test_by_horizon,
        "y_test": np.repeat(y_test_scaled, horizon, axis=1),
        "y_test_orig": y_test_original,
        "y_test_orig_by_horizon": {
            str(step): y_test_original.copy() for step in range(1, horizon + 1)
        },
        "rows_test": test_rows,
        "weeks_by_horizon": {
            str(step): list(weeks) for step in range(1, horizon + 1)
        },
        "feat_last_orig": feature_last_original_by_horizon,
        "feat_last_orig_by_horizon": feature_last_original_by_horizon,
        "feat_scaler": feature_scaler,
        "tgt_scaler": target_scaler,
        "tgt_min": target_min,
        "tgt_max": target_max,
        "tgt_range": target_max - target_min,
        "weeks": weeks,
    }


def inverse_target(scaled: np.ndarray, target_scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transform normalized predictions back to PHP/kg."""
    scaled = np.asarray(scaled)
    return target_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape)

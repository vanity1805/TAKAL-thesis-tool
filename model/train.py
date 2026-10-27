# =============================================================================
# Program      : TAKAL Stacked LSTM training (model/train.py)
# Description  : Builds, trains and lightly tunes the 3-layer Stacked LSTM that
#                forecasts the next 12 weekly NCR retail rice prices.
# Author       : TAKAL Thesis Team (contact: chavezearljustineb@gmail.com)
# System Fit   : Third stage of the offline pipeline (python -m model.run_all).
#                It consumes the bundle made by preprocessing.py and returns a
#                trained model that export_results.py, predict_future.py and
#                explain.py then use for evaluation, forecasting and SHAP.
# Date Written : 2026
# Date Revised : 2026-10-27
# Purpose      : Train the multivariate (6 indicators) and univariate (price
#                only) models under identical settings and choose the lookback
#                window L that generalizes best, without ever touching the test
#                weeks.
# Algorithms   : Stacked LSTM (64 -> 32 -> 16 units, Dropout 0.2 after the first
#                two layers, Dense(12) output); Adam optimizer on mean squared
#                error; early stopping on validation loss; grid search over L
#                averaged across several random seeds.
# Data         : NumPy tensors (samples, timesteps, features); a dict
# Structures     {lookback: mean_validation_loss} for the search grid; lists of
#                per-seed losses; one result dictionary per experiment holding
#                the model, the data bundle and the training history.
# =============================================================================
"""Stacked LSTM construction, training, and light lookback-window (L) tuning.

Architecture (per thesis): three sequential LSTM layers with Dropout after the
first two, a Dense output, Adam optimizer, MSE loss, and Early Stopping. Only the
lookback window ``L`` is tuned; every other hyperparameter is fixed in config.
"""

from __future__ import annotations

import os
import random

import numpy as np

# Keep TensorFlow quiet before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402

from . import config, preprocessing


def set_seeds(seed: int = config.SEED) -> None:
    """Fix every random number generator so a training run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_model(lookback: int, n_features: int) -> Sequential:
    """Three stacked LSTM layers (Dropout after the first two) -> Dense(12)."""
    units = config.LSTM_UNITS
    model = Sequential(name="stacked_lstm")
    model.add(Input(shape=(lookback, n_features)))
    model.add(LSTM(units[0], return_sequences=True))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(LSTM(units[1], return_sequences=True))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(LSTM(units[2], return_sequences=False))
    model.add(Dense(config.FORECAST_HORIZON))
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss="mse")
    return model


def run_experiment(weekly_data, feature_cols: list[str], lookback: int, verbose: int = 0,
                   seed: int = config.SEED) -> dict:
    """Build the supervised bundle for ``lookback`` and train one Stacked LSTM.

    Returns the trained model, the data bundle, the training history, and the
    best validation loss (used for lookback selection).
    """
    set_seeds(seed)
    bundle = preprocessing.build_supervised(weekly_data, feature_cols, lookback)
    model = build_model(lookback, len(feature_cols))

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        bundle["X_train"], bundle["y_train"],
        validation_data=(bundle["X_val"], bundle["y_val"]),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    validation_loss = float(min(history.history["val_loss"]))
    return {
        "lookback": lookback,
        "model": model,
        "bundle": bundle,
        "history": {name: [float(value) for value in values] for name, values in history.history.items()},
        "val_loss": validation_loss,
        "epochs_run": len(history.history["loss"]),
    }


def select_lookback(weekly_data, feature_cols: list[str], verbose: int = 0) -> tuple[dict, dict]:
    """Seed-averaged lookback tuning.

    For every candidate L, train the multivariate model (``feature_cols``) and the
    univariate model once per seed in ``config.SELECTION_SEEDS`` and average the
    best validation losses. L is the value with the lowest mean of the two models'
    averaged losses, so the shared L favours neither model and is not decided by
    one lucky or unlucky random initialisation. The test set is never touched.

    Returns (best_multivariate_experiment, {L: mean_val_loss}). The returned
    experiment is the multivariate model at the chosen L trained with
    ``config.SEED`` (which must be one of ``SELECTION_SEEDS``).
    """
    univariate_cols = [config.TARGET_COL]
    validation_loss_by_lookback: dict[int, float] = {}
    best_lookback, best_score = None, float("inf")
    for lookback in config.L_GRID:
        multivariate_losses, univariate_losses = [], []
        for seed in config.SELECTION_SEEDS:
            multivariate_run = run_experiment(weekly_data, feature_cols, lookback, verbose=verbose, seed=seed)
            univariate_run = run_experiment(weekly_data, univariate_cols, lookback, verbose=verbose, seed=seed)
            multivariate_losses.append(multivariate_run["val_loss"])
            univariate_losses.append(univariate_run["val_loss"])
        score = 0.5 * (float(np.mean(multivariate_losses)) + float(np.mean(univariate_losses)))
        validation_loss_by_lookback[lookback] = round(score, 6)
        if score < best_score:
            best_lookback, best_score = lookback, score
    best_experiment = run_experiment(
        weekly_data, feature_cols, best_lookback, verbose=verbose, seed=config.SEED
    )
    return best_experiment, validation_loss_by_lookback

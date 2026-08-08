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
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(lookback: int, n_features: int) -> Sequential:
    """Three stacked LSTM layers (Dropout after the first two) -> Dense(1)."""
    units = config.LSTM_UNITS
    model = Sequential(name="stacked_lstm")
    model.add(Input(shape=(lookback, n_features)))
    model.add(LSTM(units[0], return_sequences=True))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(LSTM(units[1], return_sequences=True))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(LSTM(units[2], return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss="mse")
    return model


def run_experiment(df, feature_cols: list[str], lookback: int, verbose: int = 0) -> dict:
    """Build the supervised bundle for ``lookback`` and train one Stacked LSTM.

    Returns the trained model, the data bundle, the training history, and the
    best validation loss (used for lookback selection).
    """
    set_seeds()
    bundle = preprocessing.build_supervised(df, feature_cols, lookback)
    model = build_model(lookback, len(feature_cols))

    es = EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        bundle["X_train"], bundle["y_train"],
        validation_data=(bundle["X_val"], bundle["y_val"]),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[es],
        verbose=verbose,
    )
    val_loss = float(min(history.history["val_loss"]))
    return {
        "lookback": lookback,
        "model": model,
        "bundle": bundle,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "val_loss": val_loss,
        "epochs_run": len(history.history["loss"]),
    }


def select_lookback(df, feature_cols: list[str], verbose: int = 0) -> tuple[dict, dict]:
    """Light tuning: train once per candidate L, keep the model with the lowest
    validation loss. Returns (best_experiment, {L: val_loss} grid record)."""
    grid: dict[int, float] = {}
    best: dict | None = None
    for lookback in config.L_GRID:
        result = run_experiment(df, feature_cols, lookback, verbose=verbose)
        grid[lookback] = result["val_loss"]
        if best is None or result["val_loss"] < best["val_loss"]:
            best = result
    return best, grid

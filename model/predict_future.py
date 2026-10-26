"""Operational out-of-sample 12-week forward forecast.

The evaluation pipeline scores rolling-origin forecasts of weeks that are already
in the dataset. This module does the operational step on top of the *same trained
models*: it takes the final ``L`` observed weeks, predicts the next 12 weekly
prices with both Stacked LSTM variants, and explains the multivariate forecast
with SHAP. It is called from ``export_results.build_results`` so the forecast
always comes from the exact models, scalers and lookback that produced the
reported metrics (nothing is re-fit or re-tuned here).

Steps (per model):
  1. Take the last ``L`` rows of the prepared master dataset (``L`` = the lookback
     the models were trained with; it is read from the trained bundle, never
     hard-coded).
  2. Scale them with the Min-Max scaler fitted on the training rows only.
  3. One forward pass through the model: the Dense(12) head returns all 12 scaled
     horizons at once (direct vector prediction, no recursive feeding).
  4. Inverse-transform to PHP/kg with the training target scaler.
  5. Date the 12 horizons weekly (``W-MON``) from the last observed date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, explain, preprocessing

DATE_FORMAT = "%Y-%m-%d"
ADDITIVITY_TOLERANCE = 1e-4   # PHP/kg; base + sum(SHAP) vs. prediction after rounding


# --------------------------------------------------------------------------- #
# Input window
# --------------------------------------------------------------------------- #
def extract_input_window(df: pd.DataFrame, feature_cols: list[str], lookback: int) -> np.ndarray:
    """Return the final ``lookback`` rows of ``feature_cols`` as raw values ``(L, F)``."""
    if len(df) < lookback:
        raise ValueError(f"Need at least {lookback} rows for the input window; found {len(df)}.")
    window = df.iloc[-lookback:]
    gaps = window[config.DATE_COL].diff().dt.days.iloc[1:]
    if (gaps != config.EXPECTED_WEEKLY_STEP_DAYS).any():
        print(
            f"[future] WARNING: the final {lookback}-week input window is not exactly "
            f"{config.EXPECTED_WEEKLY_STEP_DAYS} days apart; the model assumes weekly steps."
        )
    raw = window[feature_cols].to_numpy(dtype="float64")
    if not np.isfinite(raw).all():
        raise ValueError("The final input window contains missing or non-finite values.")
    return raw


def scale_window(raw_window: np.ndarray, feat_scaler) -> np.ndarray:
    """Scale a raw ``(L, F)`` window with the *training-fitted* scaler -> ``(1, L, F)``.

    Mirrors ``preprocessing.build_supervised`` exactly: ``transform`` only (never
    ``fit``), followed by the optional shock clip controlled by
    ``config.CLIP_INPUTS_TO_TRAIN_RANGE``.
    """
    scaled = feat_scaler.transform(raw_window)
    if config.CLIP_INPUTS_TO_TRAIN_RANGE:
        scaled = np.clip(scaled, 0.0, 1.0)
    return scaled[np.newaxis, ...]


def _assert_scaler_consistency(df: pd.DataFrame, bundle: dict, label: str) -> None:
    """Fail fast if the future window would be scaled differently from training/testing.

    Checks (1) the scaler saw exactly the pre-test rows, and (2) re-scaling the
    window one week earlier reproduces the last horizon-1 test window already
    stored in the bundle, which proves the same scaler and clipping are applied.
    """
    cols, lookback = bundle["feature_cols"], bundle["lookback"]
    scaler = bundle["feat_scaler"]
    split = preprocessing.split_row(len(df))
    if int(scaler.n_samples_seen_) != split:
        raise AssertionError(
            f"{label}: scaler was fitted on {scaler.n_samples_seen_} rows, expected the "
            f"{split} pre-test rows."
        )
    raw = df[cols].to_numpy(dtype="float64")
    previous = scale_window(raw[-lookback - 1:-1], scaler)[0]
    stored = bundle["X_test_by_horizon"]["1"][-1]
    if not np.allclose(previous, stored, atol=1e-9):
        raise AssertionError(f"{label}: future-window scaling differs from the test-window scaling.")


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def future_dates(last_date, horizon: int = config.FORECAST_HORIZON) -> pd.DatetimeIndex:
    """The next ``horizon`` weekly target dates after ``last_date`` (``freq='W-MON'``).

    The range starts the day after the last observed date, so the first target is
    the first Monday strictly after it (exactly one week later when the dataset
    already ends on a Monday, as the master dataset does).
    """
    last = pd.Timestamp(last_date)
    if last.dayofweek != 0:
        print(
            f"[future] NOTE: the last observed date {last.date()} is a {last.day_name()}; "
            "future target dates are snapped to Mondays (freq='W-MON')."
        )
    return pd.date_range(start=last + pd.Timedelta(days=1), periods=horizon, freq="W-MON")


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_window(model, bundle: dict, X_future: np.ndarray) -> np.ndarray:
    """One forward pass of the Dense(12) head, returned in PHP/kg ``(12,)``."""
    scaled = np.asarray(model.predict(X_future, verbose=0))
    if scaled.shape != (1, config.FORECAST_HORIZON):
        raise ValueError(
            f"Expected a (1, {config.FORECAST_HORIZON}) prediction vector; got {scaled.shape}."
        )
    return preprocessing.inverse_target(scaled[0], bundle["tgt_scaler"])


def _out_of_range_note(raw_window: np.ndarray, bundle: dict, feature_cols: list[str]) -> str:
    """Describe final-window inputs that lie outside the training range."""
    names = dict(config.MULTIVARIATE_FEATURES)
    scaler = bundle["feat_scaler"]
    parts = []
    for index, column in enumerate(feature_cols):
        low, high = float(scaler.data_min_[index]), float(scaler.data_max_[index])
        n_out = int(((raw_window[:, index] < low) | (raw_window[:, index] > high)).sum())
        if n_out:
            parts.append(
                f"{names.get(column, column)}: {n_out} of the {len(raw_window)} input weeks "
                f"fall outside the training range [{low:.4g}, {high:.4g}]."
            )
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def assert_additive(future: dict, tolerance: float = ADDITIVITY_TOLERANCE) -> None:
    """Check that ``base_value + sum(SHAP)`` equals the multivariate prediction each week."""
    base = float(future["base_value"])
    for date, predicted in zip(future["horizon_dates"], future["multivariate_predictions"]):
        total = base + sum(future["shap_attributions"]["multivariate"][date].values())
        if abs(total - float(predicted)) > tolerance:
            raise AssertionError(
                f"SHAP is not additive for {date}: base {base} + contributions {total - base:.5f} "
                f"= {total:.5f}, prediction {predicted}."
            )


def build_future_forecast(df: pd.DataFrame, mv_experiment: dict, uni_experiment: dict) -> tuple[dict, str]:
    """Build the ``future_forecast`` artifact block from the two trained experiments.

    ``df`` is the prepared master dataset; each experiment is a
    ``train.run_experiment`` result (``model`` + ``bundle``). Returns the
    ``future_forecast`` dictionary and a plain-language note about final-window
    inputs outside the training range (empty if there are none).
    """
    mv_bundle, uni_bundle = mv_experiment["bundle"], uni_experiment["bundle"]
    lookback = mv_bundle["lookback"]
    if uni_bundle["lookback"] != lookback:
        raise AssertionError("The multivariate and univariate models use different L values.")

    mv_cols, uni_cols = mv_bundle["feature_cols"], uni_bundle["feature_cols"]
    _assert_scaler_consistency(df, mv_bundle, "multivariate")
    _assert_scaler_consistency(df, uni_bundle, "univariate")

    raw_mv = extract_input_window(df, mv_cols, lookback)
    raw_uni = extract_input_window(df, uni_cols, lookback)
    X_mv = scale_window(raw_mv, mv_bundle["feat_scaler"])
    X_uni = scale_window(raw_uni, uni_bundle["feat_scaler"])

    pred_mv = predict_window(mv_experiment["model"], mv_bundle, X_mv)
    pred_uni = predict_window(uni_experiment["model"], uni_bundle, X_uni)

    last_date = pd.Timestamp(df[config.DATE_COL].iloc[-1])
    dates = [stamp.strftime(DATE_FORMAT) for stamp in future_dates(last_date)]

    shap_future = explain.explain_future(
        mv_experiment["model"], mv_bundle, X_mv, pred_mv, mv_cols
    )

    future = {
        "origin_date": last_date.strftime(DATE_FORMAT),
        "horizon_dates": dates,
        "multivariate_predictions": [round(float(value), 4) for value in pred_mv],
        "univariate_predictions": [round(float(value), 4) for value in pred_uni],
        "base_value": shap_future["base_value"],
        "shap_attributions": {
            "multivariate": {
                date: contributions
                for date, contributions in zip(dates, shap_future["contributions"])
            }
        },
    }
    assert_additive(future)

    note = _out_of_range_note(raw_mv, mv_bundle, mv_cols)
    if note:
        print(f"[future] WARNING: {note} The multivariate forward forecast extrapolates beyond "
              "the range the model was trained on.")
    return future, note

# =============================================================================
# Program      : TAKAL 12-week forward forecast (model/predict_future.py)
# Description  : Uses the already-trained 3-layer Stacked LSTM models to predict
#                the NCR retail rice price for each of the next 12 weeks, and
#                explains the multivariate forecast with SHAP.
# Author       : TAKAL Thesis Team (contact: chavezearljustineb@gmail.com)
# System Fit   : Final modelling step of the offline pipeline. export_results.py
#                calls build_future_forecast() after training and evaluation;
#                the returned block is stored under "future_forecast" in
#                artifacts/results.json and shown on the dashboard's 12-Week
#                Forecast tab. Failures are raised as ForecastInferenceError
#                (LSTM step) or ShapAttributionError (SHAP step) so the
#                dashboard can show a formatted message.
# Date Written : 2026
# Date Revised : 2026-10-27
# Purpose      : Produce the operational forecast (weeks that have not happened
#                yet) from the exact models, scalers and lookback that produced
#                the reported accuracy figures. Nothing is re-fitted here.
# Algorithms   : Direct multi-horizon forecasting: one forward pass of the
#                Stacked LSTM returns all 12 weekly prices at once (no
#                recursive feeding); Min-Max scaling with the training-fitted
#                scaler and its inverse; SHAP additivity check (baseline + sum
#                of contributions = prediction); weekly (Monday) date
#                generation.
# Data         : NumPy arrays: raw window (L, F), scaled model input (1, L, F),
# Structures     prediction vector (12,); one dictionary "future_forecast" with
#                parallel lists (dates, predictions) and a nested dictionary of
#                SHAP contributions keyed by date then by feature name.
# =============================================================================
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
from .errors import ForecastInferenceError, ShapAttributionError

DATE_FORMAT = "%Y-%m-%d"
ADDITIVITY_TOLERANCE = 1e-4   # PHP/kg; base + sum(SHAP) vs. prediction after rounding
INFERENCE_RETRY_HINT = (
    "Re-run `python -m model.run_all` to retrain the models. If the problem persists, "
    "check that THESIS-MASTER_DATASET.xlsx still has the same columns as before."
)


# --------------------------------------------------------------------------- #
# Input window
# --------------------------------------------------------------------------- #
def extract_input_window(weekly_data: pd.DataFrame, feature_cols: list[str], lookback: int) -> np.ndarray:
    """Return the final ``lookback`` rows of ``feature_cols`` as raw values ``(L, F)``."""
    if len(weekly_data) < lookback:
        raise ValueError(
            f"Need at least {lookback} rows for the input window; found {len(weekly_data)}."
        )
    window = weekly_data.iloc[-lookback:]
    gaps_in_days = window[config.DATE_COL].diff().dt.days.iloc[1:]
    if (gaps_in_days != config.EXPECTED_WEEKLY_STEP_DAYS).any():
        print(
            f"[future] WARNING: the final {lookback}-week input window is not exactly "
            f"{config.EXPECTED_WEEKLY_STEP_DAYS} days apart; the model assumes weekly steps."
        )
    raw_window = window[feature_cols].to_numpy(dtype="float64")
    if not np.isfinite(raw_window).all():
        raise ValueError("The final input window contains missing or non-finite values.")
    return raw_window


def scale_window(raw_window: np.ndarray, feature_scaler) -> np.ndarray:
    """Scale a raw ``(L, F)`` window with the *training-fitted* scaler -> ``(1, L, F)``.

    Mirrors ``preprocessing.build_supervised`` exactly: ``transform`` only (never
    ``fit``), followed by the optional shock clip controlled by
    ``config.CLIP_INPUTS_TO_TRAIN_RANGE``.
    """
    scaled_window = feature_scaler.transform(raw_window)
    if config.CLIP_INPUTS_TO_TRAIN_RANGE:
        scaled_window = np.clip(scaled_window, 0.0, 1.0)
    return scaled_window[np.newaxis, ...]


def _assert_scaler_consistency(weekly_data: pd.DataFrame, bundle: dict, label: str) -> None:
    """Fail fast if the future window would be scaled differently from training/testing.

    Checks (1) the scaler saw exactly the pre-test rows, and (2) re-scaling the
    window one week earlier reproduces the last horizon-1 test window already
    stored in the bundle, which proves the same scaler and clipping are applied.
    """
    feature_cols, lookback = bundle["feature_cols"], bundle["lookback"]
    feature_scaler = bundle["feat_scaler"]
    split_index = preprocessing.split_row(len(weekly_data))
    if int(feature_scaler.n_samples_seen_) != split_index:
        raise AssertionError(
            f"{label}: scaler was fitted on {feature_scaler.n_samples_seen_} rows, expected the "
            f"{split_index} pre-test rows."
        )
    raw_values = weekly_data[feature_cols].to_numpy(dtype="float64")
    previous_window = scale_window(raw_values[-lookback - 1:-1], feature_scaler)[0]
    stored_window = bundle["X_test_by_horizon"]["1"][-1]
    if not np.allclose(previous_window, stored_window, atol=1e-9):
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
    last_timestamp = pd.Timestamp(last_date)
    if last_timestamp.dayofweek != 0:
        print(
            f"[future] NOTE: the last observed date {last_timestamp.date()} is a "
            f"{last_timestamp.day_name()}; future target dates are snapped to Mondays "
            "(freq='W-MON')."
        )
    return pd.date_range(
        start=last_timestamp + pd.Timedelta(days=1), periods=horizon, freq="W-MON"
    )


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_window(model, bundle: dict, x_future: np.ndarray) -> np.ndarray:
    """One forward pass of the Dense(12) head, returned in PHP/kg ``(12,)``.

    Raises ``ForecastInferenceError`` if the Stacked LSTM cannot run on the window
    or returns anything other than a ``(1, 12)`` prediction vector.
    """
    try:
        scaled_predictions = np.asarray(model.predict(x_future, verbose=0))
        if scaled_predictions.shape != (1, config.FORECAST_HORIZON):
            raise ValueError(
                f"Expected a (1, {config.FORECAST_HORIZON}) prediction vector; "
                f"got {scaled_predictions.shape}."
            )
        predictions_php = preprocessing.inverse_target(scaled_predictions[0], bundle["tgt_scaler"])
        if not np.isfinite(predictions_php).all():
            raise ValueError("The Stacked LSTM returned missing or non-finite prices.")
    except Exception as error:  # TensorFlow raises many unrelated exception types.
        raise ForecastInferenceError(
            f"The Stacked LSTM could not produce a 12-week forecast: "
            f"{type(error).__name__}: {error}",
            hint=INFERENCE_RETRY_HINT,
        ) from error
    return predictions_php


def _out_of_range_note(raw_window: np.ndarray, bundle: dict, feature_cols: list[str]) -> str:
    """Describe final-window inputs that lie outside the training range."""
    display_names = dict(config.MULTIVARIATE_FEATURES)
    feature_scaler = bundle["feat_scaler"]
    note_parts = []
    for index, column in enumerate(feature_cols):
        low, high = float(feature_scaler.data_min_[index]), float(feature_scaler.data_max_[index])
        weeks_outside = int(((raw_window[:, index] < low) | (raw_window[:, index] > high)).sum())
        if weeks_outside:
            note_parts.append(
                f"{display_names.get(column, column)}: {weeks_outside} of the {len(raw_window)} "
                f"input weeks fall outside the training range [{low:.4g}, {high:.4g}]."
            )
    return " ".join(note_parts)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def assert_additive(future: dict, tolerance: float = ADDITIVITY_TOLERANCE) -> None:
    """Check that ``base_value + sum(SHAP)`` equals the multivariate prediction each week."""
    base_value = float(future["base_value"])
    for date, predicted in zip(future["horizon_dates"], future["multivariate_predictions"]):
        total = base_value + sum(future["shap_attributions"]["multivariate"][date].values())
        if abs(total - float(predicted)) > tolerance:
            raise AssertionError(
                f"SHAP is not additive for {date}: base {base_value} + contributions "
                f"{total - base_value:.5f} = {total:.5f}, prediction {predicted}."
            )


def build_future_forecast(
    weekly_data: pd.DataFrame, multivariate_experiment: dict, univariate_experiment: dict
) -> tuple[dict, str]:
    """Build the ``future_forecast`` artifact block from the two trained experiments.

    ``weekly_data`` is the prepared master dataset; each experiment is a
    ``train.run_experiment`` result (``model`` + ``bundle``). Returns the
    ``future_forecast`` dictionary and a plain-language note about final-window
    inputs outside the training range (empty if there are none).

    Raises ``ForecastInferenceError`` if the latest input window cannot be
    prepared or the Stacked LSTM fails, and ``ShapAttributionError`` if the SHAP
    explanation fails or does not add up to the prediction.
    """
    multivariate_bundle = multivariate_experiment["bundle"]
    univariate_bundle = univariate_experiment["bundle"]
    lookback = multivariate_bundle["lookback"]
    multivariate_cols = multivariate_bundle["feature_cols"]
    univariate_cols = univariate_bundle["feature_cols"]

    # Step 1: prepare the latest input window for both models.
    try:
        if univariate_bundle["lookback"] != lookback:
            raise AssertionError("The multivariate and univariate models use different L values.")
        _assert_scaler_consistency(weekly_data, multivariate_bundle, "multivariate")
        _assert_scaler_consistency(weekly_data, univariate_bundle, "univariate")

        raw_multivariate = extract_input_window(weekly_data, multivariate_cols, lookback)
        raw_univariate = extract_input_window(weekly_data, univariate_cols, lookback)
        x_multivariate = scale_window(raw_multivariate, multivariate_bundle["feat_scaler"])
        x_univariate = scale_window(raw_univariate, univariate_bundle["feat_scaler"])
    except (AssertionError, ValueError, KeyError) as error:
        raise ForecastInferenceError(
            f"The latest {lookback}-week input window could not be prepared: "
            f"{type(error).__name__}: {error}",
            stage="Preparing the latest input window",
            hint="Check the newest rows of THESIS-MASTER_DATASET.xlsx for blanks or "
                 "irregular dates, then run `python -m model.run_all` again.",
        ) from error

    # Step 2: Stacked LSTM inference (each call raises ForecastInferenceError itself).
    predicted_multivariate = predict_window(
        multivariate_experiment["model"], multivariate_bundle, x_multivariate
    )
    predicted_univariate = predict_window(
        univariate_experiment["model"], univariate_bundle, x_univariate
    )

    last_date = pd.Timestamp(weekly_data[config.DATE_COL].iloc[-1])
    dates = [stamp.strftime(DATE_FORMAT) for stamp in future_dates(last_date)]

    # Step 3: SHAP attribution (explain_future raises ShapAttributionError itself).
    shap_future = explain.explain_future(
        multivariate_experiment["model"],
        multivariate_bundle,
        x_multivariate,
        predicted_multivariate,
        multivariate_cols,
    )

    future = {
        "origin_date": last_date.strftime(DATE_FORMAT),
        "horizon_dates": dates,
        "multivariate_predictions": [round(float(value), 4) for value in predicted_multivariate],
        "univariate_predictions": [round(float(value), 4) for value in predicted_univariate],
        "base_value": shap_future["base_value"],
        "shap_attributions": {
            "multivariate": {
                date: contributions
                for date, contributions in zip(dates, shap_future["contributions"])
            }
        },
    }
    try:
        assert_additive(future)
    except AssertionError as error:
        raise ShapAttributionError(
            f"The SHAP explanation does not add up to the forecast: {error}",
            hint="Re-run `python -m model.run_all`; if it repeats, raise "
                 "config.FUTURE_SHAP_SAMPLES to reduce the SHAP approximation error.",
        ) from error

    note = _out_of_range_note(raw_multivariate, multivariate_bundle, multivariate_cols)
    if note:
        print(f"[future] WARNING: {note} The multivariate forward forecast extrapolates beyond "
              "the range the model was trained on.")
    return future, note

# =============================================================================
# Program      : TAKAL results exporter (model/export_results.py)
# Description  : Runs the modelling steps in order and assembles everything the
#                dashboard needs (forecasts, accuracy scores, significance
#                tests, SHAP explanations, 12-week forecast) into one
#                horizon-keyed results dictionary / JSON file. It also writes the
#                small pipeline_status.json that tells the dashboard whether the
#                last run worked.
# Author       : TAKAL Thesis Team (contact: chavezearljustineb@gmail.com)
# System Fit   : Orchestration layer of the offline pipeline. run_all.py calls
#                build_results() and save(); the Streamlit app (app.py) only
#                ever reads the files written here.
# Date Written : 2026
# Date Revised : 2026-10-27
# Purpose      : Keep the dashboard decoupled from TensorFlow: heavy work happens
#                once offline, and any failure is turned into a typed error with
#                a plain-language message and hint.
# Algorithms   : Pipeline orchestration; leakage and alignment assertions;
#                per-horizon RMSE / MAE / MAPE / R-squared; Diebold-Mariano test
#                on squared-error loss differentials; batch Stacked LSTM
#                inference over all 12 horizons; atomic file replacement for the
#                status file.
# Data         : Nested dictionaries keyed by horizon ("1" .. "12", "overall")
# Structures     then by model name; NumPy arrays shaped (test weeks, 12) for
#                predictions and loss differentials; lists of rounded floats for
#                the JSON output.
# =============================================================================
"""Assemble model outputs into the dashboard's horizon-keyed results.json."""

from __future__ import annotations

import datetime as _dt
import json
import os
import traceback

import numpy as np

from . import config, data_prep, evaluate, explain, predict_future, preprocessing, train
from .errors import (
    DatasetLoadError,
    ForecastInferenceError,
    TakalPipelineError,
    pipeline_stage,
)


def _predict_php_by_horizon(model, bundle: dict) -> np.ndarray:
    """Return aligned predictions shaped (test weeks, forecast horizons).

    Raises ``ForecastInferenceError`` (naming the horizon) if the Stacked LSTM
    fails on any test window.
    """
    predictions = []
    for horizon in range(1, config.FORECAST_HORIZON + 1):
        key = str(horizon)
        try:
            scaled_predictions = model.predict(bundle["X_test_by_horizon"][key], verbose=0)
            predictions.append(
                bundle["tgt_scaler"].inverse_transform(scaled_predictions[:, [horizon - 1]])[:, 0]
            )
        except Exception as error:  # TensorFlow raises many unrelated exception types.
            raise ForecastInferenceError(
                f"The Stacked LSTM failed on the held-out test weeks at horizon {horizon}: "
                f"{type(error).__name__}: {error}",
                stage="Stacked LSTM inference on the test weeks",
                hint="Re-run `python -m model.run_all` to retrain the models.",
            ) from error
    return np.column_stack(predictions)


def _assert_split_and_alignment(
    multivariate_bundle: dict, univariate_bundle: dict, split_index: int, n_test: int
) -> None:
    """Fail fast if the no-leakage or paired-test-window invariants are violated."""
    for label, bundle in (("multivariate", multivariate_bundle), ("univariate", univariate_bundle)):
        if np.any(bundle["train_target_rows"] >= split_index):
            raise AssertionError(f"{label} training target reaches the test region.")
        if np.any(bundle["val_target_rows"] >= split_index):
            raise AssertionError(f"{label} validation target reaches the test region.")
        if len(bundle["rows_test"]) != n_test:
            raise AssertionError(
                f"{label} test set has {len(bundle['rows_test'])} targets, expected {n_test}."
            )
        for horizon in range(1, config.FORECAST_HORIZON + 1):
            key = str(horizon)
            expected_origins = bundle["rows_test"] - horizon + 1
            actual_origins = bundle["origins_test_by_horizon"][key]
            if not np.array_equal(actual_origins, expected_origins):
                raise AssertionError(f"Horizon {horizon} uses incorrect forecast origins.")
            if np.any(actual_origins > bundle["rows_test"]):
                raise AssertionError(f"Horizon {horizon} uses a future input row.")
            if len(bundle["X_test_by_horizon"][key]) != n_test:
                raise AssertionError(f"Horizon {horizon} does not have {n_test} test windows.")
            if len(bundle["y_test_orig_by_horizon"][key]) != n_test:
                raise AssertionError(f"Horizon {horizon} does not have {n_test} test targets.")
            if bundle["weeks_by_horizon"][key] != bundle["weeks"]:
                raise AssertionError(f"Horizon {horizon} uses different target dates.")
            if len(bundle["feat_last_orig"][key]) != n_test:
                raise AssertionError(f"Horizon {horizon} lacks aligned last-input features.")
    if multivariate_bundle["lookback"] != univariate_bundle["lookback"]:
        raise AssertionError("The multivariate and univariate models use different L values.")
    if multivariate_bundle["weeks"] != univariate_bundle["weeks"]:
        raise AssertionError("The two experiments do not share the same target dates.")


def _data_note(weekly_data, feature_cols: list[str], split_index: int) -> str:
    """Describe, from the data itself, what a reader should double-check."""
    note_parts = []
    for column, weeks_filled in weekly_data.attrs.get("imputed_weeks", {}).items():
        note_parts.append(
            f"{column}: {weeks_filled} trailing week(s) forward-filled with the last observed value."
        )
    for column in feature_cols:
        training_values = weekly_data[column].iloc[:split_index]
        test_values = weekly_data[column].iloc[split_index:]
        weeks_outside = int(
            ((test_values < training_values.min()) | (test_values > training_values.max())).sum()
        )
        if weeks_outside:
            note_parts.append(
                f"{column}: {weeks_outside} test week(s) fall outside the training range "
                f"[{training_values.min():.4g}, {training_values.max():.4g}]; verify the source."
            )
    return " ".join(note_parts) or "No imputation or out-of-range inputs detected."


def build_results(verbose: int = 0) -> dict:
    """Run the full pipeline and return the assembled results dictionary.

    Every expected failure is raised as a ``TakalPipelineError`` subclass:
    ``DatasetLoadError`` (data), ``ForecastInferenceError`` (Stacked LSTM) or
    ``ShapAttributionError`` (SHAP). Training problems are labelled with their
    pipeline stage by ``pipeline_stage``.
    """
    # Stage 1: load and validate the NCR weekly rice price dataset.
    try:
        weekly_data = data_prep.prepare()
    except TakalPipelineError:
        raise
    except (ValueError, KeyError) as error:
        raise DatasetLoadError(
            f"The NCR rice price dataset failed validation: {error}",
            hint="Fix the dataset described above (see model/README.md, 'Adding new weeks "
                 "of data'), then run `python -m model.run_all` again.",
        ) from error

    # Stage 2: tune L on the multivariate validation windows; use that same L for both.
    multivariate_cols = [column for column, _ in config.MULTIVARIATE_FEATURES]
    multivariate_names = [name for _, name in config.MULTIVARIATE_FEATURES]
    with pipeline_stage("Training the Stacked LSTM models"):
        best_multivariate, validation_loss_by_lookback = train.select_lookback(
            weekly_data, multivariate_cols, verbose=verbose
        )
        lookback = best_multivariate["lookback"]
        print(f"Chosen lookback L: {lookback}")
        univariate = train.run_experiment(
            weekly_data, [config.TARGET_COL], lookback, verbose=verbose
        )
    multivariate_bundle, univariate_bundle = best_multivariate["bundle"], univariate["bundle"]
    split_index = preprocessing.split_row(len(weekly_data))
    n_test = len(weekly_data) - split_index
    if n_test < config.MIN_TEST_WEEKS:
        raise DatasetLoadError(
            f"Only {n_test} test weeks; at least {config.MIN_TEST_WEEKS} are needed "
            "for a meaningful evaluation.",
            hint="Add more weekly rows to the dataset (see config.MIN_TEST_WEEKS).",
        )
    _assert_split_and_alignment(multivariate_bundle, univariate_bundle, split_index, n_test)

    # Stage 3: Stacked LSTM inference on every held-out test week.
    weeks = multivariate_bundle["weeks"]
    actual = multivariate_bundle["y_test_orig"]
    predicted_multivariate = _predict_php_by_horizon(best_multivariate["model"], multivariate_bundle)
    predicted_univariate = _predict_php_by_horizon(univariate["model"], univariate_bundle)
    expected_shape = (n_test, config.FORECAST_HORIZON)
    if (predicted_multivariate.shape != predicted_univariate.shape
            or predicted_multivariate.shape != expected_shape):
        raise ForecastInferenceError(
            f"Expected paired prediction shape {expected_shape}; "
            f"got {predicted_multivariate.shape} and {predicted_univariate.shape}.",
            hint="Re-run `python -m model.run_all` to retrain the models.",
        )

    # Stage 4: accuracy metrics and the Diebold-Mariano significance test.
    metrics: dict[str, dict] = {}
    dm_tests: dict[str, dict] = {}
    loss_differentials = np.empty_like(predicted_multivariate)
    for horizon in range(1, config.FORECAST_HORIZON + 1):
        column_index = horizon - 1
        key = str(horizon)
        metrics[key] = {
            "multivariate": evaluate.regression_metrics(actual, predicted_multivariate[:, column_index]),
            "univariate": evaluate.regression_metrics(actual, predicted_univariate[:, column_index]),
        }
        error_multivariate = actual - predicted_multivariate[:, column_index]
        error_univariate = actual - predicted_univariate[:, column_index]
        loss_differentials[:, column_index] = error_multivariate ** 2 - error_univariate ** 2
        dm_tests[key] = evaluate.diebold_mariano(
            error_multivariate, error_univariate, h=horizon, power=2
        )

    pooled_actual = np.tile(actual[:, None], (1, config.FORECAST_HORIZON)).ravel()
    metrics["overall"] = {
        "multivariate": evaluate.regression_metrics(pooled_actual, predicted_multivariate.ravel()),
        "univariate": evaluate.regression_metrics(pooled_actual, predicted_univariate.ravel()),
    }
    # Overall DM uses each week's mean squared-error loss differential across all 12
    # horizons, a 12-lag HAC variance (autocovariances 0..11), and the h=12 HLN
    # factor; each horizon-specific DM above uses the same squared loss and lags 0..h-1.
    overall_weekly_differential = loss_differentials.mean(axis=1)
    dm_tests["overall"] = evaluate.diebold_mariano_differential(
        overall_weekly_differential, h=config.FORECAST_HORIZON
    )

    forecasts = {
        str(horizon): {
            "actual": [round(float(value), 8) for value in actual],
            "multivariate": [
                round(float(value), 8) for value in predicted_multivariate[:, horizon - 1]
            ],
            "univariate": [
                round(float(value), 8) for value in predicted_univariate[:, horizon - 1]
            ],
        }
        for horizon in range(1, config.FORECAST_HORIZON + 1)
    }

    # Stage 5: SHAP attribution (explain_model raises ShapAttributionError itself).
    shap_results = {
        "multivariate": explain.explain_model(
            best_multivariate["model"], multivariate_bundle, multivariate_names, "multivariate"
        ),
        "univariate": explain.explain_model(
            univariate["model"], univariate_bundle, [], "univariate"
        ),
    }

    # Stage 6: operational out-of-sample forecast: the final L observed weeks -> the
    # next 12 weeks, from the same trained models and training-fitted scalers as above.
    future_forecast, future_note = predict_future.build_future_forecast(
        weekly_data, best_multivariate, univariate
    )

    results = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "lookback_L": lookback,
            "lookback_grid_val_loss": {
                str(candidate): round(value, 6)
                for candidate, value in validation_loss_by_lookback.items()
            },
            "n_train": int(len(multivariate_bundle["X_train"])),
            "n_val": int(len(multivariate_bundle["X_val"])),
            "n_test": int(len(multivariate_bundle["rows_test"])),
            "features": multivariate_names,
            "dataset_rows": int(len(weekly_data)),
            "horizons": list(range(1, config.FORECAST_HORIZON + 1)),
            "data_note": _data_note(weekly_data, multivariate_cols, split_index),
            "data_range": [
                weekly_data[config.DATE_COL].iloc[0].strftime("%Y-%m-%d"),
                weekly_data[config.DATE_COL].iloc[-1].strftime("%Y-%m-%d"),
            ],
            "future_forecast_note": future_note,
        },
        "weeks": weeks,
        "forecast": forecasts,
        "metrics": metrics,
        "dm_test": dm_tests,
        "shap": {
            str(horizon): {
                model_name: shap_results[model_name][str(horizon)]
                for model_name in ("multivariate", "univariate")
            }
            for horizon in range(1, config.FORECAST_HORIZON + 1)
        },
        "future_forecast": future_forecast,
    }
    return results


def save(results: dict, path: str | None = None) -> str:
    """Write the results dictionary to ``results.json`` and return the path used."""
    path = path or config.RESULTS_JSON
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)
    return path


def write_pipeline_status(
    succeeded: bool,
    error: TakalPipelineError | None = None,
    path: str | None = None,
) -> str | None:
    """Record the outcome of a pipeline run for the Streamlit dashboard.

    The dashboard reads this small JSON file to explain *why* ``results.json`` is
    missing or stale. A failed status carries the error type, the pipeline stage,
    a plain-language message, a hint, the underlying cause and a short traceback.
    The file is replaced atomically. If it cannot be written the problem is only
    printed, so a status-file failure never hides the original error.
    """
    path = path or config.PIPELINE_STATUS_JSON
    payload: dict = {
        "status": "success" if succeeded else "failed",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if error is not None:
        cause = error.__cause__
        payload.update({
            "error_type": type(error).__name__,
            "stage": error.stage,
            "message": error.message,
            "hint": error.hint,
            "cause": f"{type(cause).__name__}: {cause}" if cause is not None else None,
            "traceback": "".join(traceback.format_exception(cause or error))[-4000:],
        })
    temporary_path = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(temporary_path, "w", encoding="utf-8") as status_file:
            json.dump(payload, status_file, indent=2)
        os.replace(temporary_path, path)
    except OSError as write_error:
        print(f"[status] WARNING: could not write {path}: {write_error}")
        return None
    return path


if __name__ == "__main__":
    save(build_results())
    print("Wrote", config.RESULTS_JSON)

"""Assemble model outputs into the dashboard's horizon-keyed results.json."""

from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

from . import config, data_prep, evaluate, explain, preprocessing, train


def _predict_php_by_horizon(model, bundle: dict) -> np.ndarray:
    """Return aligned predictions shaped (test weeks, forecast horizons)."""
    predictions = []
    for horizon in range(1, config.FORECAST_HORIZON + 1):
        key = str(horizon)
        pred = model.predict(bundle["X_test_by_horizon"][key], verbose=0)
        predictions.append(
            bundle["tgt_scaler"].inverse_transform(pred[:, [horizon - 1]])[:, 0]
        )
    return np.column_stack(predictions)


def _assert_split_and_alignment(mv_bundle: dict, uni_bundle: dict, split: int) -> None:
    """Fail fast if the no-leakage or paired-test-window invariants are violated."""
    for label, bundle in (("multivariate", mv_bundle), ("univariate", uni_bundle)):
        if np.any(bundle["train_target_rows"] >= split):
            raise AssertionError(f"{label} training target reaches the test region.")
        if np.any(bundle["val_target_rows"] >= split):
            raise AssertionError(f"{label} validation target reaches the test region.")
        if len(bundle["rows_test"]) != 66:
            raise AssertionError(
                f"{label} test set has {len(bundle['rows_test'])} targets, expected 66."
            )
        for horizon in range(1, config.FORECAST_HORIZON + 1):
            key = str(horizon)
            expected_origins = bundle["rows_test"] - horizon + 1
            actual_origins = bundle["origins_test_by_horizon"][key]
            if not np.array_equal(actual_origins, expected_origins):
                raise AssertionError(f"Horizon {horizon} uses incorrect forecast origins.")
            if np.any(actual_origins > bundle["rows_test"]):
                raise AssertionError(f"Horizon {horizon} uses a future input row.")
            if len(bundle["X_test_by_horizon"][key]) != 66:
                raise AssertionError(f"Horizon {horizon} does not have 66 test windows.")
            if len(bundle["y_test_orig_by_horizon"][key]) != 66:
                raise AssertionError(f"Horizon {horizon} does not have 66 test targets.")
            if bundle["weeks_by_horizon"][key] != bundle["weeks"]:
                raise AssertionError(f"Horizon {horizon} uses different target dates.")
            if len(bundle["feat_last_orig"][key]) != 66:
                raise AssertionError(f"Horizon {horizon} lacks aligned last-input features.")
    if mv_bundle["lookback"] != uni_bundle["lookback"]:
        raise AssertionError("The multivariate and univariate models use different L values.")
    if mv_bundle["weeks"] != uni_bundle["weeks"]:
        raise AssertionError("The two experiments do not share the same target dates.")


def build_results(verbose: int = 0) -> dict:
    """Run the full pipeline and return the assembled results dictionary."""
    df = data_prep.prepare()

    # Tune L on the multivariate validation windows and use that same L for both.
    mv_cols = [column for column, _ in config.MULTIVARIATE_FEATURES]
    mv_names = [name for _, name in config.MULTIVARIATE_FEATURES]
    best_mv, l_grid = train.select_lookback(df, mv_cols, verbose=verbose)
    lookback = best_mv["lookback"]
    print(f"Chosen lookback L: {lookback}")

    uni_cols = [config.TARGET_COL]
    uni = train.run_experiment(df, uni_cols, lookback, verbose=verbose)
    mv_bundle, uni_bundle = best_mv["bundle"], uni["bundle"]
    split = preprocessing.split_row(len(df))
    _assert_split_and_alignment(mv_bundle, uni_bundle, split)

    weeks = mv_bundle["weeks"]
    actual = mv_bundle["y_test_orig"]
    pred_mv = _predict_php_by_horizon(best_mv["model"], mv_bundle)
    pred_uni = _predict_php_by_horizon(uni["model"], uni_bundle)
    if pred_mv.shape != pred_uni.shape or pred_mv.shape != (66, config.FORECAST_HORIZON):
        raise AssertionError(
            f"Expected paired prediction shape (66, {config.FORECAST_HORIZON}); "
            f"got {pred_mv.shape} and {pred_uni.shape}."
        )

    metrics: dict[str, dict] = {}
    dm_tests: dict[str, dict] = {}
    loss_differentials = np.empty_like(pred_mv)
    for horizon in range(1, config.FORECAST_HORIZON + 1):
        index = horizon - 1
        key = str(horizon)
        metrics[key] = {
            "multivariate": evaluate.regression_metrics(actual, pred_mv[:, index]),
            "univariate": evaluate.regression_metrics(actual, pred_uni[:, index]),
        }
        err_mv = actual - pred_mv[:, index]
        err_uni = actual - pred_uni[:, index]
        loss_differentials[:, index] = err_mv ** 2 - err_uni ** 2
        dm_tests[key] = evaluate.diebold_mariano(
            err_mv, err_uni, h=horizon, power=2
        )

    metrics["overall"] = {
        "multivariate": evaluate.regression_metrics(
            np.tile(actual[:, None], (1, config.FORECAST_HORIZON)).ravel(),
            pred_mv.ravel(),
        ),
        "univariate": evaluate.regression_metrics(
            np.tile(actual[:, None], (1, config.FORECAST_HORIZON)).ravel(),
            pred_uni.ravel(),
        ),
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
                round(float(value), 8) for value in pred_mv[:, horizon - 1]
            ],
            "univariate": [
                round(float(value), 8) for value in pred_uni[:, horizon - 1]
            ],
        }
        for horizon in range(1, config.FORECAST_HORIZON + 1)
    }

    shap_results = {
        "multivariate": explain.explain_model(
            best_mv["model"], mv_bundle, mv_names, "multivariate"
        ),
        "univariate": explain.explain_model(
            uni["model"], uni_bundle, [], "univariate"
        ),
    }

    results = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "lookback_L": lookback,
            "lookback_grid_val_loss": {
                str(candidate): round(value, 6)
                for candidate, value in l_grid.items()
            },
            "n_train": int(len(mv_bundle["X_train"])),
            "n_val": int(len(mv_bundle["X_val"])),
            "n_test": int(len(mv_bundle["rows_test"])),
            "features": mv_names,
            "dataset_rows": int(len(df)),
            "horizons": list(range(1, config.FORECAST_HORIZON + 1)),
            "data_note": (
                "rice_import_mt: 9 trailing weeks forward-filled (5049 MT). "
                "Recent diesel_price values (Mar-Apr 2026) are unusually high; verify source."
            ),
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
    }
    return results


def save(results: dict, path: str = config.RESULTS_JSON) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return path


if __name__ == "__main__":
    save(build_results())
    print("Wrote", config.RESULTS_JSON)

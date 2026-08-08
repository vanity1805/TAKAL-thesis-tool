"""Assemble every model output into a single ``artifacts/results.json`` that the
Streamlit dashboard (``app.py``) loads in place of its mock dictionaries."""

from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

from . import config, data_prep, evaluate, explain, train


def _predict_php(model, bundle) -> np.ndarray:
    pred_norm = model.predict(bundle["X_test"], verbose=0).ravel()
    return pred_norm * bundle["tgt_range"] + bundle["tgt_min"]


def build_results(verbose: int = 0) -> dict:
    """Run the full pipeline and return the assembled results dictionary."""
    df = data_prep.prepare()

    # 1. Light lookback tuning on the multivariate model; reuse chosen L for both.
    mv_cols = [c for c, _ in config.MULTIVARIATE_FEATURES]
    mv_names = [name for _, name in config.MULTIVARIATE_FEATURES]
    best_mv, l_grid = train.select_lookback(df, mv_cols, verbose=verbose)
    lookback = best_mv["lookback"]

    # 2. Univariate control at the same (constant) lookback.
    uni_cols = [config.TARGET_COL]
    uni = train.run_experiment(df, uni_cols, lookback, verbose=verbose)

    mv_bundle, uni_bundle = best_mv["bundle"], uni["bundle"]
    weeks = mv_bundle["weeks"]
    actual = mv_bundle["y_test_orig"]

    # 3. Forecast series + metrics.
    pred_mv = _predict_php(best_mv["model"], mv_bundle)
    pred_uni = _predict_php(uni["model"], uni_bundle)

    metrics = {
        "multivariate": evaluate.regression_metrics(actual, pred_mv),
        "univariate": evaluate.regression_metrics(actual, pred_uni),
    }

    # 4. Diebold-Mariano test (multivariate vs univariate).
    dm = evaluate.diebold_mariano(actual - pred_mv, actual - pred_uni)

    # 5. SHAP explanations.
    shap_out = {
        "multivariate": explain.explain_model(best_mv["model"], mv_bundle, mv_names, "multivariate"),
        "univariate": explain.explain_model(uni["model"], uni_bundle, [], "univariate"),
    }

    results = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "lookback_L": lookback,
            "lookback_grid_val_loss": {str(k): round(v, 6) for k, v in l_grid.items()},
            "n_train": int(len(mv_bundle["X_train"])),
            "n_val": int(len(mv_bundle["X_val"])),
            "n_test": int(len(mv_bundle["X_test"])),
            "features": mv_names,
            "dataset_rows": int(len(df)),
            "data_note": (
                "rice_import_mt: 9 trailing weeks forward-filled (5049 MT). "
                "Recent diesel_price values (Mar-Apr 2026) are unusually high; verify source."
            ),
        },
        "weeks": weeks,
        "forecast": {
            "actual": [round(float(v), 3) for v in actual],
            "multivariate": [round(float(v), 3) for v in pred_mv],
            "univariate": [round(float(v), 3) for v in pred_uni],
        },
        "metrics": metrics,
        "dm_test": dm,
        "shap": shap_out,
    }
    return results


def save(results: dict, path: str = config.RESULTS_JSON) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return path


if __name__ == "__main__":
    save(build_results())
    print("Wrote", config.RESULTS_JSON)

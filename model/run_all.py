# =============================================================================
# Program      : TAKAL one-command pipeline runner (model/run_all.py)
# Description  : Runs the whole offline forecasting pipeline with one command
#                and reports clearly whether it worked.
# Author       : TAKAL Thesis Team (contact: chavezearljustineb@gmail.com)
# System Fit   : Entry point of the offline pipeline (python -m model.run_all).
#                It writes artifacts/results.json for the Streamlit dashboard
#                (app.py) and artifacts/pipeline_status.json, which lets the
#                dashboard show a formatted message if this run crashed.
# Date Written : 2026
# Date Revised : 2026-10-27
# Purpose      : Prepare the data, tune the lookback window, train the
#                multivariate and univariate Stacked LSTMs, evaluate them
#                (RMSE / MAE / MAPE / R-squared + Diebold-Mariano), compute SHAP
#                explanations and the 12-week forecast, then save everything.
# Algorithms   : None of its own; it sequences export_results.build_results().
#                The structured try-except here is the single place that turns
#                every crash into a status file plus a readable console message.
# Data         : Reads the nested results dictionary (keyed by horizon, then by
# Structures     model name) only to print a summary; the status record is a
#                flat dictionary of strings.
# =============================================================================
"""One-command end-to-end runner for the TAKAL forecasting pipeline.

    python -m model.run_all

Prepares the data, lightly tunes the lookback window, trains the multivariate
and univariate Stacked LSTM models, evaluates them (RMSE/MAE/MAPE/R2 +
Diebold-Mariano), computes SHAP explanations, and writes artifacts/results.json.
Exits with status 1 (and writes artifacts/pipeline_status.json) if anything fails.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from . import config, export_results
from .errors import TakalPipelineError


def _report_failure(error: TakalPipelineError) -> int:
    """Print a formatted failure message, save the status file and return exit code 1."""
    print("\n" + "=" * 70)
    print("TAKAL pipeline FAILED")
    print("=" * 70)
    print(f"  Where   : {error.stage}")
    print(f"  Problem : {error.message}")
    if error.hint:
        print(f"  Try     : {error.hint}")
    if error.__cause__ is not None:
        print(f"  Cause   : {type(error.__cause__).__name__}: {error.__cause__}")
    status_path = export_results.write_pipeline_status(succeeded=False, error=error)
    if status_path:
        print(f"  The dashboard will show this message (saved to {status_path}).")
    return 1


def _print_summary(results: dict, started_at: float) -> None:
    """Print the headline numbers of a successful run."""
    meta = results["meta"]
    metrics = results["metrics"]
    dm_tests = results["dm_test"]
    print(f"\nDone in {time.time() - started_at:.1f}s")
    print(f"  Chosen lookback L : {meta['lookback_L']}  (grid val_loss: {meta['lookback_grid_val_loss']})")
    print(f"  Train/Val/Test    : {meta['n_train']}/{meta['n_val']}/{meta['n_test']}")
    for horizon in [*(str(value) for value in range(1, config.FORECAST_HORIZON + 1)), "overall"]:
        label = "Overall" if horizon == "overall" else f"h={horizon}"
        print(f"  {label}:")
        for model in ("multivariate", "univariate"):
            values = metrics[horizon][model]
            print(
                f"    {model:14s} RMSE={values['rmse']:.4f} MAE={values['mae']:.4f} "
                f"MAPE={values['mape']:.4f}% R2={values['r2']:.4f}"
            )
        dm_result = dm_tests[horizon]
        print(
            f"    DM: statistic={dm_result['statistic']:.4f} p={dm_result['p_value']:.6f} "
            f"alpha={dm_result['alpha']:.3f} decision={dm_result['decision']}"
        )
    future = results["future_forecast"]
    first_date, last_date = future["horizon_dates"][0], future["horizon_dates"][-1]
    print(f"  Future forecast   : {first_date} -> {last_date} (origin {future['origin_date']})")
    for model in ("multivariate", "univariate"):
        values = future[f"{model}_predictions"]
        print(f"    {model:14s} week +1 = {values[0]:.2f}  week +12 = {values[-1]:.2f} PHP/kg")
    print(f"  Wrote             : {config.RESULTS_JSON}")


def main(verbose: int = 0) -> int:
    """Run the pipeline; return 0 on success and 1 on any failure."""
    started_at = time.time()
    print("Running TAKAL pipeline (this trains several LSTMs; please wait)...")
    try:
        results = export_results.build_results(verbose=verbose)
        export_results.save(results)
    except TakalPipelineError as error:
        return _report_failure(error)
    except OSError as error:
        # Raised by export_results.save(), e.g. a full disk or a read-only artifacts folder.
        wrapped_error = TakalPipelineError(
            f"The results file could not be written: {error}",
            stage="Saving artifacts/results.json",
            hint="Check free disk space and that the artifacts folder is writable.",
        )
        wrapped_error.__cause__ = error
        return _report_failure(wrapped_error)
    except Exception as error:  # Last-resort guard: the dashboard must always get a status file.
        wrapped_error = TakalPipelineError(
            f"{type(error).__name__}: {error}",
            stage="Unexpected pipeline failure",
            hint="Re-run `python -m model.run_all`. If it repeats, send the traceback in "
                 "artifacts/pipeline_status.json to the developer.",
        )
        wrapped_error.__cause__ = error
        return _report_failure(wrapped_error)

    export_results.write_pipeline_status(succeeded=True)
    _print_summary(results, started_at)
    return 0


if __name__ == "__main__":
    sys.exit(main())

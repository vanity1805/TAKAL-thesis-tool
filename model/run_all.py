"""One-command end-to-end runner for the TAKAL forecasting pipeline.

    python -m model.run_all

Prepares the data, lightly tunes the lookback window, trains the multivariate
and univariate Stacked LSTM models, evaluates them (RMSE/MAE/MAPE/R2 +
Diebold-Mariano), computes SHAP explanations, and writes artifacts/results.json.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from . import config, export_results


def main(verbose: int = 0) -> None:
    t0 = time.time()
    print("Running TAKAL pipeline (this trains several LSTMs; please wait)...")
    results = export_results.build_results(verbose=verbose)
    export_results.save(results)

    meta = results["meta"]
    m = results["metrics"]
    dm = results["dm_test"]
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"  Chosen lookback L : {meta['lookback_L']}  (grid val_loss: {meta['lookback_grid_val_loss']})")
    print(f"  Train/Val/Test    : {meta['n_train']}/{meta['n_val']}/{meta['n_test']}")
    print(f"  Multivariate      : RMSE={m['multivariate']['rmse']}  MAE={m['multivariate']['mae']}"
          f"  MAPE={m['multivariate']['mape']}%  R2={m['multivariate']['r2']}")
    print(f"  Univariate        : RMSE={m['univariate']['rmse']}  MAE={m['univariate']['mae']}"
          f"  MAPE={m['univariate']['mape']}%  R2={m['univariate']['r2']}")
    print(f"  Diebold-Mariano   : stat={dm['statistic']}  p={dm['p_value']}"
          f"  better={dm['better']}  significant={dm['significant']}")
    print(f"  Wrote             : {config.RESULTS_JSON}")


if __name__ == "__main__":
    main()

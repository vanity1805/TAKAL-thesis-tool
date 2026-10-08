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
    for horizon in [*(str(value) for value in range(1, config.FORECAST_HORIZON + 1)), "overall"]:
        label = "Overall" if horizon == "overall" else f"h={horizon}"
        print(f"  {label}:")
        for model in ("multivariate", "univariate"):
            values = m[horizon][model]
            print(
                f"    {model:14s} RMSE={values['rmse']:.4f} MAE={values['mae']:.4f} "
                f"MAPE={values['mape']:.4f}% R2={values['r2']:.4f}"
            )
        result = dm[horizon]
        print(
            f"    DM: statistic={result['statistic']:.4f} p={result['p_value']:.6f} "
            f"alpha={result['alpha']:.3f} decision={result['decision']}"
        )
    print(f"  Wrote             : {config.RESULTS_JSON}")


if __name__ == "__main__":
    main()

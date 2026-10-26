"""Seed-robustness check: is 'multivariate vs univariate' a real gap or LSTM noise?

    python -m model.seed_robustness            # uses the lookback L in artifacts/results.json
    python -m model.seed_robustness 4 5 --seeds 42 7 123 11 99

Trains BOTH models at the same L for several random seeds and reports pooled
test RMSE (all 12 horizons x every test week), mean +- sd, and how often each model wins.
It also prints the no-model "last observed price" (persistence) baseline so the
LSTM errors have a reference point. Use the mean +- sd in the thesis rather than
a single seed. This script is for REPORTING; never choose L from its test numbers.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from . import config, data_prep, preprocessing, train
from .export_results import _predict_php_by_horizon


def persistence_rmse(df) -> float:
    """Pooled RMSE of forecasting 'price h weeks ahead = last observed price'."""
    y = df[config.TARGET_COL].to_numpy(dtype="float64")
    rows = np.arange(preprocessing.split_row(len(y)), len(y))
    errs = [(y[rows] - y[rows - h]) for h in range(1, config.FORECAST_HORIZON + 1)]
    return float(np.sqrt(np.mean(np.square(errs))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lookbacks", nargs="*", type=int)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123, 11, 99])
    args = ap.parse_args()

    lookbacks = args.lookbacks
    if not lookbacks:
        with open(config.RESULTS_JSON) as fh:
            lookbacks = [json.load(fh)["meta"]["lookback_L"]]

    df = data_prep.prepare()
    mv_cols = [c for c, _ in config.MULTIVARIATE_FEATURES]
    uni_cols = [config.TARGET_COL]
    print(f"Persistence baseline pooled RMSE: {persistence_rmse(df):.3f} PHP/kg")

    for lookback in lookbacks:
        rmse = {"multivariate": [], "univariate": []}
        for seed in args.seeds:
            for name, cols in (("multivariate", mv_cols), ("univariate", uni_cols)):
                exp = train.run_experiment(df, cols, lookback, seed=seed)
                pred = _predict_php_by_horizon(exp["model"], exp["bundle"])
                actual = exp["bundle"]["y_test_orig"]
                rmse[name].append(float(np.sqrt(np.mean((actual[:, None] - pred) ** 2))))
        mv, uni = np.array(rmse["multivariate"]), np.array(rmse["univariate"])
        print(f"\nL={lookback}  seeds={args.seeds}")
        print(f"  multivariate RMSE {mv.mean():.3f} +- {mv.std():.3f}  (min {mv.min():.3f}, max {mv.max():.3f})")
        print(f"  univariate   RMSE {uni.mean():.3f} +- {uni.std():.3f}  (min {uni.min():.3f}, max {uni.max():.3f})")
        print(f"  multivariate better in {(mv < uni).sum()}/{len(mv)} seeds")


if __name__ == "__main__":
    main()

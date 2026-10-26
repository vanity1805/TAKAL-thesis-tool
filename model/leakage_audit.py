"""Leakage audit for the TAKAL pipeline.

    python -m model.leakage_audit            # uses L from artifacts/results.json
    python -m model.leakage_audit 4 --seeds 42 7 123

Checks, for both the multivariate and univariate experiments:
  1. Scalers are fit on pre-test rows only.
  2. Every training / validation window (inputs AND targets) lies before the test
     region, and validation targets come strictly after training targets.
  3. Test inputs for (target week t, horizon h) end at week t-h (never later) and
     match the raw data exactly.
  4. Future-invariance: overwriting every row after a cut-off with garbage leaves
     all earlier test forecasts unchanged (while later ones do change).
  5. Forbidden columns (local_price, imported_price) are not model inputs.
  6. Signal check: the multivariate model is retrained with the economic columns
     shuffled in time (price history kept). If accuracy barely moves, the economic
     columns add no usable signal for this model, which is a result, not a leak.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from . import config, data_prep, preprocessing, train
from .export_results import _predict_php_by_horizon

H = config.FORECAST_HORIZON


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def static_checks(df, cols, lookback, label):
    n, s = len(df), preprocessing.split_row(len(df))
    b = preprocessing.build_supervised(df, cols, lookback)
    raw = df[cols].to_numpy("float64")
    assert np.allclose(b["feat_scaler"].data_min_, raw[:s].min(0)), "scaler saw test rows"
    assert np.allclose(b["feat_scaler"].data_max_, raw[:s].max(0)), "scaler saw test rows"
    _ok(f"{label}: scalers fit on rows < {s} only")
    tr, va = b["train_target_rows"], b["val_target_rows"]
    assert tr.max() < s and va.max() < s
    assert va.min() > tr.max(), "validation targets overlap training targets"
    assert tr.min() - lookback >= 0
    _ok(f"{label}: train/val inputs+targets < test start; val targets after train targets "
        f"(train<= {tr.max()}, val {va.min()}..{va.max()}, test starts {s})")
    scaled = b["feat_scaler"].transform(raw)
    for h in (1, 6, 12):
        rows = b["rows_test"]
        origins = rows - h + 1
        assert np.array_equal(b["origins_test_by_horizon"][str(h)], origins)
        for i in (0, len(rows) // 2, len(rows) - 1):
            expected = scaled[origins[i] - lookback: origins[i]]
            assert np.allclose(b["X_test_by_horizon"][str(h)][i], expected)
            assert origins[i] - 1 == rows[i] - h, "input window ends after t-h"
    _ok(f"{label}: test input for (t,h) = rows ending at t-h, equal to raw data")
    return b


def future_invariance(df, cols, lookback, label, seed=config.SEED, keep=30):
    s = preprocessing.split_row(len(df))
    exp = train.run_experiment(df, cols, lookback, seed=seed)
    base = _predict_php_by_horizon(exp["model"], exp["bundle"])
    cut = s + keep
    bad = df.copy()
    for c in set(cols) | {config.TARGET_COL}:
        bad[c] = bad[c].astype("float64")
    rng = np.random.default_rng(0)
    for c in set(cols) | {config.TARGET_COL}:
        bad.loc[bad.index[cut:], c] = df[c].iloc[cut:].to_numpy() * rng.uniform(0.3, 3.0, len(df) - cut)
    b2 = preprocessing.build_supervised(bad, cols, lookback)
    alt = _predict_php_by_horizon(exp["model"], b2)
    same = np.abs(base[:keep] - alt[:keep]).max()
    diff = np.abs(base[keep + H:] - alt[keep + H:]).max()
    assert same < 1e-9, f"{label}: earlier forecasts changed by {same}"
    assert diff > 1e-6, "control failed: later forecasts did not react to altered data"
    _ok(f"{label}: scrambling rows >= {cut} changed earlier forecasts by {same:.1e} "
        f"(later ones by up to {diff:.2f}, as expected)")


def shuffled_exog_check(df, lookback, seeds):
    mv_cols = [c for c, _ in config.MULTIVARIATE_FEATURES]
    exog = [c for c in mv_cols if c != config.TARGET_COL]
    perm = np.random.default_rng(1).permutation(len(df))
    shuffled = df.copy()
    for c in exog:
        shuffled[c] = df[c].to_numpy()[perm]
    out = {"real": [], "shuffled": [], "uni": []}
    for seed in seeds:
        for key, frame, cols in (("real", df, mv_cols), ("shuffled", shuffled, mv_cols),
                                 ("uni", df, [config.TARGET_COL])):
            e = train.run_experiment(frame, cols, lookback, seed=seed)
            p = _predict_php_by_horizon(e["model"], e["bundle"])
            a = e["bundle"]["y_test_orig"]
            out[key].append(float(np.sqrt(np.mean((a[:, None] - p) ** 2))))
    print(f"  Signal check at L={lookback}, seeds={list(seeds)} (pooled test RMSE):")
    for key, label in (("real", "multivariate, real economic columns"),
                       ("shuffled", "multivariate, economic columns SHUFFLED in time"),
                       ("uni", "univariate")):
        v = np.array(out[key]); print(f"    {label:52s} {v.mean():.3f} +- {v.std():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lookback", nargs="?", type=int)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    args = ap.parse_args()
    lookback = args.lookback
    if lookback is None:
        with open(config.RESULTS_JSON) as fh:
            lookback = json.load(fh)["meta"]["lookback_L"]
    df = data_prep.prepare()
    mv_cols = [c for c, _ in config.MULTIVARIATE_FEATURES]
    uni_cols = [config.TARGET_COL]
    print(f"Leakage audit  (rows={len(df)}, L={lookback})")
    banned = {"local_price", "imported_price"}
    assert not banned & set(mv_cols + uni_cols)
    _ok("model inputs exclude local_price / imported_price (the parts of the target)")
    for label, cols in (("multivariate", mv_cols), ("univariate", uni_cols)):
        static_checks(df, cols, lookback, label)
    for label, cols in (("multivariate", mv_cols), ("univariate", uni_cols)):
        future_invariance(df, cols, lookback, label)
    shuffled_exog_check(df, lookback, args.seeds)


if __name__ == "__main__":
    main()

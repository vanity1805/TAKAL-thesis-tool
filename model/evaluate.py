"""Evaluation: regression metrics and the Diebold-Mariano significance test."""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import config


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, MAPE (%), and R^2 on the original PHP/kg scale."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / y_true)) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": round(rmse, 4), "mae": round(mae, 4),
            "mape": round(mape, 4), "r2": round(r2, 4)}


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray, h: int = 1, power: int = 2) -> dict:
    """Diebold-Mariano test (Harvey-Leybourne-Newbold small-sample correction).

    ``err_a`` = multivariate errors, ``err_b`` = univariate errors. A negative
    statistic means model A (multivariate) has the smaller loss.
    """
    err_a = np.asarray(err_a, dtype="float64")
    err_b = np.asarray(err_b, dtype="float64")
    d = np.abs(err_a) ** power - np.abs(err_b) ** power
    return diebold_mariano_differential(d, h)


def diebold_mariano_differential(differential: np.ndarray, h: int) -> dict:
    """Test a supplied loss differential with the h-step HLN correction."""
    d = np.asarray(differential, dtype="float64")
    n = len(d)
    if n < 2 or h < 1 or h > n:
        raise ValueError(f"DM test requires 1 <= h <= n and n >= 2; received h={h}, n={n}.")
    mean_d = float(np.mean(d))

    centered = d - mean_d
    long_run_variance = float(np.mean(centered ** 2))
    for lag in range(1, h):
        autocovariance = float(np.sum(centered[lag:] * centered[:-lag]) / n)
        long_run_variance += 2.0 * autocovariance
    if long_run_variance <= 0:
        if np.allclose(d, d[0]):
            return {
                "statistic": 0.0, "p_value": 1.0, "alpha": config.ALPHA,
                "better": "tie", "significant": False, "decision": "Retain H0", "n": n,
            }
        raise ValueError("DM long-run variance is non-positive; the test statistic is undefined.")

    dm = mean_d / np.sqrt(long_run_variance / n)
    harvey = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * harvey
    p_value = float(2 * stats.t.cdf(-abs(dm_hln), df=n - 1))
    significant = bool(p_value < config.ALPHA)
    if np.isclose(mean_d, 0.0):
        better = "tie"
    else:
        better = "multivariate" if mean_d < 0 else "univariate"

    return {
        "statistic": round(float(dm_hln), 4),
        "p_value": round(p_value, 6),
        "alpha": config.ALPHA,
        "better": better,
        "significant": significant,
        "decision": "Reject H0" if significant else "Retain H0",
        "n": n,
    }

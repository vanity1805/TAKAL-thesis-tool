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
    n = len(d)
    mean_d = float(np.mean(d))

    # For a one-step horizon only the lag-0 autocovariance is needed.
    gamma0 = float(np.mean((d - mean_d) ** 2))
    if gamma0 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "alpha": config.ALPHA,
                "better": "tie", "n": n}

    dm = mean_d / np.sqrt(gamma0 / n)
    harvey = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * harvey
    p_value = float(2 * stats.t.cdf(-abs(dm_hln), df=n - 1))

    better = "multivariate" if mean_d < 0 else "univariate"
    return {
        "statistic": round(float(dm_hln), 4),
        "p_value": round(p_value, 6),
        "alpha": config.ALPHA,
        "better": better,
        "significant": bool(p_value < config.ALPHA),
        "n": n,
    }

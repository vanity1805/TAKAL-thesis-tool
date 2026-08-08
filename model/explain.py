"""SHAP explainability for the Stacked LSTM models.

Uses ``shap.GradientExplainer`` (expected gradients) on the 3-D LSTM input.
For the multivariate model, per-timestep attributions are aggregated across the
lookback window into one value per feature ("Vector SHAP", per the thesis). For
the univariate model, each lookback timestep is kept as its own lag player
(Price(t-1) ... Price(t-L)).

All SHAP values are converted from normalized model-output units to PHP/kg by
multiplying by the target range, so the dashboard can present them in ₱/kg.
"""

from __future__ import annotations

import warnings

import numpy as np
import shap

warnings.filterwarnings("ignore")


def _shap_values_3d(model, background: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return SHAP values shaped (n_samples, lookback, n_features)."""
    explainer = shap.GradientExplainer(model, background)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 4 and sv.shape[-1] == 1:
        sv = sv[..., 0]
    return sv


def _normalize01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float64")
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


def _complete_to_sum(contrib: np.ndarray, target_sum: float) -> np.ndarray:
    """Adjust contributions so they sum exactly to ``target_sum`` (SHAP efficiency),
    distributing the small approximation residual proportionally to |contribution|."""
    contrib = np.asarray(contrib, dtype="float64")
    residual = target_sum - contrib.sum()
    weights = np.abs(contrib)
    total = weights.sum()
    if total < 1e-12:
        return contrib + residual / len(contrib)
    return contrib + residual * weights / total


def explain_model(model, bundle: dict, display_names: list[str], mode: str,
                  background_size: int = 100) -> dict:
    """Compute global + local SHAP outputs for one trained model.

    ``mode`` is "multivariate" (aggregate across timesteps) or "univariate"
    (per-lag). Returns a dashboard-ready dictionary.
    """
    rng = np.random.RandomState(42)
    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    lookback = bundle["lookback"]
    tgt_range = bundle["tgt_range"]
    tgt_min = bundle["tgt_min"]
    tgt_scaler = bundle["tgt_scaler"]

    k = min(background_size, len(X_train))
    background = X_train[rng.choice(len(X_train), k, replace=False)]

    sv = _shap_values_3d(model, background, X_test)          # (n, L, F)

    base_norm = float(np.mean(model.predict(background, verbose=0)))
    base_php = base_norm * tgt_range + tgt_min
    pred_norm = model.predict(X_test, verbose=0).ravel()
    pred_php = pred_norm * tgt_range + tgt_min

    if mode == "multivariate":
        # Vector SHAP: sum attributions over the lookback window -> per feature.
        contrib = sv.sum(axis=1) * tgt_range                 # (n, F) in PHP
        feature_names = list(display_names)
        feat_values = bundle["feat_last_orig"]               # (n, F) original scale
    else:
        # Univariate: one player per lookback lag. Timestep i -> Price(t-(L-i)).
        contrib = sv[:, :, 0] * tgt_range                    # (n, L) in PHP
        feature_names = [f"Price(t-{lookback - i})" for i in range(lookback)]
        feat_values = tgt_scaler.inverse_transform(
            X_test[:, :, 0].reshape(-1, 1)
        ).reshape(X_test.shape[0], lookback)

    n_samples, n_feat = contrib.shape

    mean_abs = np.abs(contrib).mean(axis=0)
    order = list(np.argsort(mean_abs)[::-1])                 # descending importance
    ordered_names = [feature_names[i] for i in order]

    bar = [round(float(mean_abs[i]), 5) for i in order]

    beeswarm: dict[str, dict] = {}
    dependence: dict[str, dict] = {}
    for i in order:
        name = feature_names[i]
        beeswarm[name] = {
            "shap": [round(float(v), 5) for v in contrib[:, i]],
            "feature_value": [round(float(v), 5) for v in _normalize01(feat_values[:, i])],
        }
        dependence[name] = {
            "x": [round(float(v), 5) for v in feat_values[:, i]],
            "y": [round(float(v), 5) for v in contrib[:, i]],
        }

    local = []
    for s in range(n_samples):
        adj = _complete_to_sum(contrib[s], float(pred_php[s] - base_php))
        contributions = {feature_names[i]: round(float(adj[i]), 5) for i in order}
        local.append({
            "contributions": contributions,
            "predicted": round(float(pred_php[s]), 4),
        })

    return {
        "features": ordered_names,
        "bar": bar,
        "beeswarm": beeswarm,
        "dependence": dependence,
        "base_value": round(float(base_php), 4),
        "local": local,
    }

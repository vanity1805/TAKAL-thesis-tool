"""Horizon-specific SHAP explanations for the Stacked LSTM models."""

from __future__ import annotations

import warnings

import numpy as np
import shap
import tensorflow as tf

from . import config

warnings.filterwarnings("ignore")


def _shap_values_3d(model, background: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return SHAP values shaped (n_samples, lookback, n_features)."""
    explainer = shap.GradientExplainer(model, background)
    values = explainer.shap_values(X, rseed=config.SEED)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    return values


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    lower, upper = float(values.min()), float(values.max())
    if upper - lower < 1e-12:
        return np.full_like(values, 0.5)
    return (values - lower) / (upper - lower)


def _complete_to_sum(contributions: np.ndarray, target_sum: float) -> np.ndarray:
    """Distribute the approximation residual so baseline + SHAP equals prediction."""
    contributions = np.asarray(contributions, dtype="float64")
    residual = target_sum - contributions.sum()
    weights = np.abs(contributions)
    total = weights.sum()
    if total < 1e-12:
        return contributions + residual / len(contributions)
    return contributions + residual * weights / total


def explain_model(
    model,
    bundle: dict,
    display_names: list[str],
    mode: str,
    background_size: int = 100,
) -> dict:
    """Compute global and local explanations for each forecast horizon."""
    rng = np.random.default_rng(config.SEED)
    X_train = bundle["X_train"]
    lookback = bundle["lookback"]
    tgt_range = bundle["tgt_range"]
    tgt_min = bundle["tgt_min"]
    tgt_scaler = bundle["tgt_scaler"]

    background_count = min(background_size, len(X_train))
    background = X_train[
        rng.choice(len(X_train), background_count, replace=False)
    ]
    all_horizons: dict[str, dict] = {}

    for horizon in range(1, config.FORECAST_HORIZON + 1):
        key = str(horizon)
        X_test = bundle["X_test_by_horizon"][key]
        # GradientExplainer is more reliable when it sees a single scalar output.
        horizon_input = tf.keras.Input(shape=(lookback, X_train.shape[-1]))
        horizon_output = tf.keras.layers.Lambda(
            lambda values, index=horizon - 1: values[:, index:index + 1],
            name=f"horizon_{horizon}_output",
        )(model(horizon_input))
        horizon_model = tf.keras.Model(
            inputs=horizon_input,
            outputs=horizon_output,
        )
        shap_values = _shap_values_3d(horizon_model, background, X_test)

        base_norm = float(np.mean(horizon_model.predict(background, verbose=0)))
        base_php = base_norm * tgt_range + tgt_min
        base_php_display = round(float(base_php), 4)
        pred_norm = np.asarray(horizon_model.predict(X_test, verbose=0)).reshape(-1)
        pred_php = pred_norm * tgt_range + tgt_min

        # Sum over the lookback window to obtain the thesis's Vector SHAP values.
        vector_contributions = shap_values.sum(axis=1) * tgt_range
        if mode == "multivariate":
            detailed_contributions = vector_contributions
            detail_names = list(display_names)
            detail_values = bundle["feat_last_orig_by_horizon"][key]
            global_contributions = vector_contributions
            global_names = list(display_names)
        else:
            detailed_contributions = shap_values[:, :, 0] * tgt_range
            detail_names = [
                f"Price(t-{lookback - i})" for i in range(lookback)
            ]
            detail_values = tgt_scaler.inverse_transform(
                X_test[:, :, 0].reshape(-1, 1)
            ).reshape(X_test.shape[0], lookback)
            # Aggregate all lag attributions first, then average their absolute value.
            global_contributions = vector_contributions.sum(axis=1, keepdims=True)
            global_names = ["Retail Rice Price Lag"]

        detail_importance = np.abs(detailed_contributions).mean(axis=0)
        detail_order = list(np.argsort(detail_importance)[::-1])
        global_importance = np.abs(global_contributions).mean(axis=0)
        global_order = list(np.argsort(global_importance)[::-1])

        beeswarm: dict[str, dict] = {}
        dependence: dict[str, dict] = {}
        for index in detail_order:
            name = detail_names[index]
            beeswarm[name] = {
                "shap": [
                    round(float(value), 5)
                    for value in detailed_contributions[:, index]
                ],
                "feature_value": [
                    round(float(value), 5)
                    for value in _normalize01(detail_values[:, index])
                ],
            }
            dependence[name] = {
                "x": [round(float(value), 5) for value in detail_values[:, index]],
                "y": [
                    round(float(value), 5)
                    for value in detailed_contributions[:, index]
                ],
            }

        local = []
        for sample_index in range(len(X_test)):
            adjusted = _complete_to_sum(
                detailed_contributions[sample_index],
                float(pred_php[sample_index] - base_php),
            )
            predicted_display = round(float(pred_php[sample_index]), 4)
            rounded_contributions = np.round(adjusted, 5)
            residual = predicted_display - base_php_display - float(rounded_contributions.sum())
            rounded_contributions[detail_order[-1]] = round(
                float(rounded_contributions[detail_order[-1]] + residual), 5
            )
            local.append({
                "contributions": {
                    detail_names[index]: round(float(rounded_contributions[index]), 5)
                    for index in detail_order
                },
                "predicted": predicted_display,
            })

        all_horizons[key] = {
            "features": [global_names[index] for index in global_order],
            "bar": [
                round(float(global_importance[index]), 5)
                for index in global_order
            ],
            "detail_features": [detail_names[index] for index in detail_order],
            "beeswarm": beeswarm,
            "dependence": dependence,
            "base_value": base_php_display,
            "local": local,
        }

    return all_horizons

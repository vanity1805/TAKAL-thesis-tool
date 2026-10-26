"""Horizon-specific SHAP explanations for the Stacked LSTM models."""

from __future__ import annotations

import warnings

import numpy as np
import shap
import tensorflow as tf

from . import config

warnings.filterwarnings("ignore")


def _shap_values_3d(
    model, background: np.ndarray, X: np.ndarray, nsamples: int | None = None
) -> np.ndarray:
    """Return SHAP values shaped (n_samples, lookback, n_features).

    ``nsamples=None`` keeps SHAP's default number of gradient samples.
    """
    explainer = shap.GradientExplainer(model, background)
    options = {} if nsamples is None else {"nsamples": int(nsamples)}
    values = explainer.shap_values(X, rseed=config.SEED, **options)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    return values


def _sample_background(X_train: np.ndarray, size: int = 100) -> np.ndarray:
    """Draw the SHAP background windows from the training set (seeded, reproducible).

    Shared by the test-set explanations and the future-window explanation so both
    are measured against the identical reference distribution.
    """
    rng = np.random.default_rng(config.SEED)
    count = min(size, len(X_train))
    return X_train[rng.choice(len(X_train), count, replace=False)]


def _horizon_model(model, horizon: int, lookback: int, n_features: int):
    """Wrap the Dense(12) model so it exposes one horizon as a scalar output.

    GradientExplainer is more reliable when it sees a single scalar output.
    """
    horizon_input = tf.keras.Input(shape=(lookback, n_features))
    horizon_output = tf.keras.layers.Lambda(
        lambda values, index=horizon - 1: values[:, index:index + 1],
        name=f"horizon_{horizon}_output",
    )(model(horizon_input))
    return tf.keras.Model(inputs=horizon_input, outputs=horizon_output)


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
    X_train = bundle["X_train"]
    lookback = bundle["lookback"]
    tgt_range = bundle["tgt_range"]
    tgt_min = bundle["tgt_min"]
    tgt_scaler = bundle["tgt_scaler"]

    background = _sample_background(X_train, background_size)
    all_horizons: dict[str, dict] = {}

    for horizon in range(1, config.FORECAST_HORIZON + 1):
        key = str(horizon)
        X_test = bundle["X_test_by_horizon"][key]
        horizon_model = _horizon_model(model, horizon, lookback, X_train.shape[-1])
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


def explain_future(
    model,
    bundle: dict,
    X_future: np.ndarray,
    pred_php: np.ndarray,
    feature_names: list[str],
    background_size: int = 100,
) -> dict:
    """SHAP attributions for the out-of-sample 12-week forward forecast.

    ``X_future`` is the single scaled input window ``(1, lookback, n_features)``
    that ends at the last observed week; ``pred_php`` holds the model's 12 forward
    predictions in PHP/kg. For each future horizon the same scalar-output
    ``GradientExplainer`` setup (and the same seeded training background) as
    :func:`explain_model` is used, and the attributions are summed over the
    lookback window into one value per feature ("Vector SHAP").

    The 12 outputs of the Dense(12) head have slightly different average
    predictions over the background, but the exported artifact has one scalar
    ``base_value``. It is the mean of the 12 per-horizon baselines, and
    :func:`_complete_to_sum` then spreads each horizon's remaining gap (SHAP
    approximation error plus that horizon's baseline offset) over the features so
    that ``base_value + sum(contributions) == predicted price`` for every week.

    Returns ``{"base_value", "contributions", "diagnostics"}`` where
    ``contributions`` is a list of 12 ``{feature: PHP/kg}`` dicts (horizon 1..12,
    features ordered by mean absolute contribution) that sum exactly, at the
    displayed rounding, to ``round(pred, 4) - base_value``.
    """
    horizon_count = config.FORECAST_HORIZON
    lookback = bundle["lookback"]
    X_future = np.asarray(X_future, dtype="float64")
    pred_php = np.asarray(pred_php, dtype="float64").reshape(-1)
    if X_future.shape != (1, lookback, len(feature_names)):
        raise ValueError(
            f"Expected one future window shaped (1, {lookback}, {len(feature_names)}); "
            f"got {X_future.shape}."
        )
    if len(pred_php) != horizon_count:
        raise ValueError(f"Expected {horizon_count} future predictions; got {len(pred_php)}.")

    tgt_range = bundle["tgt_range"]
    tgt_min = bundle["tgt_min"]
    background = _sample_background(bundle["X_train"], background_size)

    raw = np.empty((horizon_count, len(feature_names)))
    horizon_base = np.empty(horizon_count)
    for horizon in range(1, horizon_count + 1):
        horizon_model = _horizon_model(model, horizon, lookback, len(feature_names))
        shap_values = _shap_values_3d(
            horizon_model, background, X_future, nsamples=config.FUTURE_SHAP_SAMPLES
        )
        if shap_values.shape != (1, lookback, len(feature_names)):
            raise ValueError(
                f"Unexpected SHAP shape {shap_values.shape} for future horizon {horizon}."
            )
        # Sum over the lookback window (thesis "Vector SHAP"), then rescale to PHP/kg.
        raw[horizon - 1] = shap_values.sum(axis=1)[0] * tgt_range
        base_norm = float(np.mean(horizon_model.predict(background, verbose=0)))
        horizon_base[horizon - 1] = base_norm * tgt_range + tgt_min

    base_value = float(horizon_base.mean())
    base_display = round(base_value, 4)

    adjusted = np.vstack([
        _complete_to_sum(raw[index], float(pred_php[index] - base_value))
        for index in range(horizon_count)
    ])
    order = list(np.argsort(np.abs(adjusted).mean(axis=0))[::-1])

    contributions: list[dict[str, float]] = []
    for index in range(horizon_count):
        rounded = np.round(adjusted[index], 5)
        residual = round(float(pred_php[index]), 4) - base_display - float(rounded.sum())
        rounded[order[-1]] = round(float(rounded[order[-1]] + residual), 5)
        contributions.append({feature_names[i]: round(float(rounded[i]), 5) for i in order})

    return {
        "base_value": base_display,
        "contributions": contributions,
        "diagnostics": {
            # Pure SHAP approximation error: (prediction - own baseline) - sum(SHAP).
            "approximation_error": [
                float(pred_php[i] - horizon_base[i] - raw[i].sum()) for i in range(horizon_count)
            ],
            # How far each horizon's own baseline sits from the shared base_value.
            "baseline_shift": [float(horizon_base[i] - base_value) for i in range(horizon_count)],
        },
    }

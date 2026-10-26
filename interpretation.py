"""Plain-language readings of TAKAL's tables.

Every function returns a list of short sentences (plain text, no markup) computed
from the numbers passed in, so the wording changes whenever the data does. The same
sentences are shown under the dashboard tables and written into the PDF / Excel
exports, which keeps the two consistent.

This module deliberately has no Streamlit dependency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

MODEL_NAMES = {
    "multivariate": "multivariate model",
    "univariate": "univariate model",
}

CLOSE_MISS_PHP = 1.0  # a forecast within this many PHP/kg is called "close"


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def peso(value: float | None, signed: bool = False, decimals: int = 2) -> str:
    """₱1.23, or +₱1.23 / −₱1.23 when ``signed`` (true minus sign). Returns '-' when missing."""
    try:
        if value is None or not np.isfinite(float(value)):
            return "-"
        val = float(value)
    except (TypeError, ValueError):
        return "-"
    text = f"₱{abs(val):,.{decimals}f}"
    if not signed:
        return f"-{text}" if val < 0 else text
    if round(val, decimals) == 0:
        return text
    return f"{'+' if val > 0 else '−'}{text}"


def long_date(iso_date: str) -> str:
    try:
        return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%B %d, %Y").replace(" 0", " ")
    except (TypeError, ValueError):
        return str(iso_date)


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


# --------------------------------------------------------------------------- #
# Overview: forecast vs actual, with the Difference column
# --------------------------------------------------------------------------- #
def overview_reading(
    weeks: Sequence[str],
    forecast: Sequence[float],
    actual: Sequence[float | None],
    model: str,
    horizon: int,
) -> list[str]:
    """Explain the Difference (Forecasted − Actual) column for the weeks shown."""
    week_list = list(weeks)
    f = np.asarray([float(v) if v is not None else np.nan for v in forecast], dtype=float)
    a = np.asarray([float(v) if v is not None else np.nan for v in actual], dtype=float)
    keep = np.isfinite(f) & np.isfinite(a)
    lines = [
        "Difference = Forecasted − Actual. A positive difference means the model "
        "predicted a higher price than what actually happened; a negative difference "
        "means it predicted a lower price."
    ]
    if not keep.any():
        return lines + ["There are no weeks with both a forecast and an actual price to compare."]

    diff = (f - a)[keep]
    kept_weeks = [w for w, k in zip(week_list, keep) if k]
    n = len(diff)
    mean_abs = float(np.mean(np.abs(diff)))
    mean_signed = float(np.mean(diff))
    over = int(np.sum(diff > 0))
    under = int(np.sum(diff < 0))
    close = int(np.sum(np.abs(diff) <= CLOSE_MISS_PHP))

    lines.append(
        f"Across the {plural(n, 'week')} shown, the {horizon}-week-ahead forecast of the "
        f"{MODEL_NAMES.get(model, model)} was off by {peso(mean_abs)}/kg on average, "
        f"and {close} of {n} weeks ({close / n:.0%}) landed within {peso(CLOSE_MISS_PHP)}/kg of the actual price."
    )
    if abs(mean_signed) < 0.1 * max(mean_abs, 1e-9) or abs(mean_signed) < 0.05:
        lean = "The forecasts did not lean consistently high or low"
    else:
        lean = (
            f"On balance the forecasts ran {'higher' if mean_signed > 0 else 'lower'} than the actual price, "
            f"by {peso(abs(mean_signed))}/kg on average"
        )
    lines.append(f"{lean} ({plural(over, 'week')} above actual, {plural(under, 'week')} below).")

    worst = int(np.argmax(np.abs(diff)))
    lines.append(
        f"The biggest miss was in the week of {long_date(kept_weeks[worst])}: forecast "
        f"{peso(float(f[keep][worst]))} versus actual {peso(float(a[keep][worst]))} "
        f"(difference {peso(float(diff[worst]), signed=True)}/kg)."
    )
    return lines


# --------------------------------------------------------------------------- #
# Benchmarking: accuracy table
# --------------------------------------------------------------------------- #
def _r2_reading(r2: float) -> str:
    if not np.isfinite(r2):
        return "R² is not available."
    if r2 < 0:
        return (
            f"R² is {r2:.2f}. A negative R² means that, over the test weeks, the forecasts did worse "
            "than simply guessing the average price of those weeks every time."
        )
    if r2 < 0.5:
        return (
            f"R² is {r2:.2f}, so the forecasts capture only a small to moderate share of the "
            "week-to-week movement in price (1.00 would be a perfect match)."
        )
    if r2 < 0.8:
        return f"R² is {r2:.2f}, so the forecasts capture a good share of the week-to-week movement in price."
    return f"R² is {r2:.2f}, so the forecasts capture most of the week-to-week movement in price."


def metrics_reading(metrics: dict, model: str) -> list[str]:
    """Explain the accuracy-by-horizon table for ``model``."""
    other = "univariate" if model == "multivariate" else "multivariate"
    overall = metrics["overall"][model]
    mae, rmse, mape, r2 = (float(overall[k]) for k in ("mae", "rmse", "mape", "r2"))
    lines = [
        "How to read this table: each row is how far forecasts were from the actual price on the same "
        "held-out weeks, and every error column is better when it is lower. MAE is the average miss in "
        "pesos per kg, MAPE is that miss as a percentage of the price, RMSE is like MAE but punishes big "
        "misses more, and R² says how much of the price movement the forecasts follow (higher is better).",
        f"Overall, the {MODEL_NAMES[model]} missed the actual price by {peso(mae)}/kg on average "
        f"(MAE), which is about {mape:.1f}% of the price (MAPE).",
    ]
    if np.isfinite(rmse) and np.isfinite(mae) and rmse > 1.25 * mae:
        lines.append(
            f"RMSE ({peso(rmse)}) is clearly larger than MAE, which tells us that a few weeks were missed "
            "by much more than the typical week."
        )
    lines.append(_r2_reading(r2))

    horizons = [str(h) for h in range(1, 13) if str(h) in metrics]
    if horizons:
        maes = {h: float(metrics[h][model]["mae"]) for h in horizons}
        best = min(maes, key=maes.get)
        worst = max(maes, key=maes.get)
        trend = (
            "Errors generally grow the further ahead the forecast looks."
            if maes[horizons[-1]] > maes[horizons[0]] * 1.1
            else "Errors do not grow clearly with the forecast distance."
        )
        lines.append(
            f"The smallest average miss is at Week {best} ({peso(maes[best])}/kg) and the largest is at "
            f"Week {worst} ({peso(maes[worst])}/kg). {trend}"
        )
    other_mae = float(metrics["overall"][other]["mae"])
    if np.isfinite(other_mae):
        lines.append(
            f"For comparison, the {MODEL_NAMES[other]} has an overall MAE of {peso(other_mae)}/kg, "
            f"versus {peso(mae)}/kg for the {MODEL_NAMES[model]}."
        )
    return lines


# --------------------------------------------------------------------------- #
# Benchmarking: Diebold–Mariano table
# --------------------------------------------------------------------------- #
def dm_reading(dm_test: dict) -> list[str]:
    """Explain the Diebold–Mariano table (it compares the two models, not one)."""
    overall = dm_test["overall"]
    p_value = float(overall["p_value"])
    alpha = float(overall["alpha"])
    significant = bool(overall.get("significant", p_value < alpha))
    better = overall.get("better")

    lines = [
        "How to read this table: the Diebold–Mariano (DM) test checks whether the two models' forecast "
        "errors are really different, or whether the gap could just be luck. The p-value is the chance of "
        f"seeing a gap this large if the models were equally good. A p-value below {alpha:g} means the gap "
        "is unlikely to be luck."
    ]
    if significant:
        lines.append(
            f"Overall, p = {p_value:.4f} is below {alpha:g}: the difference is statistically significant, "
            f"and the {MODEL_NAMES.get(better, 'better')} model's errors are reliably smaller."
        )
    else:
        lines.append(
            f"Overall, p = {p_value:.4f} is above {alpha:g}, so the test could not separate the two models. "
            "The null hypothesis (H₀: equal accuracy) is retained. This means there is no evidence of a "
            "difference. It does not prove that the models are identical, because with a limited number of "
            "test weeks the test can miss small differences."
        )
    sig_horizons = []
    for h in range(1, 13):
        row = dm_test.get(str(h))
        if not row:
            continue
        if bool(row.get("significant", float(row["p_value"]) < float(row["alpha"]))):
            sig_horizons.append((h, row.get("better")))
    if sig_horizons:
        parts = ", ".join(f"Week {h} (favours the {MODEL_NAMES.get(b, b)})" for h, b in sig_horizons)
        lines.append(f"Looking horizon by horizon, a significant difference appears only at: {parts}.")
    else:
        lines.append("Looking horizon by horizon, no forecast distance shows a significant difference.")
    return lines


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def importance_rows(features: Sequence[str], bar: Sequence[float]) -> list[dict]:
    """Rank indicators by average absolute SHAP value, with their share of the total."""
    values = np.asarray(bar, dtype=float)
    total = float(values.sum()) or 1.0
    order = np.argsort(-values)
    return [
        {"name": features[i], "impact": float(values[i]), "share": float(values[i] / total)}
        for i in order
    ]


def direction_text(feature_values: Sequence[float], shap_values: Sequence[float]) -> str:
    """Typical direction of an indicator's effect from the beeswarm points."""
    x = np.asarray(feature_values, dtype=float)
    y = np.asarray(shap_values, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return "No clear pattern"
    rho = float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])
    if rho >= 0.7:
        return "Higher values strongly raise the forecast"
    if rho >= 0.4:
        return "Higher values tend to raise the forecast"
    if rho <= -0.7:
        return "Higher values strongly lower the forecast"
    if rho <= -0.4:
        return "Higher values tend to lower the forecast"
    return "No consistent direction"


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def importance_reading(rows: list[dict], model: str, horizon: int) -> list[str]:
    """Explain the indicator-importance table."""
    if not rows:
        return ["No indicator importance values are available."]
    top = rows[0]
    lines = [
        "How to read this table: SHAP importance is the average amount, in pesos per kg, that an "
        f"indicator pushed the {horizon}-week-ahead forecast away from the model's baseline price during "
        "the test weeks. The share column shows each indicator's portion of all that pushing. "
        "It describes how the model uses its inputs; it does not prove that an indicator causes price changes.",
        f"{top['name']} mattered most, shifting forecasts by {peso(top['impact'])}/kg on average "
        f"({top['share']:.0%} of the total).",
    ]
    if len(rows) >= 3:
        top_two = rows[0]["share"] + rows[1]["share"]
        lines.append(
            f"The top two indicators together account for {top_two:.0%} of the total influence, "
            f"while {rows[-1]['name']} mattered least ({rows[-1]['share']:.0%})."
        )
    return lines


def local_reading(contributions: Sequence[tuple[str, float]], baseline: float, predicted: float) -> list[str]:
    """Explain a single-week waterfall table."""
    ups = [(n, c) for n, c in contributions if c > 0]
    downs = [(n, c) for n, c in contributions if c < 0]
    lines = [
        "How to read this table: the model starts from its baseline price, then each indicator adds or "
        "subtracts a little. Adding all the steps gives that week's forecast. Rows are ordered from the "
        "largest effect to the smallest, and the order is for presentation only (it is not a timeline)."
    ]
    if contributions:
        name, value = max(contributions, key=lambda item: abs(item[1]))
        lines.append(
            f"The biggest single effect came from {name}, which {'raised' if value >= 0 else 'lowered'} "
            f"the estimate by {peso(abs(value))}/kg."
        )
    lines.append(
        f"{plural(len(ups), 'indicator')} raised the estimate (by {peso(sum(c for _, c in ups))}/kg in total) "
        f"and {plural(len(downs), 'indicator')} lowered it (by {peso(abs(sum(c for _, c in downs)))}/kg in total), "
        f"moving the price from the baseline of {peso(baseline)}/kg to the forecast of {peso(predicted)}/kg."
    )
    return lines


# --------------------------------------------------------------------------- #
# 12-week forward forecast (weeks after the last observed week)
# --------------------------------------------------------------------------- #
def _movement(change: float) -> str:
    """'rise by' / 'fall by' / 'stay about the same' for a PHP/kg change."""
    if round(change, 2) == 0:
        return "stay about the same"
    return "rise by" if change > 0 else "fall by"


def future_reading(
    origin_date: str,
    last_observed: float,
    horizon_dates: Sequence[str],
    predictions: dict[str, Sequence[float]],
    active_model: str,
    contributions: Sequence[dict[str, float]] | None = None,
    base_value: float | None = None,
    has_range_warning: bool = False,
) -> list[str]:
    """Explain the 12-week forward forecast in plain language."""
    active = np.asarray(predictions[active_model], dtype=float)
    weeks_ahead = len(active)
    change = float(active[-1] - last_observed)
    pct = change / last_observed * 100 if last_observed else float("nan")
    verb = _movement(change)
    amount = "" if verb == "stay about the same" else f" {peso(abs(change))}/kg"

    top_label = None
    if contributions:
        labels = list(contributions[0])
        matrix = np.array([[float(week[label]) for label in labels] for week in contributions])
        order = np.argsort(np.abs(matrix).mean(axis=0))[::-1]
        top_label = labels[int(order[0])]

    headline = (
        f"Under the {MODEL_NAMES[active_model]}, prices are projected to {verb}{amount} over the next "
        f"{weeks_ahead} weeks, from {peso(last_observed)}/kg (observed {long_date(origin_date)}) to "
        f"{peso(float(active[-1]))}/kg (week of {long_date(horizon_dates[-1])})"
    )
    if np.isfinite(pct):
        headline += f", {'+' if pct >= 0 else '−'}{abs(pct):.1f}%"
    if top_label and active_model == "multivariate":
        headline += f", driven primarily by {top_label}."
    else:
        headline += "."

    lines = [
        "The solid line is the observed price. The dashed lines are what each model projects for the "
        f"next {weeks_ahead} weeks, starting from the last observed week. These are model outputs, not "
        "guarantees, and weeks further out are less certain. Indicators are described by how the model "
        "uses them, which is not evidence that they cause price changes.",
        headline,
    ]

    if contributions and top_label is not None and base_value is not None:
        mean_signed = matrix.mean(axis=0)
        parts = [
            f"{labels[int(i)]} ({peso(float(mean_signed[int(i)]), signed=True)}/kg)" for i in order[:3]
        ]
        lead = (
            f"In the multivariate model, the largest average pulls on the forecast, measured against "
            f"its baseline of {peso(float(base_value))}/kg, are {parts[0]}"
        )
        if len(parts) > 1:
            lead += ", followed by " + " and ".join(parts[1:])
        lines.append(lead + ".")

        delta = matrix[-1] - matrix[0]
        path_change = float(delta.sum())
        key = int(np.argmax(np.abs(delta)))
        if abs(delta[key]) >= 0.005:
            lines.append(
                f"Within the multivariate path itself, the forecast moves {peso(path_change, signed=True)}/kg "
                f"from Week +1 to Week +{weeks_ahead}; the largest single shift comes from {labels[key]}, "
                f"whose contribution goes from {peso(float(matrix[0][key]), signed=True)} to "
                f"{peso(float(matrix[-1][key]), signed=True)}/kg."
            )

    other = "univariate" if active_model == "multivariate" else "multivariate"
    other_end = float(predictions[other][-1])
    other_change = other_end - last_observed
    other_verb = _movement(other_change)
    other_amount = "" if other_verb == "stay about the same" else f" {peso(abs(other_change))}/kg"
    gap = float(active[-1]) - other_end
    relation = (
        "matching" if round(gap, 2) == 0
        else f"{peso(abs(gap))}/kg {'above' if gap > 0 else 'below'}"
    )
    lines.append(
        f"For comparison, the {MODEL_NAMES[other]} projects prices to {other_verb}{other_amount} over the "
        f"same period, ending {relation} the {MODEL_NAMES[active_model]} at Week +{weeks_ahead}."
    )

    if has_range_warning:
        lines.append(
            "Caution: some recent inputs lie outside the range the models saw in training, so the "
            "multivariate projection is an extrapolation and is best read as indicative."
        )
    return lines
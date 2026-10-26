"""Shared configuration and cached data helpers for the TAKAL dashboard."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(APP_DIR, "artifacts", "results.json")
LOGO_PATH = os.path.join(APP_DIR, "Logo_New.png")
BRAND_LOGO_PATH = os.path.join(APP_DIR, "New_Logo_FF.png")

COLORS = {
    # Dashboard Surfaces & Text (Vizora Theme)
    "white": "#FFFFFF",
    "off_white": "#F4F6F5",
    "bg": "#F4F6F5",
    "border": "#E5EBE8",
    "text": "#0F1F1C",
    "muted": "#758A83",

    # Sidebar
    "sidebar_bg": "#0D2B24",
    "sidebar_active": "#163E34",
    "sidebar_border": "#1B473C",
    "sidebar_text": "#CFDFDA",

    # Core Data & Forecast Series
    "actual": "#10B981",          # Emerald Green (Observed prices)
    "multivariate": "#0EA5E9",    # Sky Blue (Multivariate model)
    "univariate": "#64748B",      # Slate Grey (Univariate baseline)

    # SHAP Colors (High contrast: Red/Magenta to Blue or original gradient)
    "shap_low": "#0EA5E9",        # Low feature value: Blue
    "shap_high": "#E11D48",       # High feature value: Vibrant Red/Rose

    # Badges
    "badge_bg": "#E9F8F3",
    "badge_text": "#0B8561",
}


@st.cache_data(show_spinner="Loading TAKAL forecast data...")
def load_results(path: str, modified_at: float) -> dict:
    """Load exported model results; the modification time invalidates cache."""
    with open(path, encoding="utf-8") as results_file:
        return json.load(results_file)


def _same_actual_series(first: list, second: list) -> bool:
    if len(first) != len(second):
        return False
    for a, b in zip(first, second):
        a_null = a is None or (isinstance(a, float) and np.isnan(a))
        b_null = b is None or (isinstance(b, float) and np.isnan(b))
        if a_null and b_null:
            continue
        if a_null != b_null or a != b:
            return False
    return True


def validate_results(results: dict) -> dict:
    """Validate the horizon-keyed forecast, metric, DM, and SHAP schema."""
    required_sections = ("weeks", "forecast", "metrics", "dm_test", "shap")
    missing_sections = [section for section in required_sections if section not in results]
    if missing_sections:
        raise ValueError(f"Missing required sections: {', '.join(missing_sections)}")

    weeks = results["weeks"]
    forecast = results["forecast"]
    horizon_keys = [str(horizon) for horizon in range(1, 13)]
    if not weeks or any(key not in forecast for key in horizon_keys):
        raise ValueError("Forecast results must contain horizons 1 through 12.")
    expected = (results.get("meta") or {}).get("n_test")
    if expected is not None and len(weeks) != int(expected):
        raise ValueError(f"Expected {expected} common test target weeks, found {len(weeks)}.")
    reference_actual = None
    for key in horizon_keys:
        horizon_forecast = forecast[key]
        if any(
            len(horizon_forecast[model]) != len(weeks)
            for model in ("actual", "multivariate", "univariate")
        ):
            raise ValueError(
                f"Forecast series for horizon {key} must contain one value per week."
            )
        actual = horizon_forecast["actual"]
        if reference_actual is None:
            reference_actual = actual
        elif not _same_actual_series(actual, reference_actual):
            raise ValueError("All horizons must use the same observed target weeks.")

    for section_name, section in (("metrics", results["metrics"]), ("dm_test", results["dm_test"])):
        missing = [key for key in [*horizon_keys, "overall"] if key not in section]
        if missing:
            raise ValueError(
                f"{section_name} is missing horizon entries: {', '.join(missing)}"
            )
    for horizon in [*horizon_keys, "overall"]:
        for model in ("multivariate", "univariate"):
            for metric in ("rmse", "mae", "mape", "r2"):
                if metric not in results["metrics"][horizon][model]:
                    raise ValueError(
                        f"Missing {metric} for {model} horizon {horizon}."
                    )
    for horizon in [*horizon_keys, "overall"]:
        dm = results["dm_test"][horizon]
        if any(field not in dm for field in ("statistic", "p_value", "alpha", "decision")):
            raise ValueError(f"Incomplete DM test result for {horizon}.")

    for horizon in horizon_keys:
        if horizon not in results["shap"]:
            raise ValueError(f"SHAP results are missing horizon {horizon}.")
        for model in ("multivariate", "univariate"):
            if model not in results["shap"][horizon]:
                raise ValueError(f"SHAP results are missing {model} horizon {horizon}.")

    # Optional: results.json files written before the forward forecast existed
    # simply lack the key, and the dashboard then hides that section.
    if results.get("future_forecast") is not None:
        validate_future_forecast(results["future_forecast"])

    return results


# Dataset column -> dashboard label. ``future_forecast.shap_attributions`` is keyed by
# the raw dataset column; this mirrors ``MULTIVARIATE_FEATURES`` in model/config.py
# (the dashboard deliberately does not import the modelling package).
FUTURE_FEATURE_LABELS = {
    "diesel_price_php_liter": "Diesel Fuel Price",
    "usdphp_exchange_rate": "USD/PHP Exchange Rate",
    "rice_stockpile_mt": "Rice Stockpile Inventory",
    "rice_import_mt": "Rice Import Volume",
    "rice_price_php_kg average": "Retail Rice Price Lag",
    "palay_production_mt": "Palay Production",
}
FUTURE_HORIZON_WEEKS = 12
FUTURE_ADDITIVITY_TOLERANCE = 1e-3  # PHP/kg; the exporter guarantees 1e-4 after rounding


def future_feature_label(column: str) -> str:
    """Dashboard label for a dataset column; unknown columns get a readable fallback."""
    return FUTURE_FEATURE_LABELS.get(column, str(column).replace("_", " ").strip().title())


def validate_future_forecast(future: dict) -> dict:
    """Validate the ``future_forecast`` block (schema written by model/predict_future.py)."""
    required = (
        "origin_date", "horizon_dates", "multivariate_predictions",
        "univariate_predictions", "base_value", "shap_attributions",
    )
    missing = [key for key in required if key not in future]
    if missing:
        raise ValueError(f"future_forecast is missing: {', '.join(missing)}")

    dates = future["horizon_dates"]
    try:
        origin = datetime.strptime(future["origin_date"], "%Y-%m-%d")
        parsed = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
    except (TypeError, ValueError) as error:
        raise ValueError(f"future_forecast contains an invalid date: {error}") from error
    if len(dates) != FUTURE_HORIZON_WEEKS:
        raise ValueError(f"future_forecast needs {FUTURE_HORIZON_WEEKS} horizon dates, found {len(dates)}.")
    if parsed[0] <= origin or any(later <= earlier for earlier, later in zip(parsed, parsed[1:])):
        raise ValueError("future_forecast horizon dates must increase and start after the origin date.")

    for model in ("multivariate", "univariate"):
        values = future[f"{model}_predictions"]
        if len(values) != FUTURE_HORIZON_WEEKS or not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(
                f"future_forecast {model}_predictions must hold {FUTURE_HORIZON_WEEKS} finite values."
            )
    if not np.isfinite(float(future["base_value"])):
        raise ValueError("future_forecast base_value must be finite.")

    attributions = (future["shap_attributions"] or {}).get("multivariate")
    if not attributions:
        raise ValueError("future_forecast.shap_attributions needs a 'multivariate' entry.")
    for date, predicted in zip(dates, future["multivariate_predictions"]):
        if date not in attributions or not attributions[date]:
            raise ValueError(f"future_forecast SHAP attributions are missing {date}.")
        total = float(future["base_value"]) + sum(float(v) for v in attributions[date].values())
        if abs(total - float(predicted)) > FUTURE_ADDITIVITY_TOLERANCE:
            raise ValueError(
                f"future_forecast SHAP for {date} does not add up: base value plus contributions "
                f"is {total:.4f}, but the prediction is {float(predicted):.4f}."
            )
    return future


@st.cache_data(show_spinner=False)
def load_excel_data(file_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded workbooks once and return a session-safe DataFrame copy."""
    return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")


@st.cache_data(show_spinner=False)
def beeswarm_points(
    model: str,
    feature: str,
    shap_values: tuple[float, ...],
    feature_values: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(shap_values, dtype=float)
    features = np.asarray(feature_values, dtype=float)
    
    seed_text = f"{model}:{feature}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "big")
    jitter = np.random.default_rng(seed).random(len(values)) * 0.7 - 0.35
    
    return values, features, jitter


@st.cache_data(show_spinner=False)
def dependence_points(x_values: tuple[float, ...], y_values: tuple[float, ...]):
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def format_week_date(date_value: str) -> str:
    try:
        return datetime.strptime(date_value, "%Y-%m-%d").strftime("%B %d, %Y")
    except (TypeError, ValueError):
        return str(date_value)


_LAG_PATTERN = re.compile(r"^Price\(t-(\d+)\)$")


def feature_label(name: str, week: str | None = None) -> str:
    """Plain-language label for a SHAP feature name.
    "Price(t-2)" -> "Rice price 2 weeks ago". When ``week`` (the forecast week
    as "YYYY-MM-DD") is given, the real date is appended, e.g.
    "Rice price 2 weeks ago (Apr 13)". Any other name (the multivariate
    indicators) is returned unchanged.
    """
    match = _LAG_PATTERN.match(str(name).strip())
    if not match:
        return name

    weeks_back = int(match.group(1))
    unit = "week" if weeks_back == 1 else "weeks"
    label = f"Rice price {weeks_back} {unit} ago"

    if week:
        try:
            lagged = datetime.strptime(week, "%Y-%m-%d") - timedelta(weeks=weeks_back)
            label += f" ({lagged:%b %d})"
        except (TypeError, ValueError):
            pass
            
    return label
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
    if len(weeks) != 66:
        raise ValueError(f"Expected 66 common test target weeks, found {len(weeks)}.")
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
        elif actual != reference_actual:
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

    return results

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
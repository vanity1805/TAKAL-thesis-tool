"""Shared configuration and cached data helpers for the TAKAL dashboard."""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime

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
    """Validate the fields shared by the dashboard's overview and analysis views."""
    required_sections = ("weeks", "forecast", "metrics", "dm_test", "shap")
    missing_sections = [section for section in required_sections if section not in results]
    if missing_sections:
        raise ValueError(f"Missing required sections: {', '.join(missing_sections)}")

    weeks = results["weeks"]
    forecast = results["forecast"]
    if not weeks or any(len(forecast[key]) != len(weeks) for key in ("actual", "multivariate", "univariate")):
        raise ValueError("Forecast series must each contain one value for every week.")
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
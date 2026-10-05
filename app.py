"""TAKAL Streamlit entry point."""

from __future__ import annotations

import json
import os

import streamlit as st

from components import apply_styles
from utils import LOGO_PATH, RESULTS_PATH, load_results, validate_results
from views.benchmarking import render as render_benchmarking
from views.explainability import render as render_explainability
from views.overview import render as render_overview


st.set_page_config(
    page_title="TAKAL | Rice Price Forecasts",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()

try:
    RESULTS = validate_results(load_results(RESULTS_PATH, os.path.getmtime(RESULTS_PATH)))
except FileNotFoundError:
    st.error("`artifacts/results.json` was not found. Generate it with `python -m model.run_all`.")
    st.stop()
except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
    st.error(f"The saved forecast results could not be read: {error}")
    st.stop()

st.session_state.setdefault("technical_details", False)
st.session_state.setdefault("technical_tabs_version", 0)


def reset_to_overview() -> None:
    if not st.session_state.get("technical_details", False):
        st.toast("You are already on the Overview section.")
        return
    st.session_state["overview_page"] = 0
    st.session_state["technical_tab_default"] = "Forecast & Overview"
    st.session_state["technical_tabs_version"] += 1


with st.sidebar:
    st.button("TAKAL", key="home_reset", on_click=reset_to_overview, use_container_width=True)

    model_label = st.selectbox(
        "Forecast shown",
        ["Multivariate model", "Univariate baseline"],
        key="active_model_select",
    )
    active_model = "multivariate" if "Multivariate" in model_label else "univariate"

    st.divider()
    st.toggle("View Technical Details", key="technical_details")

weeks = RESULTS["weeks"]
if not st.session_state["technical_details"]:
    st.markdown('<div class="page-heading">Overview</div>', unsafe_allow_html=True)
    render_overview(
        weeks,
        RESULTS["forecast"],
        active_model,
        show_graph=False,
    )
else:
    tab_suffix = "\u200b" * int(st.session_state.get("technical_tabs_version", 0))
    tab_forecast, tab_metrics, tab_shap = st.tabs(
        [
            f"Forecast & Overview{tab_suffix}",
            f"Benchmarking / Metrics{tab_suffix}",
            f"Explainability / SHAP{tab_suffix}",
        ],
        default=(
            f"{st.session_state.get('technical_tab_default', 'Forecast & Overview')}{tab_suffix}"
            if st.session_state.get("technical_tab_default", "Forecast & Overview") == "Forecast & Overview"
            else st.session_state.get("technical_tab_default", "Forecast & Overview")
        ),
    )

    with tab_forecast:
        st.markdown('<div class="page-heading">Overview</div>', unsafe_allow_html=True)
        render_overview(weeks, RESULTS["forecast"], active_model, show_graph=True)

    with tab_metrics:
        st.markdown('<div class="page-heading">Benchmarking</div>', unsafe_allow_html=True)
        render_benchmarking(
            RESULTS["metrics"],
            RESULTS["dm_test"],
            len(weeks),
            active_model,
        )

    with tab_shap:
        st.markdown('<div class="page-heading">Explainability / SHAP</div>', unsafe_allow_html=True)
        render_explainability(RESULTS["shap"], weeks, active_model)
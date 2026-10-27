# =============================================================================
# Program      : TAKAL Streamlit dashboard (app.py)
# Description  : Interactive web dashboard that shows 12-week retail rice price
#                forecasts for the National Capital Region (NCR), how accurate
#                they were, and which economic factors drove each forecast.
# Author       : Ayson Rodel B., Chavez, Earl Justine B., Constantino, Ian Nathaniel M., and Samson, Lander S.
# System Fit   : Presentation layer of TAKAL. It never trains a model. It reads
#                artifacts/results.json (and artifacts/pipeline_status.json)
#                produced by the offline pipeline (python -m model.run_all) and
#                hands the data to the pages in views/. Start it with
#                `streamlit run app.py`.
# Date Written : 2026-10-14
# Date Revised : 2026-10-27
# Purpose      : Let non-technical stakeholders explore the forecasts, and show
#                a clear, formatted message (instead of a stack trace) if the NCR
#                rice price data cannot be loaded or the pipeline crashed.
# Algorithms   : No machine learning here. The page validates the saved results,
#                restores the user's choices from the URL so a browser reload
#                keeps them, and routes between Overview, 12-Week Forecast,
#                Benchmarking and Explainability (SHAP) pages. The numbers on
#                those pages come from a 3-layer Stacked LSTM and SHAP.
# Data         : A nested dictionary parsed from JSON (keyed by horizon "1" ..
# Structures     "12", then by model) holding the 66-week test series and the
#                12-week forecast; a list of week labels; Streamlit session
#                state and URL query parameters (both dictionary-like) that
#                remember the selected model and technical-details switch.
# =============================================================================
"""TAKAL Streamlit entry point."""

from __future__ import annotations

import json
import os
import traceback

import streamlit as st

from components import apply_styles
from export_ui import export_dialog
from utils import (
    LOGO_PATH,
    PIPELINE_STATUS_PATH,
    RESULTS_PATH,
    load_pipeline_status,
    load_results,
    render_error_panel,
    render_pipeline_status_error,
    validate_results,
)
from views.benchmarking import render as render_benchmarking
from views.explainability import render as render_explainability
from views.future_forecast import render as render_future_forecast
from views.overview import render as render_overview


st.set_page_config(
    page_title="TAKAL | Rice Price Forecasts",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()

RESULTS_RERUN_HINT = (
    "Regenerate the forecast by running `python -m model.run_all` in the project folder, "
    "then refresh this page."
)
SECTION_ERROR_HINT = (
    "Refresh the page. If the problem continues, regenerate the forecast by running "
    "`python -m model.run_all` in the project folder."
)


def load_dashboard_results() -> tuple[dict, float]:
    """Load and validate the saved forecast results, or show a formatted error and stop.

    Returns ``(results, modified_at)``. Every failure mode (file missing, damaged
    JSON, incomplete content, unreadable file, or an earlier pipeline crash) ends
    in ``render_error_panel`` followed by ``st.stop()`` so the page never shows a
    raw traceback.
    """
    pipeline_status = load_pipeline_status(PIPELINE_STATUS_PATH)
    pipeline_failed = pipeline_status is not None and pipeline_status.get("status") == "failed"

    try:
        modified_at = os.path.getmtime(RESULTS_PATH)
        results = validate_results(load_results(RESULTS_PATH, modified_at))
    except FileNotFoundError:
        if pipeline_failed:
            # The pipeline crashed before it could write results (for example, the
            # NCR rice price workbook failed to load): explain that crash instead.
            render_pipeline_status_error(pipeline_status)
        else:
            render_error_panel(
                "The forecast results have not been generated yet",
                "The dashboard needs `artifacts/results.json`, which is created by the "
                "forecasting pipeline from the NCR rice price dataset.",
                stage="Loading the saved forecast results",
                hint=RESULTS_RERUN_HINT,
            )
        st.stop()
    except json.JSONDecodeError as error:
        render_error_panel(
            "The saved forecast file is damaged",
            "`artifacts/results.json` is not valid JSON, so the NCR rice price forecasts "
            "cannot be shown.",
            stage="Reading the saved forecast results",
            hint=RESULTS_RERUN_HINT,
            details=f"{type(error).__name__}: {error}",
        )
        st.stop()
    except (KeyError, TypeError, ValueError) as error:
        render_error_panel(
            "The saved forecast file is incomplete or out of date",
            f"The forecast results failed the dashboard's checks: {error}",
            stage="Validating the saved forecast results",
            hint=RESULTS_RERUN_HINT,
            details=f"{type(error).__name__}: {error}",
        )
        st.stop()
    except OSError as error:
        render_error_panel(
            "The saved forecast file could not be opened",
            "The dashboard was not allowed to read `artifacts/results.json`, or the disk "
            "returned an error.",
            stage="Opening the saved forecast results",
            hint="Check that the file exists and is not open or locked by another program, "
                 "then refresh this page.",
            details=f"{type(error).__name__}: {error}",
        )
        st.stop()
    except Exception as error:  # Last-resort guard so the page never shows a raw traceback.
        render_error_panel(
            "The dashboard could not start",
            f"An unexpected problem occurred while loading the forecast ({type(error).__name__}).",
            stage="Loading the saved forecast results",
            hint=RESULTS_RERUN_HINT,
            details=traceback.format_exc(),
        )
        st.stop()

    # The results still load, but the latest pipeline run failed after they were
    # written: tell the reader these are not from the newest run.
    if pipeline_failed:
        try:
            status_is_newer = os.path.getmtime(PIPELINE_STATUS_PATH) > modified_at
        except OSError:
            status_is_newer = False
        if status_is_newer:
            st.warning(
                f"The latest forecasting run failed at **{pipeline_status.get('stage', 'an unknown step')}**, "
                "so you are viewing results from an earlier successful run. "
                f"{pipeline_status.get('hint') or RESULTS_RERUN_HINT}",
                icon=":material/warning:",
            )
    return results, modified_at


def render_section(section_name: str, render_function, *args, **kwargs) -> None:
    """Draw one dashboard section; show a formatted message if it crashes.

    A failure in one tab (for example, a missing field in the SHAP results) then
    stays inside that tab instead of taking down the whole page.
    """
    try:
        render_function(*args, **kwargs)
    except Exception as error:  # Streamlit's stop / rerun signals are not Exceptions.
        render_error_panel(
            f"The {section_name} section could not be displayed",
            f"Something unexpected went wrong while drawing this section ({type(error).__name__}).",
            stage=f"Displaying the {section_name} section",
            hint=SECTION_ERROR_HINT,
            details=traceback.format_exc(),
        )


results, results_modified_at = load_dashboard_results()

# --- Restore state from URL Query Params to survive browser reloads ---
if "technical_details" not in st.session_state:
    st.session_state["technical_details"] = st.query_params.get("tech", "false").lower() == "true"

if "active_model_select" not in st.session_state:
    url_model = st.query_params.get("model", "multivariate").lower()
    if "univariate" in url_model:
        st.session_state["active_model_select"] = "Univariate baseline"
    else:
        st.session_state["active_model_select"] = "Multivariate model"

st.session_state.setdefault("technical_details", False)
st.session_state.setdefault("technical_tabs_version", 0)


def reset_to_overview() -> None:
    """Sidebar "TAKAL" button: jump back to the first technical tab, or just say we are home."""
    if st.session_state.get("technical_details", False):
        st.session_state["overview_page"] = 0
        st.session_state["technical_tab_default"] = "Forecast & Overview"
        st.session_state["technical_tabs_version"] += 1
    st.toast("You are already viewing the Overview section.")


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
    st.divider()
    if st.button("Export report", key="export_open", icon=":material/download:", use_container_width=True):
        render_section("report export", export_dialog, results, results_modified_at, active_model)

# --- Sync the URL with the current session state ---
st.query_params["tech"] = str(st.session_state["technical_details"]).lower()
st.query_params["model"] = "univariate" if active_model == "univariate" else "multivariate"

weeks = results["weeks"]
future_forecast = results.get("future_forecast")

if not st.session_state["technical_details"]:
    st.markdown('<div class="page-heading">Overview</div>', unsafe_allow_html=True)
    render_section(
        "overview",
        render_overview,
        weeks,
        results["forecast"],
        active_model,
        show_graph=False,
        future_forecast=future_forecast,
    )
else:
    tab_suffix = "​" * int(st.session_state.get("technical_tabs_version", 0))
    tab_forecast, tab_future, tab_metrics, tab_shap = st.tabs(
        [
            f"Forecast & Overview{tab_suffix}",
            f"12-Week Forecast{tab_suffix}",
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
        render_section(
            "overview",
            render_overview,
            weeks,
            results["forecast"],
            active_model,
            show_graph=True,
            future_forecast=future_forecast,
        )

    with tab_future:
        st.markdown('<div class="page-heading">12-Week Forecast</div>', unsafe_allow_html=True)
        render_section("12-week forecast", render_future_forecast, results, active_model)

    with tab_metrics:
        st.markdown('<div class="page-heading">Benchmarking</div>', unsafe_allow_html=True)
        render_section(
            "benchmarking",
            render_benchmarking,
            results["metrics"],
            results["dm_test"],
            len(weeks),
            active_model,
        )

    with tab_shap:
        st.markdown('<div class="page-heading">Explainability / SHAP</div>', unsafe_allow_html=True)
        render_section("explainability (SHAP)", render_explainability, results["shap"], weeks, active_model)

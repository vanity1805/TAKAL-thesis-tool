"""TAKAL Streamlit entry point."""

from __future__ import annotations

import hashlib
import json
import os

import streamlit as st

from components import apply_styles, review_import_dialog
from utils import LOGO_PATH, RESULTS_PATH, load_excel_data, load_results, validate_results
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
st.session_state.setdefault("handled_upload_fingerprint", None)
st.session_state.setdefault("pending_import", None)
st.session_state.setdefault("imported_data", None)
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
        key="active_model",
    )
    active_model = "multivariate" if model_label.startswith("Multivariate") else "univariate"

    st.divider()
    st.subheader("Import spreadsheet")
    uploaded_file = st.file_uploader(
        "Choose an Excel workbook",
        type=["xlsx"],
        accept_multiple_files=False,
        help="Review and edit the workbook before storing it in this session.",
    )
    if st.session_state.get("imported_filename"):
        imported_frame = st.session_state.get("imported_data")
        if imported_frame is not None:
            st.caption(f"Session import: {st.session_state['imported_filename']} · {len(imported_frame):,} rows")
    st.toggle("View Technical Details", key="technical_details")

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    upload_fingerprint = hashlib.sha256(uploaded_file.name.encode("utf-8") + uploaded_bytes).hexdigest()
    if upload_fingerprint != st.session_state.get("handled_upload_fingerprint"):
        st.session_state["handled_upload_fingerprint"] = upload_fingerprint
        try:
            preview_data = load_excel_data(uploaded_bytes)
            if preview_data.empty and len(preview_data.columns) == 0:
                st.warning("The workbook has no readable table to import.")
            else:
                st.session_state["pending_import"] = {
                    "filename": uploaded_file.name,
                    "data": preview_data,
                    "editor_key": f"excel_preview_{upload_fingerprint[:12]}",
                }
        except Exception as error:
            st.error(f"Could not read this `.xlsx` workbook: {error}")

if st.session_state.get("pending_import"):
    review_import_dialog()

if st.session_state.get("import_notice"):
    st.success(st.session_state.pop("import_notice"))

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
"""Export dialog: pick sections and a file format, then download a PDF or Excel report."""

from __future__ import annotations

from datetime import date

import streamlit as st

from report_data import MODEL_TITLES, SECTION_BLURBS, SECTION_ORDER, build_report
from report_excel import build_excel
from report_pdf import build_pdf
from utils import BRAND_LOGO_PATH, LOGO_PATH, feature_label, format_week_date

SECTION_KEYS = {"Overview": "export_sec_overview", "Benchmarking": "export_sec_benchmarking",
                "Explainability": "export_sec_explainability"}
FILE_TYPES = {
    "PDF": ("pdf", "application/pdf"),
    "Excel": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
}


@st.cache_data(show_spinner="Preparing your file...", max_entries=8)
def _make_file(
    _results: dict, results_stamp: float, file_format: str, sections: tuple[str, ...],
    model: str, horizon: int, local_week: str | None, day: str,
) -> bytes:
    report = build_report(_results, sections, model, horizon, local_week, feature_label)
    if file_format == "PDF":
        import os
        logo = BRAND_LOGO_PATH if os.path.exists(BRAND_LOGO_PATH) else LOGO_PATH
        return build_pdf(report, logo_path=logo)
    return build_excel(report)


def _select_all_changed() -> None:
    for key in SECTION_KEYS.values():
        st.session_state[key] = st.session_state["export_select_all"]


def _section_changed() -> None:
    st.session_state["export_select_all"] = all(st.session_state[k] for k in SECTION_KEYS.values())


@st.dialog("Export report", width="large")
def export_dialog(results: dict, results_stamp: float, active_model: str) -> None:
    weeks = results["weeks"]
    for key in SECTION_KEYS.values():
        st.session_state.setdefault(key, True)
    st.session_state.setdefault("export_select_all", all(st.session_state[k] for k in SECTION_KEYS.values()))

    st.markdown("**1. Choose what to include**")
    st.checkbox("Select all sections", key="export_select_all", on_change=_select_all_changed)
    for name in SECTION_ORDER:
        st.checkbox(f"{name}  —  {SECTION_BLURBS[name]}", key=SECTION_KEYS[name], on_change=_section_changed)
    chosen = tuple(n for n in SECTION_ORDER if st.session_state[SECTION_KEYS[n]])

    st.markdown("**2. Choose the file type**")
    file_format = st.radio("File type", list(FILE_TYPES), horizontal=True, key="export_format", label_visibility="collapsed")

    needs_horizon = "Overview" in chosen or "Explainability" in chosen
    horizon = int(st.session_state.get("overview_horizon", 12))
    local_week = None
    if needs_horizon:
        st.markdown("**3. Report settings**")
        columns = st.columns(2)
        with columns[0]:
            horizon = st.selectbox("Forecast horizon (weeks ahead)", list(range(1, 13)),
                                   index=horizon - 1, key="export_horizon")
        if "Explainability" in chosen:
            with columns[1]:
                local_week = st.selectbox("Week for the single-week explanation", weeks, index=len(weeks) - 1,
                                          format_func=format_week_date, key="export_week")
    model_name = MODEL_TITLES[active_model]
    st.caption(
        f"The report uses the **{model_name}** currently selected in the sidebar"
        + (" (Benchmarking always shows both models side by side)." if "Benchmarking" in chosen else ".")
    )

    if not chosen:
        st.warning("Select at least one section to export.")
        st.download_button("Download", data=b"", disabled=True, key="export_download_disabled")
        return

    extension, mime = FILE_TYPES[file_format]
    data = _make_file(results, results_stamp, file_format, chosen, active_model, int(horizon), local_week, date.today().isoformat())
    stem = "_".join(s.lower() for s in chosen) if len(chosen) < len(SECTION_ORDER) else "full_report"
    st.download_button(
        f"Download {file_format} ({len(data) / 1024:,.0f} KB)",
        data=data,
        file_name=f"TAKAL_{stem}_{active_model}_{date.today():%Y%m%d}.{extension}",
        mime=mime,
        type="primary",
        on_click="ignore",
        key="export_download",
    )

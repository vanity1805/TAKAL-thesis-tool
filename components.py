"""Reusable visual components for the TAKAL Streamlit dashboard."""

from __future__ import annotations

import base64
import html
import os

import plotly.graph_objects as go
import streamlit as st

from utils import COLORS, LOGO_PATH


def apply_styles() -> None:
    logo_b64 = ""
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as logo_file:
            logo_b64 = base64.b64encode(logo_file.read()).decode("ascii")

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
          :root {{
            --primary-color: {COLORS['multivariate']};
            --primary: {COLORS['multivariate']};
            --primary-color-hover: #0284c7;
            --primary-color-active: #0284c7;
            --takal-emerald: {COLORS['actual']};
            --takal-blue: {COLORS['multivariate']};
            --takal-slate: {COLORS['univariate']};
            --takal-white: {COLORS['white']};
            --takal-paper: {COLORS['off_white']};
            --takal-border: {COLORS['border']};
            --takal-text: {COLORS['text']};
            --takal-muted: {COLORS['muted']};
          }}
          html, body, [class*="css"], .stApp {{
            color: var(--takal-text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
          }}
          .stApp {{ background: var(--takal-paper); }}
          [data-testid="stAppViewContainer"] .main .block-container {{
            max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;
          }}
          [data-testid="stSidebar"] {{ background: var(--takal-white); border-right: 1px solid var(--takal-border); }}
          [data-testid="stSidebar"] > div:first-child {{
            min-height: 100vh; padding-top: 1.25rem; padding-bottom: 2.75rem;
          }}
          [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            min-height: calc(100vh - 4rem); display: flex; flex-direction: column;
          }}
          .st-key-home_reset button {{
            min-height: 3.75rem; justify-content: flex-start; background: var(--takal-white);
            border: 1px solid transparent; color: var(--takal-text); box-shadow: none;
          }}
          .st-key-home_reset button:hover {{ background: var(--takal-paper); border-color: var(--takal-border); color: var(--takal-text); }}
          .st-key-home_reset button::before {{
            content: ""; width: 2.75rem; height: 2.75rem; flex: 0 0 2.75rem;
            margin-right: .65rem; background: center / contain no-repeat url("data:image/png;base64,{logo_b64}");
          }}
          .st-key-home_reset button p {{ font-size: 1.2rem; font-weight: 750; }}
          .sidebar-branding {{ margin: -.15rem 0 1rem .25rem; }}
          .sidebar-branding__title {{ color: var(--takal-text); font-size: 1.1rem; font-weight: 800; }}
          .sidebar-branding__subtitle {{ color: var(--takal-muted); font-size: .72rem; line-height: 1.45; }}
          [data-testid="stSidebar"] [data-testid="stFileUploader"] {{ margin-top: -.35rem; }}
          [data-testid="stSidebar"] [data-testid="stFileUploader"] > section {{ gap: .35rem; }}
          [data-testid="stSidebar"] [data-testid="stFileUploader"] label {{
            color: var(--takal-text); font-size: .84rem; font-weight: 600; margin-bottom: .15rem;
          }}
          [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
            min-height: 0; padding: .75rem .8rem; border: 1px dashed var(--takal-border);
            border-radius: 6px; background: var(--takal-paper);
          }}
          [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div {{ gap: .35rem; }}
          [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {{
            color: var(--takal-muted); font-size: .7rem; line-height: 1.25;
          }}
          [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
            min-height: 2rem; padding: .35rem .7rem; border: 1px solid var(--takal-blue);
            border-radius: 5px; background: var(--takal-blue); color: var(--takal-white);
            font-size: .78rem; font-weight: 600;
          }}
          [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {{
            border-color: #0284c7; background: #0284c7; color: var(--takal-white);
          }}
          [data-testid="stSidebar"] [data-testid="stFileUploader"] + div {{ margin-top: .35rem; }}
          [data-testid="stSidebar"] [data-testid="stFileUploader"] + div [data-testid="stToggle"] {{ margin-top: 0; }}
          [data-testid="stSidebar"] .st-key-technical_details {{
            margin-top: auto; padding: .85rem 0 1.25rem; border-top: 1px solid var(--takal-border);
          }}
          h1, h2, h3 {{ color: var(--takal-text); letter-spacing: 0; }}
          .page-heading {{ color: var(--takal-text) !important; font-size: 1.8rem; font-weight: 750; margin: .1rem 0 .2rem; }}
          [data-baseweb="tab-highlight"] {{ background-color: var(--takal-blue) !important; }}
          button[role="tab"][aria-selected="true"] {{ color: var(--takal-blue) !important; }}
          [data-testid="stToggle"] input:checked + div {{ background-color: var(--takal-blue) !important; border-color: var(--takal-blue) !important; }}
          [data-testid="stToggle"] input:checked + div:hover {{ background-color: #0284c7 !important; border-color: #0284c7 !important; }}
          div[data-baseweb="select"] > div,
          div[data-baseweb="select"] [role="combobox"] {{ border-color: var(--takal-border) !important; box-shadow: none !important; }}
          div[data-baseweb="select"] [role="combobox"]:focus,
          div[data-baseweb="select"] [role="combobox"][aria-expanded="true"],
          div[data-baseweb="select"] > div:focus-within {{ border-color: var(--takal-blue) !important; box-shadow: 0 0 0 1px var(--takal-blue) !important; }}
          div[data-baseweb="select"] [role="combobox"], div[data-baseweb="select"] input,
          div[data-baseweb="select"] * {{ caret-color: transparent !important; cursor: pointer !important; user-select: none !important; }}
          div[data-baseweb="select"] [data-type="selected"] {{ background-color: rgba(14, 165, 233, .08) !important; color: var(--takal-text) !important; }}
          .stButton button[kind="primary"] {{ background-color: var(--takal-blue); border-color: var(--takal-blue); }}
          .stButton button[kind="primary"]:hover {{ background-color: #0284c7; border-color: #0284c7; }}
          .page-intro {{ color: var(--takal-muted); margin-bottom: 1.2rem; }}
          .metric-card {{
            background: var(--takal-white); border: 1px solid var(--takal-border); border-radius: 6px;
            padding: 1.1rem 1.2rem; min-height: 126px;
          }}
          .metric-label {{ color: var(--takal-muted); font-size: .76rem; font-weight: 700; text-transform: uppercase; }}
          .metric-label-row {{ display: flex; align-items: center; gap: .4rem; }}
          .metric-info {{
            position: relative; display: inline-flex; align-items: center; justify-content: center;
            width: 1rem; height: 1rem; border: 1px solid var(--takal-border); border-radius: 50%;
            color: var(--takal-muted); font-size: .68rem; font-weight: 700; text-transform: none;
            cursor: help; outline-offset: 3px;
          }}
          .metric-info::after {{
            content: attr(data-tooltip); position: absolute; z-index: 20; left: 50%; bottom: calc(100% + .6rem);
            width: 220px; padding: .65rem .75rem; border: 1px solid var(--takal-border); border-radius: 5px;
            background: var(--takal-text); color: var(--takal-white); font-size: .76rem; font-weight: 400;
            line-height: 1.4; text-transform: none; box-shadow: 0 5px 16px rgba(15, 23, 42, .14);
            opacity: 0; pointer-events: none; transform: translate(-50%, .2rem); transition: opacity .15s ease, transform .15s ease;
          }}
          .metric-info:hover::after, .metric-info:focus::after {{ opacity: 1; transform: translate(-50%, 0); }}
          .metric-value {{ color: var(--takal-text); font-size: 1.7rem; line-height: 1.35; font-weight: 750; font-variant-numeric: tabular-nums; }}
          .metric-detail {{ color: var(--takal-muted); font-size: .82rem; }}
          [data-testid="stExpander"] {{ border: 1px solid var(--takal-border); border-radius: 6px; background: var(--takal-white); }}
          [data-testid="stExpander"] summary {{ padding: .55rem .8rem; }}
          [data-testid="stExpander"] .streamlit-expanderContent {{ padding: .2rem .8rem .7rem !important; }}
          .weekly-table-wrap {{ overflow-x: auto; border: 1px solid var(--takal-border); border-radius: 6px; background: var(--takal-white); }}
          table.weekly-table {{ width: 100%; border-collapse: collapse; color: var(--takal-text); }}
          .weekly-table thead th {{ position: sticky; top: 0; z-index: 1; background: var(--takal-white); }}
          .weekly-table th {{ padding: .85rem 1rem; border-bottom: 1px solid var(--takal-border); color: var(--takal-muted); font-size: .8rem; font-weight: 700; text-align: left; white-space: nowrap !important; }}
          .weekly-table th:nth-child(2), .weekly-table th:nth-child(3), .weekly-table td:nth-child(2), .weekly-table td:nth-child(3) {{ text-align: right; }}
          .weekly-table td {{ padding: .82rem 1rem; border-bottom: 1px solid var(--takal-border); font-size: .98rem; white-space: nowrap !important; }}
          .weekly-table tbody tr:last-child td {{ border-bottom: 0; }}
          .weekly-table tbody tr:hover {{ background: var(--takal-paper); }}
          .weekly-table tbody tr.current-week {{ background: rgba(16, 185, 129, .09); }}
          .weekly-table tbody tr.current-week:hover {{ background: rgba(16, 185, 129, .14); }}
          .weekly-table-date {{ font-weight: 550; }}
          .weekly-current-badge {{ display: block; margin-top: .15rem; color: var(--takal-emerald); font-size: .72rem; font-weight: 700; }}
          .weekly-forecast {{ color: var(--takal-blue); font-weight: 750; font-variant-numeric: tabular-nums; }}
          .weekly-actual {{ color: var(--takal-emerald); font-weight: 750; font-variant-numeric: tabular-nums; }}
          .weekly-not-yet {{ color: var(--takal-muted); font-style: italic; }}
          .pagination-label {{ padding-top: .45rem; color: var(--takal-muted); font-size: .82rem; }}
          .technical-card {{
            background: var(--takal-white); border: 1px solid var(--takal-border); border-radius: 6px;
            padding: 1rem 1.1rem; height: 100%;
          }}
          .technical-card-label {{ color: var(--takal-muted); font-size: .74rem; font-weight: 700; text-transform: uppercase; }}
          .technical-card-value {{ color: var(--takal-text); font-size: 1.35rem; font-weight: 750; margin-top: .3rem; }}
          .technical-card-detail {{ color: var(--takal-muted); font-size: .78rem; margin-top: .2rem; }}
          .result-panel {{ border-left: 4px solid var(--takal-blue); background: var(--takal-white); padding: 1rem 1.2rem; margin-top: 1rem; }}
          .chart-skeleton {{
            height: 380px; border-radius: 6px; border: 1px solid var(--takal-border);
            background: linear-gradient(90deg, var(--takal-paper) 25%, #eef2f6 50%, var(--takal-paper) 75%);
            background-size: 200% 100%; animation: takal-pulse 1.2s ease-in-out infinite;
          }}
          @keyframes takal-pulse {{ from {{ background-position: 100% 0; }} to {{ background-position: -100% 0; }} }}
          div[data-testid="stDataFrame"] {{ border: 1px solid var(--takal-border); border-radius: 6px; }}
          [data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding: .25rem 0; }}
          @media (prefers-reduced-motion: reduce) {{ .chart-skeleton {{ animation: none; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str, tooltip: str) -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_detail = html.escape(detail)
    safe_tooltip = html.escape(tooltip, quote=True)
    st.markdown(
        f"""
        <section class="metric-card" aria-label="{safe_label}: {safe_value}">
          <div class="metric-label-row">
            <span class="metric-label">{safe_label}</span>
            <span class="metric-info" tabindex="0" role="img" aria-label="About {safe_label}" data-tooltip="{safe_tooltip}">i</span>
          </div>
          <div class="metric-value">{safe_value}</div>
          <div class="metric-detail">{safe_detail}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_chart(placeholder, figure: go.Figure, key: str) -> None:
    placeholder.markdown(
        '<div class="chart-skeleton" role="status" aria-label="Rendering chart"></div>',
        unsafe_allow_html=True,
    )
    placeholder.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
        key=key,
    )


def chart_layout(figure: go.Figure, height: int = 420) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"t": 20, "r": 24, "b": 24, "l": 12},
        paper_bgcolor=COLORS["white"],
        plot_bgcolor=COLORS["white"],
        font={"family": "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", "color": COLORS["text"]},
        hoverlabel={"bgcolor": COLORS["white"], "font_color": COLORS["text"]},
    )
    figure.update_xaxes(gridcolor=COLORS["border"], zeroline=False)
    figure.update_yaxes(gridcolor=COLORS["border"], zeroline=False)
    return figure


@st.dialog("Review your spreadsheet", width="large")
def review_import_dialog() -> None:
    pending = st.session_state.get("pending_import")
    if not pending:
        st.info("There is no spreadsheet waiting for review.")
        return

    st.write(f"Review and correct **{pending['filename']}** before importing it into this session.")
    edited_data = st.data_editor(
        pending["data"],
        key=pending["editor_key"],
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
    )
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("Import reviewed data", type="primary", use_container_width=True):
            st.session_state["imported_data"] = edited_data
            st.session_state["imported_filename"] = pending["filename"]
            st.session_state["pending_import"] = None
            st.session_state["import_notice"] = (
                f"Imported {len(edited_data):,} editable rows from {pending['filename']}. "
                "These rows are stored in this session and do not replace the trained forecast results."
            )
            st.rerun()
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.session_state["pending_import"] = None
            st.rerun()
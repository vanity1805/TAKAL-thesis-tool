"""Reusable visual components for the TAKAL Streamlit dashboard."""

from __future__ import annotations

import base64
import html
import os

import plotly.graph_objects as go
import streamlit as st

from utils import BRAND_LOGO_PATH, COLORS


def apply_styles() -> None:
    brand_logo_b64 = ""
    if os.path.exists(BRAND_LOGO_PATH):
        with open(BRAND_LOGO_PATH, "rb") as brand_logo_file:
            brand_logo_b64 = base64.b64encode(brand_logo_file.read()).decode("ascii")

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&family=Work+Sans:wght@400;450;500;600&display=swap');
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
            font-family: 'Work Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
          }}
          .stApp {{ background: var(--takal-paper); }}
          [data-testid="stAppViewContainer"] .main .block-container {{
            max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;
          }}
          [data-testid="stSidebar"] {{
            background: var(--takal-white);
            border-right: 1px solid var(--takal-border);
          }}
          [data-testid="stSidebar"],
          [data-testid="stSidebar"] > div:first-child,
          [data-testid="stSidebar"] [data-testid="stSidebarContent"],
          [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            overflow: hidden !important;
            scrollbar-width: none !important;
          }}
          [data-testid="stSidebar"] *::-webkit-scrollbar {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
          }}
          [data-testid="stSidebar"] > div:first-child,
          [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            height: 100dvh !important;
            max-height: 100dvh !important;
            min-height: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            box-sizing: border-box !important;
          }}
          [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            padding-top: 0.5rem !important;
            padding-bottom: 1.25rem !important;
            display: flex !important;
            flex-direction: column !important;
            box-sizing: border-box !important;
          }}
          [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div,
          [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
          [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div,
          [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
            flex: 1 1 auto !important;
            display: flex !important;
            flex-direction: column !important;
            min-height: 0 !important;
          }}
          .st-key-home_reset {{
            display: flex !important;
            justify-content: center !important;
            margin: 0.75rem 0 1.25rem 0 !important;
          }}
          .st-key-home_reset button {{
            min-height: 10.5rem !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: transparent !important;
            border: 1px solid transparent !important;
            color: var(--takal-text);
            box-shadow: none !important;
            transition: none !important;
            padding: 0 !important;
          }}
          .st-key-home_reset button:hover,
          .st-key-home_reset button:focus,
          .st-key-home_reset button:active {{
            background: transparent !important;
            border-color: transparent !important;
            color: var(--takal-text) !important;
            box-shadow: none !important;
            transform: none !important;
          }}
          .st-key-home_reset button::before {{
            content: "";
            display: block;
            width: 86%;
            height: 9.5rem;
            margin: 0 auto;
            background: center / contain no-repeat url("data:image/png;base64,{brand_logo_b64}");
            filter: brightness(1.05) contrast(1.03);
          }}
          .st-key-home_reset button p {{
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
          }}
          [data-testid="stSidebar"] hr {{
            width: 100% !important;
            border: 0 !important;
            border-top: 1px solid var(--takal-border) !important;
            opacity: 1 !important;
          }}
          [data-testid="stSidebar"] .st-key-technical_details {{
            margin-top: auto !important;
            padding: .85rem 0 .25rem !important;
            border-top: 1px solid var(--takal-border) !important;
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
          }}
          [data-testid="stSidebar"] .st-key-technical_details > div,
          [data-testid="stSidebar"] .st-key-technical_details label {{
            width: fit-content !important;
            margin: 0 auto !important;
          }}
          h1, h2, h3 {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: var(--takal-text);
            letter-spacing: -0.025em;
          }}
          h1 {{ font-size: 2.25rem !important; font-weight: 800 !important; line-height: 1.2 !important; }}
          h2 {{ font-size: 1.75rem !important; font-weight: 750 !important; line-height: 1.25 !important; }}
          h3 {{ font-size: 1.35rem !important; font-weight: 700 !important; line-height: 1.3 !important; }}
          .page-heading {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: var(--takal-text) !important;
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.2;
            margin: .1rem 0 .3rem;
          }}
          [data-baseweb="tab-highlight"] {{ background-color: var(--takal-blue) !important; }}
          button[role="tab"][aria-selected="true"] {{ color: var(--takal-blue) !important; }}
          [data-testid="stToggle"] input:checked + div {{ background-color: var(--takal-blue) !important; border-color: var(--takal-blue) !important; }}
          [data-testid="stToggle"] input:checked + div:hover {{ background-color: #0284c7 !important; border-color: #0284c7 !important; }}
          
          div[data-baseweb="select"] > div {{
            border-color: var(--takal-border) !important;
            box-shadow: none !important;
          }}
          div[data-baseweb="select"] > div:focus-within {{
            border-color: var(--takal-blue) !important;
            box-shadow: 0 0 0 1px var(--takal-blue) !important;
          }}
          div[data-baseweb="select"] input,
          div[data-baseweb="select"] [role="combobox"] {{
            caret-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            width: 0 !important;
            min-width: 0 !important;
            opacity: 0 !important;
            pointer-events: none !important;
          }}
          div[data-baseweb="select"] * {{
            caret-color: transparent !important;
            cursor: pointer !important;
            user-select: none !important;
          }}
          div[data-baseweb="select"] [data-type="selected"] {{ background-color: rgba(14, 165, 233, .08) !important; color: var(--takal-text) !important; }}
          .stButton button[kind="primary"] {{ background-color: var(--takal-blue); border-color: var(--takal-blue); }}
          .stButton button[kind="primary"]:hover {{ background-color: #0284c7; border-color: #0284c7; }}
          .page-intro {{ color: var(--takal-muted); margin-bottom: 1.2rem; }}
          .metric-card {{
            background: var(--takal-white); border: 1px solid var(--takal-border); border-radius: 6px;
            padding: 1.1rem 1.2rem; min-height: 126px;
          }}
          .metric-label {{ color: var(--takal-muted); font-size: .76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }}
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
          .metric-value {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: var(--takal-text); font-size: 1.7rem; line-height: 1.35; font-weight: 800; font-variant-numeric: tabular-nums; letter-spacing: -0.02em;
          }}
          .metric-detail {{ color: var(--takal-muted); font-size: .82rem; }}
          [data-testid="stExpander"] {{ border: 1px solid var(--takal-border); border-radius: 6px; background: var(--takal-white); }}
          [data-testid="stExpander"] summary {{ padding: .55rem .8rem; }}
          [data-testid="stExpander"] .streamlit-expanderContent {{ padding: .2rem .8rem .7rem !important; }}
          .st-key-overview_default_button {{
            display: flex !important;
            justify-content: flex-end !important;
            width: 100% !important;
          }}
          .st-key-overview_default_button > div {{
            width: fit-content !important;
            margin-left: auto !important;
          }}
          .weekly-table-wrap {{ overflow-x: auto; border: 1px solid var(--takal-border); border-radius: 6px; background: var(--takal-white); margin-bottom: 1rem; }}
          table.weekly-table {{ width: 100%; border-collapse: collapse; color: var(--takal-text); }}
          .weekly-table thead th {{ position: sticky; top: 0; z-index: 1; background: var(--takal-white); }}
          .weekly-table th, .weekly-table td {{ padding: .82rem 1rem; border-bottom: 1px solid var(--takal-border); font-size: .98rem; text-align: left; white-space: nowrap !important; }}
          .weekly-table th {{ color: var(--takal-muted); font-size: .8rem; font-weight: 700; }}
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
          .weekly-table-wrap + div {{ margin-top: 1.5rem !important; }}
          .technical-card {{
            background: var(--takal-white); border: 1px solid var(--takal-border); border-radius: 6px;
            padding: 1rem 1.1rem; height: 100%;
          }}
          .technical-card-label {{ color: var(--takal-muted); font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }}
          .technical-card-value {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: var(--takal-text); font-size: 1.35rem; font-weight: 800; margin-top: .3rem; letter-spacing: -0.02em;
          }}
          .technical-card-detail {{ color: var(--takal-muted); font-size: .78rem; margin-top: .2rem; }}
          .result-panel {{ border-left: 4px solid var(--takal-blue); background: var(--takal-white); padding: 1rem 1.2rem; margin-top: 1rem; }}
          .chart-skeleton {{
            height: 380px; border-radius: 6px; border: 1px solid var(--takal-border);
            background: linear-gradient(90deg, var(--takal-paper) 25%, #eef2f6 50%, var(--takal-paper) 75%);
            background-size: 200% 100%; animation: takal-pulse 1.2s ease-in-out infinite;
          }}
          @keyframes takal-pulse {{ from {{ background-position: 100% 0; }} to {{ background-position: -100% 0; }} }}
          div[data-testid="stDataFrame"] {{ border: 1px solid var(--takal-border); border-radius: 6px; }}
          div[data-testid="stDataFrame"] [role="columnheader"],
          div[data-testid="stDataFrame"] [role="gridcell"],
          div[data-testid="stDataFrame"] [role="columnheader"] > div,
          div[data-testid="stDataFrame"] [role="gridcell"] > div {{
            text-align: left !important; justify-content: flex-start !important;
          }}
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
        font={"family": "Work Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", "color": COLORS["text"]},
        hoverlabel={"bgcolor": COLORS["white"], "font_color": COLORS["text"]},
    )
    figure.update_xaxes(gridcolor=COLORS["border"], zeroline=False)
    figure.update_yaxes(gridcolor=COLORS["border"], zeroline=False)
    return figure
"""
Time-Series Analysis & Knowledge Attribution for Local Rice (TAKAL) — Streamlit dashboard.

Displays REAL outputs of the Stacked LSTM pipeline. All predictions, metrics,
Diebold-Mariano results, and SHAP values are loaded from ``artifacts/results.json``,
which is produced by the offline model pipeline:

    python -m model.run_all

If that file is missing, run the pipeline first (see model/README.md).
"""

import json
import os
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Load real model results (produced by `python -m model.run_all`)
# --------------------------------------------------------------------------- #
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "results.json")


@st.cache_data(show_spinner=False)
def load_results(path: str, _mtime: float) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

# TAKAL Brand Palette (Strictly Adhered To)
CHART_COLORS = {
    "actual": "#4a7c59",       # Muted Green
    "multivariate": "#2c3e50", # Dark Blue-Gray
    "univariate": "#94a3b8"    # Neutral Slate
}
BAR_COLOR = "#2c3e50"

try:
    RESULTS = load_results(RESULTS_PATH, os.path.getmtime(RESULTS_PATH))
except FileNotFoundError:
    st.error(
        "**artifacts/results.json not found.**\n\n"
        "Train the models and export results first:\n\n"
        "```\npython -m model.run_all\n```"
    )
    st.stop()

WEEKS = RESULTS["weeks"]
N_WEEKS = len(WEEKS)
forecast_data = RESULTS["forecast"]
metrics = RESULTS["metrics"]
dm_test = RESULTS["dm_test"]

shap_data = {
    model: list(zip(RESULTS["shap"][model]["features"], RESULTS["shap"][model]["bar"]))
    for model in ("multivariate", "univariate")
}
BASE_VALUE = {model: RESULTS["shap"][model]["base_value"] for model in ("multivariate", "univariate")}

_JITTER_RNG = np.random.default_rng(42)

def _beeswarm_points(model: str, feature: str):
    d = RESULTS["shap"][model]["beeswarm"][feature]
    sv = np.asarray(d["shap"], dtype=float)
    fv = np.asarray(d["feature_value"], dtype=float)
    jitter = _JITTER_RNG.random(len(sv)) * 0.7 - 0.35
    return sv, fv, jitter

def _dependence_points(model: str, variable: str):
    d = RESULTS["shap"][model]["dependence"][variable]
    return np.asarray(d["x"], dtype=float), np.asarray(d["y"], dtype=float)

def local_shap(model: str, week_idx: int):
    entry = RESULTS["shap"][model]["local"][week_idx]
    contributions = entry["contributions"]
    feats = list(contributions.keys())
    contrib = np.array([contributions[f] for f in feats], dtype=float)
    base = BASE_VALUE[model]
    predicted = entry["predicted"]

    order = np.argsort(np.abs(contrib))
    feats = [feats[i] for i in order]
    contrib = contrib[order]
    return feats, contrib, base, predicted

def format_week_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B %d, %Y")
    except Exception:
        return date_str

# --------------------------------------------------------------------------- #
# Page config + CSS (Adaptive Modern SaaS Dashboard System)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="TAKAL - Time-Series Analysis & Knowledge Attribution for Local Rice",
    page_icon="Logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      
      :root {
        --brand-navy: #2c3e50;
        --brand-green: #4a7c59;
        --brand-slate: #64748b;
        --border-adaptive: rgba(128, 128, 128, 0.2); 
      }
      
      /* Base Typography */
      html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
      }
      
      .block-container { 
        padding-top: 3.5rem !important; 
        padding-bottom: 4rem;
        max-width: 1400px; 
      }

      /* Sidebar Customization */
      [data-testid="stSidebar"] {
        border-right: 1px solid var(--border-adaptive);
      }
      .sidebar-brand-wrapper {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin-bottom: 1.75rem;
      }
      .sidebar-brand-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
      }
      .sidebar-brand {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--brand-navy);
        line-height: 1;
        margin: 0;
      }
      .sidebar-subtitle {
        font-size: 0.72rem;
        color: var(--text-color);
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        line-height: 1.4;
        margin: 0;
      }

      /* Page & Section Headers */
      .page-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-color);
        margin-top: 2.5rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
      }
      .custom-tooltip {
        position: relative;
        display: inline-flex;
        align-items: center;
        cursor: help;
        color: var(--text-color);
        opacity: 0.4;
        transition: opacity 0.2s ease;
      }
      .custom-tooltip:hover {
        opacity: 0.8;
      }
      .custom-tooltip-text {
        visibility: hidden;
        width: max-content;
        max-width: 320px;
        background-color: var(--brand-navy);
        color: #ffffff;
        text-align: left;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        position: absolute;
        z-index: 9999;
        bottom: 130%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.2s ease, bottom 0.2s ease;
        font-size: 0.8rem;
        font-weight: 500;
        line-height: 1.5;
        letter-spacing: normal;
        text-transform: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      }
      .custom-tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: var(--brand-navy) transparent transparent transparent;
      }
      .custom-tooltip:hover .custom-tooltip-text {
        visibility: visible;
        opacity: 1;
        bottom: 150%;
      }
      .info-icon {
        width: 18px;
        height: 18px;
        fill: currentColor;
      }

      /* Base Flat Card Classes */
      .metric-card, .bench-card, .dm-result-card, .table-container {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-adaptive);
        border-radius: 8px;
        box-shadow: none !important;
      }

      /* Overview Metric Cards */
      .metric-card {
        padding: 1.25rem 1.5rem;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        height: 100%;
      }
      .metric-header {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-color);
        opacity: 0.7;
      }
      .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-color);
        line-height: 1.2;
        margin: 0;
        font-variant-numeric: tabular-nums;
      }
      .metric-sub {
        font-size: 0.8rem;
        color: var(--text-color);
        opacity: 0.8;
      }
      .metric-sub strong {
        color: var(--brand-green);
        font-weight: 600;
      }
      .metric-sub strong.neutral {
        color: var(--text-color);
        opacity: 0.9;
      }

      /* Benchmarking Specific Cards */
      .bench-card {
        padding: 1.25rem 1.5rem;
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        height: 100%;
      }
      .bench-header { font-size: 0.85rem; font-weight: 600; color: var(--text-color); opacity: 0.7; }
      .bench-value { font-size: 1.25rem; font-weight: 700; color: var(--text-color); }
      .bench-sub { font-size: 0.75rem; color: var(--text-color); opacity: 0.5; }

      .dm-result-card {
        padding: 1.5rem;
        margin-top: 1rem;
      }
      .dm-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.6; margin-bottom: 0.5rem;}
      .dm-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
      .dm-text { font-size: 0.9rem; opacity: 0.7; line-height: 1.5; }

      /* Modern Tables */
      .table-container {
        overflow: hidden;
      }
      table.modern-table { 
        width: 100%; 
        border-collapse: collapse; 
      }
      table.modern-table th, table.modern-table td { 
        padding: 1rem 1.5rem; 
        font-size: 0.85rem; 
      }
      table.modern-table th { 
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 600;
        text-align: right; 
        border-bottom: 1px solid var(--border-adaptive); 
        background-color: rgba(128, 128, 128, 0.05);
      }
      table.modern-table th:first-child, table.modern-table td:first-child { 
        text-align: left; 
      }
      table.modern-table td { 
        text-align: right; 
        color: var(--text-color);
        font-weight: 500;
        border-bottom: 1px solid var(--border-adaptive);
      }
      table.modern-table tr:last-child td {
        border-bottom: none;
      }

      /* --------------------------------------------------- */
      /* Streamlit UX & Color Overrides */
      /* --------------------------------------------------- */
      
      .st-key-active_model div[data-baseweb="select"] { border-radius: 6px; }

      div[data-baseweb="select"], 
      div[data-baseweb="select"] > div,
      div[data-baseweb="select"] input {
        cursor: pointer !important;
      }
      div[data-baseweb="select"] {
        user-select: none !important;
      }
      div[data-baseweb="select"] input {
        caret-color: transparent !important;
      }
      
      div[data-baseweb="select"] > div:hover {
        border-color: var(--brand-navy) !important;
      }
      div[data-baseweb="select"] > div:focus, 
      div[data-baseweb="select"] > div:focus-within, 
      div[data-baseweb="select"] > div:active {
        border-color: var(--brand-navy) !important;
        box-shadow: 0 0 0 1px var(--brand-navy) !important;
        outline: none !important;
      }
      ul[role="listbox"] li[aria-selected="true"] {
        background-color: rgba(44, 62, 80, 0.1) !important; 
        color: var(--brand-navy) !important;
      }

      /* --------------------------------------------------- */
      /* Sidebar Vertical Tabs (Buttons instead of Radio) */
      /* --------------------------------------------------- */
      [data-testid="stSidebar"] button {
        justify-content: flex-start !important;
        padding-left: 1rem !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
        height: 2.75rem !important;
      }
      [data-testid="stSidebar"] button p {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
      }
      [data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: var(--text-color) !important;
        opacity: 0.8;
      }
      [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: rgba(128, 128, 128, 0.08) !important;
        opacity: 1;
      }
      [data-testid="stSidebar"] button[kind="primary"] {
        background-color: var(--brand-navy) !important;
        border: 1px solid var(--brand-navy) !important;
        color: #ffffff !important;
      }
      [data-testid="stSidebar"] button[kind="primary"] p {
        font-weight: 600 !important;
      }

      /* Main Area Tabs (Bar / Beeswarm) */
      button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
      }
      button[data-baseweb="tab"]:focus {
        outline: none !important;
        box-shadow: none !important;
      }
      div[data-baseweb="tab-list"] button[aria-selected="true"] {
        color: var(--brand-navy) !important;
        font-weight: 600 !important;
      }
      div[data-baseweb="tab-highlight"] {
        background-color: var(--brand-navy) !important;
      }
      
      /* Global Focus Fallback */
      *:focus {
        outline: none !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

def render_section_header(title, tooltip_text):
    icon_svg = '''<svg class="info-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 15h-2v-6h2zm0-8h-2V7h2z"/></svg>'''
    return f"""
    <div class="section-title">
        {title}
        <div class="custom-tooltip">
            {icon_svg}
            <span class="custom-tooltip-text">{tooltip_text}</span>
        </div>
    </div>
    """

def render_metric(label, value, sub_val, sub_text, is_positive=True):
    color_class = "" if is_positive else "neutral"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-header">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub"><strong class="{color_class}">{sub_val}</strong> {sub_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_bench_card(title, value, sub):
    return f"""
    <div class="bench-card">
        <div class="bench-header">{title}</div>
        <div class="bench-value">{value}</div>
        <div class="bench-sub">{sub}</div>
    </div>
    """

# --------------------------------------------------------------------------- #
# State Initialization (Sidebar Tabs)
# --------------------------------------------------------------------------- #
if "nav_selection" not in st.session_state:
    st.session_state.nav_selection = "Overview"

def set_nav(selection):
    st.session_state.nav_selection = selection

# --------------------------------------------------------------------------- #
# Sidebar Layout
# --------------------------------------------------------------------------- #
with st.sidebar:
    logo_path = "Logo.png"
    encoded_logo = ""
    if os.path.exists(logo_path):
        import base64
        with open(logo_path, "rb") as f:
            encoded_logo = base64.b64encode(f.read()).decode()
    
    logo_img = f'<img src="data:image/png;base64,{encoded_logo}" style="width: 44px; height: auto;" />' if encoded_logo else ""

    st.markdown(
        f"""
        <div class="sidebar-brand-wrapper">
            <div class="sidebar-brand-container">
                {logo_img}
                <h1 class="sidebar-brand">TAKAL</h1>
            </div>
            <div class="sidebar-subtitle">Time-Series Analysis & Knowledge Attribution for Local Rice</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("### General")
    st.button("Overview", on_click=set_nav, args=["Overview"], use_container_width=True, type="primary" if st.session_state.nav_selection == "Overview" else "secondary")
    st.button("Benchmarking", on_click=set_nav, args=["Benchmarking"], use_container_width=True, type="primary" if st.session_state.nav_selection == "Benchmarking" else "secondary")
    st.button("Explainability", on_click=set_nav, args=["Explainability"], use_container_width=True, type="primary" if st.session_state.nav_selection == "Explainability" else "secondary")
    
    st.write("")
    st.write("### Configuration")
    model_label = st.selectbox(
        "Active Model",
        ["Multivariate Model", "Univariate Model (Baseline)"],
        key="active_model"
    )

active = "multivariate" if model_label.startswith("Multivariate") else "univariate"
active_name = "Multivariate" if active == "multivariate" else "Univariate"
nav_selection = st.session_state.nav_selection

# --------------------------------------------------------------------------- #
# Main Content Area
# --------------------------------------------------------------------------- #
st.markdown(f'<div class="page-header"><div>{nav_selection}</div></div>', unsafe_allow_html=True)

if nav_selection == "Overview":
    
    vals = forecast_data[active]
    avg_gap = float(np.mean(np.abs(np.array(vals) - np.array(forecast_data["actual"]))))
    change = vals[-1] - vals[0]
    pct = change / vals[0] * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Starting Price", f"₱{vals[0]:.2f}", format_week_date(WEEKS[0]), "Base week", True)
    with c2:
        render_metric("Ending Price", f"₱{vals[-1]:.2f}", format_week_date(WEEKS[-1]), "Forecast horizon", True)
    with c3:
        render_metric("Price Change", f"₱{change:+.2f}", f"{pct:+.1f}%", f"over {N_WEEKS} weeks", pct >= 0)
    with c4:
        render_metric("Avg. Gap vs Actual", f"₱{avg_gap:.2f}", "Mean", "Absolute Deviation", False)

    st.write("")

    tt_text = f"Predicted vs. actual observed retail rice prices over the {N_WEEKS}-week held-out test horizon (NCR, {WEEKS[0]} → {WEEKS[-1]}). The {active_name} model is active; the Actual Price line is the ground-truth reference."
    st.markdown(render_section_header(f"{N_WEEKS}-Week Forecast vs. Actual", tt_text), unsafe_allow_html=True)
    
    st.markdown(
        "<div style='font-size: 0.75rem; color: var(--text-color); opacity: 0.6; font-weight: 500; margin-top: -1rem; margin-bottom: 1rem;'>"
        "(Click legend items below to toggle visibility)</div>", 
        unsafe_allow_html=True
    )
    
    fig = go.Figure()

    def _line(key, name):
        is_active = key == active
        fig.add_trace(
            go.Scatter(
                x=WEEKS,
                y=forecast_data[key],
                name=name,
                mode="lines+markers" if is_active else "lines",
                line=dict(color=CHART_COLORS[key], width=2.5 if is_active else 1.5),
                opacity=1.0 if is_active else 0.4,
                hovertemplate="%{y:.2f} ₱/kg<extra>" + name + "</extra>",
            )
        )

    _line("multivariate", "Multivariate Model")
    _line("univariate", "Univariate Model")
    
    fig.add_trace(
        go.Scatter(
            x=WEEKS,
            y=forecast_data["actual"],
            name="Actual Price",
            mode="lines",
            line=dict(color=CHART_COLORS["actual"], width=2, dash="dash"),
            hovertemplate="%{y:.2f} ₱/kg<extra>Actual</extra>",
        )
    )
    
    _allv = forecast_data["actual"] + forecast_data["multivariate"] + forecast_data["univariate"]
    _ylo, _yhi = min(_allv), max(_allv)
    _ypad = max(0.5, (_yhi - _ylo) * 0.12)
    
    fig.update_yaxes(range=[_ylo - _ypad, _yhi + _ypad], tickprefix="₱", title=dict(text="Price (PHP/kg)", font=dict(size=13, family="Inter", color="#2c3e50")))
    fig.update_layout(
        height=450,
        margin=dict(t=10, r=20, l=10, b=20),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.03, 
            x=0,
            bgcolor="rgba(128, 128, 128, 0.05)",
            bordercolor="rgba(128, 128, 128, 0.2)",
            borderwidth=1,
            font=dict(size=13),
            itemwidth=40
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(family="Inter")
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128, 128, 128, 0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128, 128, 128, 0.15)", zeroline=False)
    
    st.plotly_chart(fig, use_container_width=True)


elif nav_selection == "Benchmarking":

    st.markdown(render_section_header("Model Performance Metrics", f"Regression benchmarking for the {active_name} model."), unsafe_allow_html=True)
    
    m = metrics[active]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_bench_card("RMSE", f"{m['rmse']}", "Root Mean Square Error"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_bench_card("MAE", f"{m['mae']}", "Mean Absolute Error"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_bench_card("MAPE", f"{m['mape']}%", "Mean Absolute Percentage Error"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_bench_card("R² Score", f"{m['r2']}", "Coefficient of Determination"), unsafe_allow_html=True)

    st.markdown(render_section_header("Model Comparison", "Regression metrics for the multivariate and univariate models evaluated over the same test period."), unsafe_allow_html=True)
    
    mv, uv = metrics["multivariate"], metrics["univariate"]
    st.markdown(
        f"""
        <div class="table-container">
            <table class="modern-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>RMSE</th>
                  <th>MAE</th>
                  <th>MAPE</th>
                  <th>R²</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style="font-weight: 600;">Multivariate Model</td>
                  <td>{mv['rmse']}</td>
                  <td>{mv['mae']}</td>
                  <td>{mv['mape']}%</td>
                  <td>{mv['r2']}</td>
                </tr>
                <tr>
                  <td style="font-weight: 600;">Univariate Model</td>
                  <td>{uv['rmse']}</td>
                  <td>{uv['mae']}</td>
                  <td>{uv['mape']}%</td>
                  <td>{uv['r2']}</td>
                </tr>
              </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(render_section_header("Diebold-Mariano Test Result", "Statistical comparison of forecast errors between the two models."), unsafe_allow_html=True)

    _sig = dm_test.get("significant", dm_test.get("p_value", 1) < dm_test.get("alpha", 0.05))
    stat = dm_test.get("statistic", 0.0) 
    pval = dm_test.get("p_value", 1.0)
    alpha = dm_test.get("alpha", 0.05)

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown(render_bench_card("Test Statistic", f"{stat:.4f}", ""), unsafe_allow_html=True)
    with dc2:
        st.markdown(render_bench_card("P-Value", f"{pval:.6f}", ""), unsafe_allow_html=True)
    with dc3:
        st.markdown(render_bench_card("Significance Level", f"α = {alpha}", ""), unsafe_allow_html=True)

    if _sig:
        result_title = "Reject H₀"
        result_desc = "The Diebold-Mariano test indicates a statistically significant difference in forecasting accuracy between the two models. The test result identifies the multivariate model as having the lower forecast loss under the evaluated loss function."
    else:
        result_title = "Fail to Reject H₀"
        result_desc = "The Diebold-Mariano test does not indicate a statistically significant difference in forecasting accuracy between the two models. Neither model demonstrates statistically superior forecast loss under the evaluated loss function."

    st.markdown(
        f"""
        <div class="dm-result-card">
            <div class="dm-label">STATISTICAL RESULT</div>
            <div class="dm-title">{result_title}</div>
            <div class="dm-text">{result_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif nav_selection == "Explainability":
    
    tooltip_shap = (
        "Measures how strongly each economic indicator drives retail rice price forecasts across the test period."
    )
    st.markdown(render_section_header("Global Feature Importance (SHapley Additive exPlanations)", tooltip_shap), unsafe_allow_html=True)
    
    features = [f for f, _ in shap_data[active]]
    values = [v for _, v in shap_data[active]]

    tab_bar, tab_bee = st.tabs(["Bar Chart", "Beeswarm Plot"])

    with tab_bar:
        fig_bar = go.Figure(
            go.Bar(
                x=values[::-1],
                y=features[::-1],
                orientation="h",
                marker_color=BAR_COLOR,
                marker_line_width=0,
                width=0.6,
                hovertemplate="%{x:.3f} ₱/kg<extra>%{y}</extra>",
            )
        )
        fig_bar.update_layout(
            height=350, 
            margin=dict(t=10, r=20, l=10, b=20),
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="mean(|SHAP value|) in ₱/kg",
            font=dict(family="Inter")
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor="rgba(128, 128, 128, 0.15)", range=[0, max(values) * 1.15])
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab_bee:
        fig_bee = go.Figure()
        for idx, feat in enumerate(features):
            sv, fv, jit = _beeswarm_points(active, feat)
            fig_bee.add_trace(
                go.Scatter(
                    x=sv,
                    y=np.full_like(sv, idx) + jit,
                    mode="markers",
                    marker=dict(
                        color=fv, 
                        colorscale=[[0, "#94a3b8"], [0.5, "#4a7c59"], [1, "#2c3e50"]],
                        cmin=0, cmax=1, size=6, opacity=0.8,
                        colorbar=dict(title="Value", tickvals=[0, 1], ticktext=["Low", "High"], thickness=10) if idx == 0 else None,
                        showscale=idx == 0,
                    ),
                    showlegend=False,
                    hovertemplate="SHAP %{x:.3f}<extra>" + feat + "</extra>",
                )
            )
        fig_bee.add_vline(x=0, line_color="rgba(128, 128, 128, 0.5)", line_width=1)
        fig_bee.update_yaxes(tickvals=list(range(len(features))), ticktext=features, autorange="reversed")
        fig_bee.update_xaxes(title="SHAP value (impact on model output)", showgrid=True, gridcolor="rgba(128, 128, 128, 0.15)", zeroline=False)
        fig_bee.update_layout(height=400, margin=dict(t=10, r=20, l=10, b=20), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
        st.plotly_chart(fig_bee, use_container_width=True)

    st.write("---")

    st.markdown(render_section_header("Local Explanation", "Per-prediction SHAP waterfall: how each indicator pushes a specific week's forecast up or down from the baseline to the final predicted price."), unsafe_allow_html=True)
    
    col_sel_wf, _ = st.columns([1, 3])
    with col_sel_wf:
        sel_week = st.selectbox(
            "Select forecast week", 
            WEEKS, 
            index=len(WEEKS) - 1, 
            label_visibility="collapsed",
            format_func=format_week_date
        )
        
    w_idx = WEEKS.index(sel_week)
    feats_l, contrib_l, base_v, pred_v = local_shap(active, w_idx)

    fig_wf = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(feats_l),
            y=feats_l,
            x=contrib_l,
            base=base_v,
            text=[f"{c:+.2f}" for c in contrib_l],
            textposition="outside",
            connector=dict(line=dict(color="rgba(128, 128, 128, 0.3)", width=1)),
            increasing=dict(marker=dict(color="#4a7c59")), 
            decreasing=dict(marker=dict(color="#2c3e50")), 
            hovertemplate="%{y}: %{x:+.3f} ₱/kg<extra></extra>",
        )
    )
    
    fig_wf.add_vline(x=base_v, line_dash="dash", line_color="rgba(128, 128, 128, 0.5)", annotation_text="Baseline", annotation_position="top")
    fig_wf.add_vline(x=pred_v, line_dash="solid", line_color="#2c3e50", annotation_text="Predicted", annotation_position="top")
    
    pad = max(0.4, abs(pred_v - base_v) * 0.6)
    fig_wf.update_xaxes(title="Predicted price (PHP/kg)", showgrid=True, gridcolor="rgba(128, 128, 128, 0.15)", range=[min(base_v, pred_v) - pad, max(base_v, pred_v) + pad])
    fig_wf.update_layout(height=350, margin=dict(t=30, r=20, l=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown(
        "<div style='display: flex; justify-content: center; gap: 2rem; font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1.5rem; color: var(--text-color); opacity: 0.8;'>"
        "<div><span style='color:#4a7c59'>■</span> Raises price</div>"
        "<div><span style='color:#2c3e50'>■</span> Lowers price</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    move = pred_v - base_v
    top_feat = feats_l[int(np.argmax(np.abs(contrib_l)))]
    top_val = contrib_l[int(np.argmax(np.abs(contrib_l)))]
    direction = "increase" if move >= 0 else "decrease"
    verbose_date = format_week_date(sel_week)
    
    st.markdown(
        f"<div style='font-size: 0.95rem; color: var(--text-color); margin-bottom: 2rem; line-height: 1.5; padding: 0.5rem 0;'>"
        f"<strong>{verbose_date}</strong>: the {active_name} model predicts <strong>₱{pred_v:.2f}/kg</strong>, a "
        f"<strong>₱{abs(move):.2f}</strong> {direction} from the baseline. The largest single driver is "
        f"<strong>{top_feat}</strong> ({top_val:+.2f} ₱/kg)."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.write("---")

    st.markdown(render_section_header("Feature Dependence Analysis", "How an individual variable drives model attribution across test observations."), unsafe_allow_html=True)
    
    col_sel_dep, _ = st.columns([1, 3])
    with col_sel_dep:
        dep_var = st.selectbox("Isolate variable", [f for f, _ in shap_data[active]])
        
    dx, dy = _dependence_points(active, dep_var)
    colors = ["#4a7c59" if v > 0 else "#2c3e50" for v in dy]
    
    fig_dep = go.Figure(
        go.Scatter(
            x=dx, y=dy, mode="markers",
            marker=dict(color=colors, opacity=0.7, size=7, line=dict(width=0.5, color="rgba(128,128,128,0.2)")),
            hovertemplate=f"{dep_var}: " + "%{x:.2f}<br>Attribution: %{y:+.3f} ₱/kg<extra></extra>",
        )
    )
    fig_dep.update_xaxes(title=dep_var, showgrid=True, gridcolor="rgba(128, 128, 128, 0.15)", zeroline=False)
    fig_dep.update_yaxes(title="Price (PHP/kg)", showgrid=True, gridcolor="rgba(128, 128, 128, 0.15)", zeroline=False)
    fig_dep.update_layout(height=400, margin=dict(t=10, r=20, l=10, b=20), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
    st.plotly_chart(fig_dep, use_container_width=True)
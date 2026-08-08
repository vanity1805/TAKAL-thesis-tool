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

CHART_COLORS = {"actual": "#10b981", "multivariate": "#0ea5e9", "univariate": "#64748b"}
BAR_COLORS = ["#f59e0b", "#fbbf24", "#fcd34d", "#fde047", "#fef08a", "#fef3c7"]

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

# Ordered (feature, mean|SHAP|) pairs per model — drives the bar chart and the
# variable pickers. SHAP values are expressed in PHP/kg.
shap_data = {
    model: list(zip(RESULTS["shap"][model]["features"], RESULTS["shap"][model]["bar"]))
    for model in ("multivariate", "univariate")
}
# Baseline E[f(x)] each local explanation departs from (PHP/kg).
BASE_VALUE = {model: RESULTS["shap"][model]["base_value"] for model in ("multivariate", "univariate")}

# Seeded jitter only affects the vertical spread of beeswarm points (cosmetic).
_JITTER_RNG = np.random.default_rng(42)


def _beeswarm_points(model: str, feature: str):
    """Real per-observation SHAP values + normalized feature values for colouring."""
    d = RESULTS["shap"][model]["beeswarm"][feature]
    sv = np.asarray(d["shap"], dtype=float)
    fv = np.asarray(d["feature_value"], dtype=float)
    jitter = _JITTER_RNG.random(len(sv)) * 0.7 - 0.35
    return sv, fv, jitter


def _dependence_points(model: str, variable: str):
    """Real (feature value, SHAP attribution) pairs across test observations."""
    d = RESULTS["shap"][model]["dependence"][variable]
    return np.asarray(d["x"], dtype=float), np.asarray(d["y"], dtype=float)


def local_shap(model: str, week_idx: int):
    """Real per-feature SHAP contributions for a SINGLE predicted week.

    Returns features ordered by absolute impact (ascending), their signed
    contributions (which sum to predicted - baseline), the baseline E[f(x)],
    and the predicted value f(x).
    """
    entry = RESULTS["shap"][model]["local"][week_idx]
    contributions = entry["contributions"]
    feats = list(contributions.keys())
    contrib = np.array([contributions[f] for f in feats], dtype=float)
    base = BASE_VALUE[model]
    predicted = entry["predicted"]

    order = np.argsort(np.abs(contrib))  # ascending: small near base, large near f(x)
    feats = [feats[i] for i in order]
    contrib = contrib[order]
    return feats, contrib, base, predicted


# --------------------------------------------------------------------------- #
# Page config + CSS (card styling)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="TAKAL — Time-Series Analysis & Knowledge Attribution for Local Rice",
    page_icon="🍚",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 4rem; max-width: 1280px; }
      .app-title {
        font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;
        line-height: 1.3; margin: 0; padding: 0.15rem 0; margin-top: 0.4rem;
      }
      .card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1rem 1.25rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 0.5rem;
      }
      .stat {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 0.85rem 1rem; height: 100%;
      }
      .stat-label { font-size: 0.8rem; color: #64748b; font-weight: 600; margin: 0; }
      .stat-value { font-size: 1.6rem; font-weight: 700; margin: 0.2rem 0 0 0; }
      .stat-sub   { font-size: 0.72rem; color: #94a3b8; margin: 0; }
      .badge {
        display: inline-block; padding: 0.1rem 0.55rem; border-radius: 9999px;
        font-size: 0.7rem; font-weight: 600; color: #fff;
      }
      table.cmp { width: 100%; border-collapse: collapse; }
      table.cmp th, table.cmp td { padding: 0.6rem 0.9rem; font-size: 0.85rem; }
      table.cmp th { color: #64748b; text-align: right; border-bottom: 1px solid #e2e8f0; }
      table.cmp th:first-child, table.cmp td:first-child { text-align: left; }
      table.cmp td { text-align: right; font-variant-numeric: tabular-nums; }
      .verdict {
        background: #f0fdf4; border-radius: 10px; padding: 1rem; text-align: center;
      }
      /* Make the Active Model dropdown non-typeable (read-only look) while
         keeping it clickable: block the search input but allow the menu to open. */
      .st-key-active_model div[data-baseweb="select"] input {
        caret-color: transparent;
        pointer-events: none;
        user-select: none;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def stat_card(label, value, sub="", accent=None):
    border = f"border-left:4px solid {accent};" if accent else ""
    st.markdown(
        f"""<div class="stat" style="{border}">
              <p class="stat-label">{label}</p>
              <p class="stat-value">{value}</p>
              <p class="stat-sub">{sub}</p>
            </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Header + global model toggle
# --------------------------------------------------------------------------- #
head_left, head_right = st.columns([3, 2])
with head_left:
    st.markdown(
        '<p class="app-title">Time-Series Analysis &amp; Knowledge Attribution for Local Rice (TAKAL)</p>',
        unsafe_allow_html=True,
    )
with head_right:
    model_label = st.selectbox(
        "Active Model",
        ["Multivariate Model (Economic Indicators)", "Univariate Model (Baseline)"],
        key="active_model",
    )
active = "multivariate" if model_label.startswith("Multivariate") else "univariate"
active_name = "Multivariate" if active == "multivariate" else "Univariate"

st.divider()

tab_forecast, tab_metrics, tab_shap = st.tabs(
    ["📈 Price Forecast", "📊 Benchmarking", "💡 Explainability"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Price Forecast (vs Actual)
# --------------------------------------------------------------------------- #
with tab_forecast:
    st.subheader(f"{N_WEEKS}-Week Forecast vs. Actual")
    st.caption(
        f"Predicted vs. actual observed retail rice prices over the {N_WEEKS}-week held-out "
        f"test horizon (NCR, {WEEKS[0]} → {WEEKS[-1]}). The **{active_name}** model is active; the "
        f"**Actual Price** line is the ground-truth reference."
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
                line=dict(color=CHART_COLORS[key], width=3 if is_active else 1.5),
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
            mode="lines+markers",
            line=dict(color=CHART_COLORS["actual"], width=2.5, dash="dash"),
            hovertemplate="%{y:.2f} ₱/kg<extra>Actual</extra>",
        )
    )
    _allv = forecast_data["actual"] + forecast_data["multivariate"] + forecast_data["univariate"]
    _ylo, _yhi = min(_allv), max(_allv)
    _ypad = max(0.5, (_yhi - _ylo) * 0.12)
    fig.update_yaxes(range=[_ylo - _ypad, _yhi + _ypad], tickprefix="₱", title="Price (PHP/kg)")
    fig.update_layout(
        height=420,
        margin=dict(t=30, r=20, l=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7")
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7")
    st.plotly_chart(fig, width='stretch')

    vals = forecast_data[active]
    avg_gap = float(np.mean(np.abs(np.array(vals) - np.array(forecast_data["actual"]))))
    change = vals[-1] - vals[0]
    pct = change / vals[0] * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Starting Price", f"₱{vals[0]:.2f}", WEEKS[0])
    with c2:
        stat_card("Ending Price", f"₱{vals[-1]:.2f}", WEEKS[-1])
    with c3:
        stat_card("Price Change", f"₱{change:+.2f}", f"{pct:+.1f}% over {N_WEEKS} weeks")
    with c4:
        stat_card("Avg. Gap vs. Actual", f"₱{avg_gap:.2f}",
                  "Mean absolute deviation", accent="#10b981")

# --------------------------------------------------------------------------- #
# Tab 2: Benchmarking
# --------------------------------------------------------------------------- #
with tab_metrics:
    st.subheader("Model Performance Metrics")
    st.caption(f"Regression benchmarking for the **{active_name}** model.")

    m = metrics[active]
    r2_ok = m["r2"] >= 0.9
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("RMSE", f"{m['rmse']}", "Root Mean Square Error", accent="#0ea5e9")
    with c2:
        stat_card("MAE", f"{m['mae']}", "Mean Absolute Error", accent="#10b981")
    with c3:
        stat_card("MAPE", f"{m['mape']}%", "Mean Absolute Percentage Error", accent="#f59e0b")
    with c4:
        stat_card(
            "R² Score", f"{m['r2']}",
            "✓ Statistically Significant" if r2_ok else "Coefficient of Determination",
            accent="#10b981" if r2_ok else "#64748b",
        )

    st.markdown("##### Model Comparison")
    mv, uv = metrics["multivariate"], metrics["univariate"]
    g, r = "color:#16a34a;background:#f0fdf4;", "color:#dc2626;background:#fef2f2;"
    st.markdown(
        f"""
        <table class="cmp">
          <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>R²</th></tr></thead>
          <tbody>
            <tr>
              <td>🟦 Multivariate</td>
              <td style="{g}">{mv['rmse']}</td><td style="{g}">{mv['mae']}</td>
              <td style="{g}">{mv['mape']}%</td><td style="{g}">{mv['r2']}</td>
            </tr>
            <tr>
              <td>⬛ Univariate</td>
              <td style="{r}">{uv['rmse']}</td><td style="{r}">{uv['mae']}</td>
              <td style="{r}">{uv['mape']}%</td><td style="{r}">{uv['r2']}</td>
            </tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Diebold-Mariano Test Result")
    d1, d2, d3 = st.columns(3)
    with d1:
        stat_card("Test Statistic", f"{dm_test['statistic']}")
    with d2:
        stat_card("P-Value", f"{dm_test['p_value']}")
    with d3:
        stat_card("Significance Level", f"α = {dm_test['alpha']}")

    _sig = dm_test.get("significant", dm_test["p_value"] < dm_test["alpha"])
    _better_name = "multivariate" if dm_test.get("better") == "multivariate" else "univariate"
    if _sig:
        _v_head, _v_color, _v_bg = "Reject H₀", "#15803d", "#f0fdf4"
        _v_msg = (f"The {_better_name} model provides statistically significantly "
                  f"superior forecasting accuracy.")
    else:
        _v_head, _v_color, _v_bg = "Fail to Reject H₀", "#92400e", "#fffbeb"
        _v_msg = ("No statistically significant difference in forecasting accuracy "
                  "between the two models.")
    st.markdown(
        f"""
        <div class="verdict" style="background:{_v_bg};">
          <p style="color:{_v_color};font-weight:600;margin:0;">Verdict:</p>
          <p style="color:{_v_color};font-weight:700;font-size:1.2rem;margin:0.3rem 0;">{_v_head}</p>
          <p style="color:{_v_color};font-weight:500;margin:0;">
            {_v_msg}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Tab 3: Explainability (SHAP)
# --------------------------------------------------------------------------- #
with tab_shap:
    st.subheader("Global Feature Importance (SHAP)")
    st.caption(f"SHAP summary of feature contributions for the **{active_name}** model.")

    plot_type = st.radio("Plot type", ["Bar", "Beeswarm"], horizontal=True,
                         label_visibility="collapsed")

    features = [f for f, _ in shap_data[active]]
    values = [v for _, v in shap_data[active]]

    if plot_type == "Bar":
        fig_bar = go.Figure(
            go.Bar(
                x=values[::-1],
                y=features[::-1],
                orientation="h",
                marker_color=[BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(features))][::-1],
                hovertemplate="%{x:.3f}<extra>%{y}</extra>",
            )
        )
        fig_bar.update_layout(
            height=400, margin=dict(t=20, r=40, l=10, b=20),
            plot_bgcolor="white", xaxis_title="mean(|SHAP value|)",
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor="#eef2f7", range=[0, max(values) * 1.15])
        st.plotly_chart(fig_bar, width='stretch')
    else:
        fig_bee = go.Figure()
        for idx, feat in enumerate(features):
            sv, fv, jit = _beeswarm_points(active, feat)
            fig_bee.add_trace(
                go.Scatter(
                    x=sv,
                    y=np.full_like(sv, idx) + jit,
                    mode="markers",
                    marker=dict(
                        color=fv, colorscale=[[0, "#3b82f6"], [0.5, "#a855f7"], [1, "#ec4899"]],
                        cmin=0, cmax=1, size=6, opacity=0.7,
                        colorbar=dict(title="Feature<br>value", tickvals=[0, 1],
                                      ticktext=["Low", "High"]) if idx == 0 else None,
                        showscale=idx == 0,
                    ),
                    showlegend=False,
                    hovertemplate="SHAP %{x:.3f}<extra>" + feat + "</extra>",
                )
            )
        fig_bee.add_vline(x=0, line_color="#cbd5e1", line_width=2)
        fig_bee.update_yaxes(tickvals=list(range(len(features))), ticktext=features,
                             autorange="reversed")
        fig_bee.update_xaxes(title="SHAP value (impact on model output)",
                             showgrid=True, gridcolor="#eef2f7")
        fig_bee.update_layout(height=480, margin=dict(t=20, r=20, l=10, b=40),
                              plot_bgcolor="white")
        st.plotly_chart(fig_bee, width='stretch')

    st.divider()
    st.subheader("Local Explanation — Why this week's price?")
    st.caption(
        "Per-prediction SHAP waterfall: how each indicator pushes a **specific week's** "
        "forecast up or down from the baseline E[f(x)] to the final predicted price f(x)."
    )

    sel_week = st.selectbox("Select forecast week", WEEKS, index=len(WEEKS) - 1)
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
            connector=dict(line=dict(color="#cbd5e1")),
            increasing=dict(marker=dict(color="#f59e0b")),  # pushes price UP
            decreasing=dict(marker=dict(color="#3b82f6")),  # pushes price DOWN
            hovertemplate="%{y}: %{x:+.3f} ₱/kg<extra></extra>",
        )
    )
    fig_wf.add_vline(x=base_v, line_dash="dash", line_color="#94a3b8",
                     annotation_text=f"E[f(x)] = ₱{base_v:.2f}", annotation_position="top")
    fig_wf.add_vline(x=pred_v, line_dash="dash", line_color="#16a34a",
                     annotation_text=f"f(x) = ₱{pred_v:.2f}", annotation_position="top")
    pad = max(0.4, abs(pred_v - base_v) * 0.6)
    fig_wf.update_xaxes(title="Predicted price (PHP/kg)", showgrid=True, gridcolor="#eef2f7",
                        range=[min(base_v, pred_v) - pad, max(base_v, pred_v) + pad])
    fig_wf.update_layout(height=420, margin=dict(t=40, r=30, l=10, b=40), plot_bgcolor="white")
    st.plotly_chart(fig_wf, width='stretch')

    move = pred_v - base_v
    top_feat = feats_l[int(np.argmax(np.abs(contrib_l)))]
    top_val = contrib_l[int(np.argmax(np.abs(contrib_l)))]
    direction = "increase" if move >= 0 else "decrease"
    st.markdown(
        f"**{sel_week}:** the {active_name} model predicts **₱{pred_v:.2f}/kg**, a "
        f"**₱{move:+.2f}** {direction} from the baseline. The largest single driver is "
        f"**{top_feat}** ({top_val:+.2f} ₱/kg). "
        f"<span style='color:#f59e0b'>■</span> raises price · "
        f"<span style='color:#3b82f6'>■</span> lowers price.",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Feature Dependence Analysis")
    st.caption("How an individual variable drives model attribution across test observations.")

    dep_var = st.selectbox("Select Variable", [f for f, _ in shap_data[active]])
    dx, dy = _dependence_points(active, dep_var)
    colors = ["#f59e0b" if v > 0 else "#3b82f6" for v in dy]
    fig_dep = go.Figure(
        go.Scatter(
            x=dx, y=dy, mode="markers",
            marker=dict(color=colors, opacity=0.6, size=7),
            hovertemplate=f"{dep_var}: " + "%{x:.1f}<br>SHAP: %{y:.3f}<extra></extra>",
        )
    )
    fig_dep.add_hline(y=0, line_dash="dash", line_color="#cbd5e1",
                      annotation_text="Zero Attribution", annotation_position="right")
    fig_dep.update_xaxes(title=dep_var, showgrid=True, gridcolor="#eef2f7")
    fig_dep.update_yaxes(title="SHAP Attribution Value", showgrid=True, gridcolor="#eef2f7")
    fig_dep.update_layout(height=400, margin=dict(t=20, r=20, l=10, b=40), plot_bgcolor="white")
    st.plotly_chart(fig_dep, width='stretch')
    st.markdown(f"**How {dep_var} Drives Forecast Attribution**")

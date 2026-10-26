"""12-week forward forecast: trajectory chart, future SHAP waterfall, plain-language reading.

Reads the ``future_forecast`` block written by ``model/predict_future.py``. The
history shown on the chart comes from the observed prices already stored in
``results.json`` (the held-out test weeks end at the last observed week), so the
dashboard never needs the Excel dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import chart_layout, render_chart, render_reading
from interpretation import future_reading
from utils import COLORS, format_week_date, future_feature_label

HISTORY_MONTHS = 6

# Same palette as the Overview forecast chart so the two read as one family.
OBSERVED_COLOR = "#000000"
MULTIVARIATE_COLOR = "#1BB385"
UNIVARIATE_COLOR = "#6E998B"


# --------------------------------------------------------------------------- #
# Data helpers (pure functions, no Streamlit state)
# --------------------------------------------------------------------------- #
def history_window(
    weeks: list[str], forecast: dict, origin_date: str, months: int = HISTORY_MONTHS
) -> tuple[list[str], list[float]]:
    """Observed prices for the ``months`` up to and including the origin week."""
    actual = forecast["1"]["actual"]
    origin = pd.Timestamp(origin_date)
    start = origin - pd.DateOffset(months=months)
    pairs = [
        (week, float(value))
        for week, value in zip(weeks, actual)
        if value is not None and np.isfinite(float(value)) and start <= pd.Timestamp(week) <= origin
    ]
    if not pairs:  # origin outside the stored weeks: fall back to the latest 6 months' worth
        pairs = [
            (w, float(v))
            for w, v in zip(weeks, actual)
            if v is not None and np.isfinite(float(v))
        ][-26:]
    return [week for week, _ in pairs], [value for _, value in pairs]


def labelled_contributions(future: dict) -> list[dict[str, float]]:
    """Per-week multivariate SHAP contributions keyed by dashboard label."""
    attributions = future["shap_attributions"]["multivariate"]
    return [
        {future_feature_label(column): float(value) for column, value in attributions[date].items()}
        for date in future["horizon_dates"]
    ]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def build_trajectory_figure(
    hist_weeks: list[str], hist_actual: list[float], future: dict, active_model: str
) -> go.Figure:
    """Observed history joined to the two dashed future trajectories."""
    dates = future["horizon_dates"]
    mv = future["multivariate_predictions"]
    uni = future["univariate_predictions"]
    anchor_x, anchor_y = hist_weeks[-1], hist_actual[-1]

    figure = go.Figure()
    figure.add_shape(  # shaded forecast region
        type="rect", xref="x", yref="paper", x0=anchor_x, x1=dates[-1], y0=0, y1=1,
        fillcolor="rgba(37, 201, 152, 0.06)", line={"width": 0}, layer="below",
    )
    figure.add_shape(  # divider at the last observed week
        type="line", xref="x", yref="paper", x0=anchor_x, x1=anchor_x, y0=0, y1=1,
        line={"color": COLORS["muted"], "width": 1, "dash": "dot"}, layer="below",
    )
    for text, anchor, shift in (("Observed", "right", -8), ("12-week forecast", "left", 8)):
        figure.add_annotation(
            x=anchor_x, y=1, xref="x", yref="paper", text=text, showarrow=False,
            xanchor=anchor, yanchor="top", xshift=shift,
            font={"size": 12, "color": COLORS["muted"]},
        )

    figure.add_trace(go.Scatter(
        x=hist_weeks, y=hist_actual, name="Observed price", mode="lines",
        line={"color": OBSERVED_COLOR, "width": 2.4},
        hovertemplate="%{y:.2f} ₱/kg<extra>Observed price</extra>",
    ))

    series = (
        ("univariate", "Univariate baseline (forecast)", uni, UNIVARIATE_COLOR, "dashdot"),
        ("multivariate", "Multivariate forecast", mv, MULTIVARIATE_COLOR, "dash"),
    )
    for model, name, values, color, dash in series:
        is_active = model == active_model
        line = {"color": color, "width": 2.8 if is_active else 2.0, "dash": dash}
        figure.add_trace(go.Scatter(  # connector: last observed point -> first forecast week
            x=[anchor_x, dates[0]], y=[anchor_y, values[0]], mode="lines",
            line=line, hoverinfo="skip", showlegend=False,
        ))
        figure.add_trace(go.Scatter(
            x=dates, y=values, name=name,
            mode="lines+markers" if is_active else "lines",
            marker={"size": 5, "color": color},
            line=line,
            hovertemplate="%{y:.2f} ₱/kg<extra>" + name + "</extra>",
        ))

    prices = np.concatenate([np.asarray(hist_actual, dtype=float), np.asarray(mv, dtype=float), np.asarray(uni, dtype=float)])
    finite_prices = prices[np.isfinite(prices)]
    min_price, max_price = (float(np.nanmin(finite_prices)), float(np.nanmax(finite_prices))) if len(finite_prices) else (0.0, 100.0)
    padding = max(0.5, (max_price - min_price) * 0.15)
    chart_layout(figure, 440)
    figure.update_layout(
        hovermode="x unified",
        legend={
            "orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "xanchor": "left",
            "font": {"size": 12, "color": "#0B221B"},
        },
        margin={"t": 80, "r": 15, "b": 30, "l": 10},
    )
    figure.update_yaxes(
        title_text="Retail price (PHP / kg)", tickprefix="₱",
        range=[min_price - padding, max_price + padding],
    )
    figure.update_xaxes(title_text="Week", tickangle=-30, hoverformat="%b %d, %Y")
    return figure


def build_waterfall_figure(
    week: str, baseline: float, predicted: float, contributions: dict[str, float]
) -> go.Figure:
    """Horizontal SHAP waterfall for one future week (same look as the test-week waterfall)."""
    items = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    figure = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(items),
            y=[name for name, _ in items],
            x=[float(value) for _, value in items],
            base=baseline,
            text=[f"{float(value):+.2f}" for _, value in items],
            textposition="outside",
            cliponaxis=False,
            textfont={"size": 13, "color": "#2F4A41"},
            connector={"line": {"color": COLORS["border"], "width": 1}},
            increasing={"marker": {"color": COLORS["actual"]}},
            decreasing={"marker": {"color": COLORS["multivariate"]}},
            hovertemplate="%{y}: %{x:+.3f} PHP/kg<extra></extra>",
        )
    )
    figure.add_vline(x=baseline, line_dash="dash", line_color=COLORS["univariate"], layer="below")
    figure.add_vline(x=predicted, line_color=COLORS["multivariate"], layer="below")
    for line_x, text, color in (
        (baseline, "Baseline", COLORS["univariate"]),
        (predicted, "Forecast", COLORS["multivariate"]),
    ):
        figure.add_annotation(
            x=line_x, y=1.0, xref="x", yref="paper", text=f"<b>{text}</b>", showarrow=False,
            yanchor="bottom",
            xanchor="right" if line_x <= min(baseline, predicted) and baseline != predicted else "left",
            font={"size": 13, "color": color},
        )

    running = baseline + np.cumsum([float(value) for _, value in items])
    reach = [baseline, predicted, *running.tolist()]
    figure.update_xaxes(
        title_text="Running forecast estimate (PHP per kg)",
        range=[min(reach) - 0.75, max(reach) + 0.75],
    )
    chart_layout(figure, max(340, 50 * len(items)))
    figure.update_layout(margin={"t": 60, "r": 25, "b": 30, "l": 20})
    return figure


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render(results: dict, active_model: str) -> None:
    future = results.get("future_forecast")
    if not future:
        st.info(
            "This results file has no 12-week forward forecast yet. Run "
            "`python -m model.run_all` to regenerate `artifacts/results.json`."
        )
        return

    dates = future["horizon_dates"]
    origin = future["origin_date"]
    hist_weeks, hist_actual = history_window(results["weeks"], results["forecast"], origin)
    note = ((results.get("meta") or {}).get("future_forecast_note") or "").strip()

    st.markdown(
        f'<div class="page-intro">Out-of-sample <strong>12-week forward forecast</strong> of '
        f"<strong>Regular-Milled</strong> retail rice prices in the <strong>National Capital Region "
        f"(NCR)</strong>, projected from the last observed week "
        f"({format_week_date(origin)}) to the week of {format_week_date(dates[-1])}.</div>",
        unsafe_allow_html=True,
    )

    # ---- Trajectory chart ------------------------------------------------- #
    st.markdown('<div class="section-subheading">Projected price trajectory</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-caption">Solid black line: observed prices for the last {HISTORY_MONTHS} '
        "months. Dashed lines: each model's projection for the next 12 weeks.</div>",
        unsafe_allow_html=True,
    )
    with st.container(key="chart_card_future"):
        render_chart(
            st.empty(),
            build_trajectory_figure(hist_weeks, hist_actual, future, active_model),
            "future_trajectory_chart",
        )
    if note:
        with st.expander("Inputs outside the training range", expanded=False):
            st.markdown(
                f"{note}\n\nThe scalers are fitted on training weeks only, so these inputs reach the "
                "model as values outside the range it learned from. Treat the multivariate "
                "projection as indicative."
            )

    # ---- Plain-language reading ------------------------------------------- #
    contributions = labelled_contributions(future)
    render_reading(
        "Reading the 12-week outlook",
        future_reading(
            origin,
            hist_actual[-1],
            dates,
            {
                "multivariate": future["multivariate_predictions"],
                "univariate": future["univariate_predictions"],
            },
            active_model,
            contributions,
            float(future["base_value"]),
            has_range_warning=bool(note),
        ),
    )

    # ---- Future SHAP waterfall -------------------------------------------- #
    st.markdown(
        '<hr style="margin-bottom: 0.25rem; margin-top: 1rem; border: 0; border-top: 1px solid var(--takal-border);">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subheading" style="margin-top: 0.5rem;">'
        "What drives the forecast - Single-Week SHAP Waterfall (Multivariate model)</div>",
        unsafe_allow_html=True,
    )

    options = list(range(len(dates)))
    if st.session_state.get("future_shap_week") not in options:
        st.session_state.pop("future_shap_week", None)
    selected = st.selectbox(
        "Future forecast week",
        options,
        index=0,
        format_func=lambda i: f"Week +{i + 1} · {format_week_date(dates[i])}",
        key="future_shap_week",
    )
    week = dates[selected]
    baseline = float(future["base_value"])
    predicted = float(future["multivariate_predictions"][selected])
    week_contributions = contributions[selected]
    ranked = sorted(week_contributions.items(), key=lambda item: abs(item[1]), reverse=True)

    top_name, top_value = ranked[0]
    st.markdown(
        f"For **Week +{selected + 1} ({format_week_date(week)})** the calculation starts at the "
        f"model's baseline of **₱{baseline:.2f}/kg** and adds each indicator's contribution. "
        f"**{top_name}** has the largest effect: it {'raises' if top_value >= 0 else 'lowers'} the "
        f"estimate by **₱{abs(top_value):.2f}/kg**. After all contributions the multivariate forecast "
        f"is **₱{predicted:.2f}/kg**."
    )
    render_chart(
        st.empty(),
        build_waterfall_figure(week, baseline, predicted, week_contributions),
        "future_shap_waterfall",
    )
    st.markdown(
        '<div class="section-caption">The baseline is the multivariate model\'s average forecast over '
        "training windows. SHAP contributions are additive, so the baseline plus all contributions "
        "equals the forecast for that week. They describe how the model uses its inputs, not what "
        "causes price changes.</div>",
        unsafe_allow_html=True,
    )
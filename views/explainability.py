"""SHAP explainability screens for global and individual forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import chart_layout, render_chart, render_table
from utils import COLORS, beeswarm_points, dependence_points, feature_label, format_week_date


def render(shap_results: dict, weeks: list[str], active_model: str) -> None:
    shap_model = shap_results[active_model]
    feature_rows = list(zip(shap_model["features"], shap_model["bar"]))

    feature_names = [feature for feature, _ in feature_rows]
    importance_values = [float(value) for _, value in feature_rows]
    feature_labels = {name: feature_label(name) for name in feature_names}

    st.markdown(
        '<div class="page-intro">These explanations describe how the model used its input indicators. '
        'They explain the model’s calculations; they do not prove that an indicator caused a market-price change.</div>',
        unsafe_allow_html=True,
    )

    importance_tab, beeswarm_tab = st.tabs([
        "Global Importance (SHAP Summary Bar)",
        "Directional Effects (SHAP Beeswarm)",
    ])

    with importance_tab:
        st.write("A longer bar means that an indicator moved the model’s predictions more, on average, across the test period.")
        importance_figure = go.Figure(
            go.Bar(
                x=importance_values[::-1],
                y=[feature_labels[name] for name in feature_names][::-1],
                orientation="h",
                marker={"color": COLORS["multivariate"] if active_model == "multivariate" else COLORS["univariate"]},
                hovertemplate="Average impact: %{x:.3f} PHP/kg<extra>%{y}</extra>",
            )
        )
        importance_figure.update_layout(xaxis_title="Average absolute SHAP value (PHP per kg)")
        importance_figure.update_xaxes(range=[0, max(importance_values or [0.0]) * 1.15])
        chart_layout(importance_figure, max(330, 54 * len(feature_names)))
        render_chart(st.empty(), importance_figure, "shap_importance")

    with beeswarm_tab:
        st.write("Each dot represents one test week. Its horizontal position shows the direction and size of that indicator’s contribution for that week.")
        beeswarm_figure = go.Figure()

        for feature_index, feature_name in enumerate(feature_names):
            point_data = shap_model["beeswarm"][feature_name]
            shap_values, feature_values, jitter = beeswarm_points(
                active_model,
                feature_name,
                tuple(point_data["shap"]),
                tuple(point_data["feature_value"]),
            )

            beeswarm_figure.add_trace(
                go.Scatter(
                    x=shap_values,
                    y=np.full_like(shap_values, feature_index, dtype=float) + jitter,
                    mode="markers",
                    marker={
                        "color": feature_values,
                        "colorscale": [
                            [0, COLORS["univariate"]],
                            [0.5, COLORS["actual"]],
                            [1, COLORS["multivariate"]],
                        ],
                        "cmin": 0,
                        "cmax": 1,
                        "size": 7,
                        "opacity": 0.82,
                        "showscale": feature_index == 0,
                        "colorbar": {
                            "title": "Input value",
                            "tickvals": [0, 1],
                            "ticktext": ["Low", "High"],
                        },
                    },
                    showlegend=False,
                    hovertemplate="Contribution: %{x:+.3f} PHP/kg<extra>" + feature_labels[feature_name] + "</extra>",
                )
            )

        beeswarm_figure.add_vline(x=0, line_color=COLORS["muted"], line_width=1)
        beeswarm_figure.update_yaxes(
            tickvals=list(range(len(feature_names))),
            ticktext=[feature_labels[name] for name in feature_names],
            autorange="reversed",
        )
        beeswarm_figure.update_xaxes(title_text="Contribution to predicted price (PHP per kg)")
        chart_layout(beeswarm_figure, max(360, 58 * len(feature_names)))
        render_chart(st.empty(), beeswarm_figure, "shap_beeswarm")

    st.markdown('<hr style="margin-bottom: 0.25rem; margin-top: 1rem; border: 0; border-top: 1px solid var(--takal-border);">', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading" style="margin-top: 0.5rem;">Local Feature Attribution — Single-Week SHAP Waterfall</div>', unsafe_allow_html=True)

    selected_week = st.selectbox(
        "Forecast week",
        weeks,
        index=len(weeks) - 1,
        format_func=format_week_date,
        key="local_explanation_week",
    )
    week_index = weeks.index(selected_week)
    local_entry = shap_model["local"][week_index]

    baseline = float(shap_model["base_value"])
    predicted_price = float(local_entry["predicted"])
    contributions = local_entry["contributions"]

    sorted_contributions = sorted(contributions.items(), key=lambda item: abs(float(item[1])), reverse=True)
    week_label = lambda name: feature_label(name, selected_week)

    running_price = baseline
    explanation_rows = []
    for feature_name, contribution_value in sorted_contributions:
        contribution_value = float(contribution_value)
        running_price += contribution_value
        explanation_rows.append(
            {
                "Step": len(explanation_rows) + 1,
                "Indicator": week_label(feature_name),
                "Contribution (PHP/kg)": contribution_value,
                "Effect": "Raises the estimate" if contribution_value > 0 else "Lowers the estimate" if contribution_value < 0 else "No change",
                "Estimate after this step (PHP/kg)": running_price,
            }
        )

    if sorted_contributions:
        largest_feature, largest_value = sorted_contributions[0]
        largest_value = float(largest_value)
        largest_effect = "raised" if largest_value >= 0 else "lowered"
        largest_summary = (
            f"For **{format_week_date(selected_week)}**, **{week_label(largest_feature)}** had the largest individual "
            f"effect: it {largest_effect} the estimate by **₱{abs(largest_value):.2f}/kg**. "
        )
    else:
        largest_summary = "No indicator contributions were recorded for this week. "

    st.markdown(
        f"The calculation starts at the model’s baseline of **₱{baseline:.2f}/kg**. It then adds each "
        f"indicator’s contribution below. A positive contribution raises the running estimate; "
        f"a negative contribution lowers it. {largest_summary}After all listed contributions, the model "
        f"forecast is **₱{predicted_price:.2f}/kg**."
    )

    # Graph rendered FIRST before the table
    local_figure = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(sorted_contributions),
            y=[week_label(feature) for feature, _ in sorted_contributions],
            x=[float(value) for _, value in sorted_contributions],
            base=baseline,
            text=[f"{float(value):+.2f}" for _, value in sorted_contributions],
            textposition="outside",
            cliponaxis=False,
            textfont={"size": 13, "color": "#2F4A41"},
            connector={"line": {"color": COLORS["border"], "width": 1}},
            increasing={"marker": {"color": COLORS["actual"]}}, # Green
            decreasing={"marker": {"color": COLORS["multivariate"]}}, # Blue
            hovertemplate="%{y}: %{x:+.3f} PHP/kg<extra></extra>",
        )
    )

    # Reference lines sit behind the bars; their labels live above the plot area
    # (paper coordinates) so they never overlap a bar or a bar's value label.
    local_figure.add_vline(x=baseline, line_dash="dash", line_color=COLORS["univariate"], layer="below")
    local_figure.add_vline(x=predicted_price, line_color=COLORS["multivariate"], layer="below")
    for line_x, line_text, line_color in (
        (baseline, "Baseline", COLORS["univariate"]),
        (predicted_price, "Forecast", COLORS["multivariate"]),
    ):
        local_figure.add_annotation(
            x=line_x,
            y=1.0,
            xref="x",
            yref="paper",
            text=f"<b>{line_text}</b>",
            showarrow=False,
            yanchor="bottom",
            # Put the lower value's label to the left of its line and the higher value's to the right
            xanchor="right" if line_x <= min(baseline, predicted_price) and baseline != predicted_price else "left",
            font={"size": 13, "color": line_color},
        )

    # Dynamic padding to prevent label overlap
    x_values = [baseline, predicted_price] + [baseline + sum([float(v) for _, v in sorted_contributions[:i+1]]) for i in range(len(sorted_contributions))]
    min_x = min(x_values) - 0.75
    max_x = max(x_values) + 0.75
    
    local_figure.update_xaxes(title_text="Running forecast estimate (PHP per kg)", range=[min_x, max_x])
    
    chart_layout(local_figure, max(340, 50 * len(sorted_contributions)))
    local_figure.update_layout(margin={"t": 60, "r": 25, "b": 30, "l": 20})
    render_chart(st.empty(), local_figure, "shap_local_waterfall")

    # Table rendered BELOW the graph
    render_table(
        pd.DataFrame(explanation_rows),
        formats={
            "Contribution (PHP/kg)": "{:+.3f}".format,
            "Estimate after this step (PHP/kg)": "{:.3f}".format,
        },
        label_columns=["Indicator"],
        effect_column="Effect",
    )

    st.markdown('<hr style="margin-bottom: 0.25rem; margin-top: 1rem; border: 0; border-top: 1px solid var(--takal-border);">', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading" style="margin-top: 0.5rem;">SHAP Dependence Plot — Indicator Value vs. Price Contribution</div>', unsafe_allow_html=True)

    selected_feature = st.selectbox(
        "Indicator",
        feature_names,
        format_func=lambda name: feature_labels[name],
        key="dependence_feature",
    )
    selected_label = feature_labels[selected_feature]

    dependence_data = shap_model["dependence"][selected_feature]
    x_points, y_points = dependence_points(tuple(dependence_data["x"]), tuple(dependence_data["y"]))

    point_colors = [COLORS["actual"] if value >= 0 else COLORS["multivariate"] for value in y_points]

    dependence_figure = go.Figure(
        go.Scatter(
            x=x_points,
            y=y_points,
            mode="markers",
            marker={
                "color": point_colors,
                "opacity": 0.78,
                "size": 8,
                "line": {"width": 0.5, "color": COLORS["border"]},
            },
            hovertemplate=f"{selected_label}: "
            + "%{x:.3f}<br>Contribution: %{y:+.3f} PHP/kg<extra></extra>",
        )
    )

    dependence_figure.add_hline(y=0, line_color=COLORS["univariate"], line_dash="dash")
    dependence_figure.update_xaxes(title_text=selected_label)
    dependence_figure.update_yaxes(title_text="Contribution to forecast (PHP per kg)")

    chart_layout(dependence_figure, 390)
    render_chart(st.empty(), dependence_figure, "shap_dependence")
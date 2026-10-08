"""Benchmark metrics and paired forecast-error comparison screen."""

import pandas as pd
import streamlit as st

from components import render_table


def render(metrics: dict, dm_test: dict, week_count: int, active_model: str) -> None:
    st.markdown(
        '<div class="page-intro">These measures compare forecast errors on the same held-out weeks. '
        'Lower error values mean predictions were closer to the observed prices.</div>',
        unsafe_allow_html=True,
    )
    current_metrics = metrics[active_model]
    metric_columns = st.columns(4)
    metric_descriptions = {
        "rmse": ("RMSE", "Penalizes larger misses more strongly"),
        "mae": ("MAE", "Average absolute miss, in PHP per kg"),
        "mape": ("MAPE", "Average absolute miss as a percentage"),
        "r2": ("R² score", "How much of the observed variation is captured"),
    }
    for column, metric_key in zip(metric_columns, metric_descriptions):
        title, description = metric_descriptions[metric_key]
        suffix = "%" if metric_key == "mape" else ""
        with column:
            st.markdown(
                f'<div class="technical-card"><div class="technical-card-label">{title}</div>'
                f'<div class="technical-card-value">{current_metrics[metric_key]}{suffix}</div>'
                f'<div class="technical-card-detail">{description}</div></div>',
                unsafe_allow_html=True,
            )

    st.subheader("Side-by-side results")
    comparison = pd.DataFrame(
        [
            {"Model": "Multivariate model", **metrics["multivariate"]},
            {"Model": "Univariate baseline", **metrics["univariate"]},
        ]
    ).rename(columns={"rmse": "RMSE", "mae": "MAE", "mape": "MAPE (%)", "r2": "R²"})
    render_table(
        comparison,
        formats={column: "{:.4f}".format for column in comparison.columns if column != "Model"},
        label_columns=["Model"],
    )

    st.subheader("Statistical Significance — Diebold-Mariano Test")
    statistic = float(dm_test.get("statistic", 0.0))
    p_value = float(dm_test.get("p_value", 1.0))
    alpha = float(dm_test.get("alpha", 0.05))
    sample_size = int(dm_test.get("n", week_count))
    statistically_different = bool(dm_test.get("significant", p_value < alpha))
    better_key = dm_test.get("better", "multivariate")
    better_name = (
        "Multivariate Stacked LSTM model"
        if better_key == "multivariate"
        else "Univariate Stacked LSTM baseline"
    )

    dm_columns = st.columns(3)
    for column, label, value in zip(
        dm_columns,
        (
            "DM test statistic (HLN-corrected)",
            "Two-tailed p-value",
            'Significance level (<span class="no-upper">α</span>)',
        ),
        (f"{statistic:.4f}", f"{p_value:.6f}", f"{alpha:g}"),
    ):
        with column:
            st.markdown(
                f'<div class="technical-card"><div class="technical-card-label">{label}</div>'
                f'<div class="technical-card-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    if statistically_different:
        dm_title = (
            f"Statistically Significant Difference Found — Null Hypothesis (H₀) Rejected "
            f"(p = {p_value:.4f} < {alpha:g})"
        )
        dm_text = (
            f"Across all {sample_size} paired hold-out test weeks, the Diebold–Mariano test confirms that the "
            f"forecasting error reduction achieved by the <strong>{better_name}</strong> is statistically significant "
            f"at the α = {alpha:g} level (DM = {statistic:.4f}, p = {p_value:.4f}). Combined with lower RMSE, MAE, "
            f"and MAPE values and a higher R², the null hypothesis (H₀) is formally rejected in favor of the "
            f"alternative hypothesis."
        )
    else:
        dm_title = f"No Statistically Significant Difference Found — Null Hypothesis (H₀) Retained (p = {p_value:.4f})"
        dm_text = (
            f"Across {sample_size} paired test weeks, the Diebold–Mariano test did not establish a statistically "
            f"significant difference at the α = {alpha:g} threshold (p = {p_value:.4f})."
        )

    st.markdown(
        f'<div class="result-panel"><strong>{dm_title}</strong><br>{dm_text}<br><br>'
        '<span class="metric-detail">Why this matters: two models can have different average error scores '
        'simply because of which weeks happened to be in the test set. The Diebold–Mariano test compares '
        'their paired errors week by week to assess whether the difference is consistent enough to be '
        'unlikely under a no-difference assumption.</span></div>',
        unsafe_allow_html=True,
    )
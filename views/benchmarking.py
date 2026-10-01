"""Benchmark metrics and paired forecast-error comparison screen."""

import pandas as pd
import streamlit as st


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
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.subheader("Are the forecast differences likely to be meaningful?")
    statistic = float(dm_test.get("statistic", 0.0))
    p_value = float(dm_test.get("p_value", 1.0))
    alpha = float(dm_test.get("alpha", 0.05))
    sample_size = int(dm_test.get("n", week_count))
    statistically_different = bool(dm_test.get("significant", p_value < alpha))
    better_key = dm_test.get("better", "multivariate")
    better_name = "multivariate model" if better_key == "multivariate" else "univariate baseline"
    dm_columns = st.columns(3)
    for column, label, value in zip(
        dm_columns,
        ("Test statistic", "P-value", "Significance threshold"),
        (f"{statistic:.4f}", f"{p_value:.6f}", f"{alpha:g}"),
    ):
        with column:
            st.markdown(
                f'<div class="technical-card"><div class="technical-card-label">{label}</div>'
                f'<div class="technical-card-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    if statistically_different:
        dm_text = (
            f"Across {sample_size} matching test weeks, this test found evidence that the models' forecast "
            f"errors differ beyond what would usually be expected from random week-to-week variation "
            f"(p = {p_value:.4f}, threshold = {alpha:g}). Under the loss comparison used by the pipeline, "
            f"the {better_name} had the lower forecast loss. This is evidence about these test weeks, "
            "not a guarantee that it will always be better in future periods."
        )
        dm_title = "A measurable difference was found"
    else:
        dm_text = (
            f"Across {sample_size} matching test weeks, the test did not find strong enough evidence to "
            f"separate the models' forecast errors from ordinary variation (p = {p_value:.4f}, "
            f"threshold = {alpha:g}). The observed score difference alone is not enough to conclude that "
            "either model is reliably more accurate; more evaluation data could change the result."
        )
        dm_title = "No reliable difference was established"
    st.markdown(
        f'<div class="result-panel"><strong>{dm_title}</strong><br>{dm_text}<br><br>'
        '<span class="metric-detail">Why this matters: two models can have different average error scores '
        'simply because of which weeks happened to be in the test set. The Diebold–Mariano test compares '
        'their paired errors week by week to assess whether the difference is consistent enough to be '
        'unlikely under a no-difference assumption.</span></div>',
        unsafe_allow_html=True,
    )
"""Horizon-wise regression metrics and paired forecast-error tests."""

import pandas as pd
import streamlit as st

from components import render_table


def render(metrics: dict, dm_test: dict, week_count: int, active_model: str) -> None:
    model_label = (
        "Multivariate mode"
        if active_model == "multivariate"
        else "Univariate mode"
    )
    st.markdown(
        '<div class="page-intro">These measures compare forecast errors on the same held-out weeks. '
        'Lower error values mean predictions were closer to the observed prices.</div>',
        unsafe_allow_html=True,
    )

    st.subheader(f"Forecast Accuracy by Horizon ({model_label})")
    metric_rows = []
    for horizon in [*(str(value) for value in range(1, 13)), "overall"]:
        row = {"Horizon": "Overall" if horizon == "overall" else f"Week {horizon}"}
        values = metrics[horizon][active_model]
        row.update({
            "RMSE": values["rmse"],
            "MAE": values["mae"],
            "MAPE (%)": values["mape"],
            "R²": values["r2"],
        })
        metric_rows.append(row)
    comparison = pd.DataFrame(metric_rows)
    render_table(
        comparison,
        formats={
            column: "{:.4f}".format
            for column in comparison.columns
            if column != "Horizon"
        },
        label_columns=["Horizon"],
    )

    st.subheader("Diebold-Mariano Tests by Horizon")
    dm_rows = []
    for horizon in [*(str(value) for value in range(1, 13)), "overall"]:
        result = dm_test[horizon]
        dm_rows.append({
            "Horizon": "Overall" if horizon == "overall" else f"Week {horizon}",
            "DM statistic": result["statistic"],
            "Two-tailed p-value": result["p_value"],
            "Alpha": result["alpha"],
            "Decision": result["decision"],
        })
    render_table(
        pd.DataFrame(dm_rows),
        formats={
            "DM statistic": "{:.4f}".format,
            "Two-tailed p-value": "{:.6f}".format,
            "Alpha": "{:.3f}".format,
        },
        label_columns=["Horizon", "Decision"],
    )

    overall = dm_test["overall"]
    statistic = float(overall["statistic"])
    p_value = float(overall["p_value"])
    alpha = float(overall["alpha"])
    sample_size = int(overall.get("n", week_count))
    statistically_different = bool(overall.get("significant", p_value < alpha))
    better_name = (
        "Multivariate Stacked LSTM model"
        if overall.get("better") == "multivariate"
        else "Univariate Stacked LSTM baseline"
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
        dm_title = (
            f"No Statistically Significant Difference Found — Null Hypothesis (H₀) Retained "
            f"(p = {p_value:.4f})"
        )
        dm_text = (
            f"Across {sample_size} paired test weeks, the Diebold–Mariano test did not establish a statistically "
            f"significant difference at the α = {alpha:g} threshold (p = {p_value:.4f})."
        )

    st.markdown(
        f'<div class="result-panel"><strong>{dm_title}</strong><br>{dm_text}<br><br>'
        '<span class="metric-detail">The overall result compares each week’s average squared-error loss '
        'differential across all 12 forecast horizons.</span></div>',
        unsafe_allow_html=True,
    )

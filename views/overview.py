"""Overview screen with price summaries and a paginated weekly forecast table."""

from __future__ import annotations

import html
import math
from datetime import date, datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components import chart_layout, metric_card, render_chart
from utils import COLORS, format_week_date


ROWS_PER_PAGE_OPTIONS = (10, 25, 50, 100)
DATE_ORDER_OPTIONS = ("Latest first", "Oldest first")


def _reset_week_page() -> None:
    st.session_state["overview_page"] = 0


def _from_date_changed() -> None:
    from_date = st.session_state["overview_from_date"]
    if from_date > st.session_state["overview_to_date"]:
        st.session_state["overview_to_date"] = from_date
    _reset_week_page()


def _to_date_changed() -> None:
    to_date = st.session_state["overview_to_date"]
    if to_date < st.session_state["overview_from_date"]:
        st.session_state["overview_from_date"] = to_date
    _reset_week_page()


def _money(value: float) -> str:
    if not np.isfinite(value):
        return "Not available"
    return f"₱{value:.2f}"


def _render_week_table(weeks: list[str], forecast: dict, active_model: str) -> None:
    records = list(zip(weeks, forecast[active_model], forecast["actual"]))
    records.sort(key=lambda record: record[0], reverse=True)
    available_dates = {
        week: datetime.strptime(week, "%Y-%m-%d").date()
        for week, _, _ in records
    }
    earliest_data_date = min(available_dates.values(), default=date(2025, 1, 1))
    latest_data_date = max(available_dates.values(), default=date(2026, 4, 30))
    today = date.today()
    current_week = (today - timedelta(days=today.weekday())).isoformat()

    st.subheader("Every week")
    if not any(week == current_week for week, _, _ in records):
        st.caption("No forecast is available for the current week in the saved results.")
    st.caption(
        f"Available dates: {format_week_date(earliest_data_date.isoformat())} to "
        f"{format_week_date(latest_data_date.isoformat())}."
    )

    st.session_state.setdefault("overview_rows_per_page", ROWS_PER_PAGE_OPTIONS[0])
    st.session_state.setdefault("overview_page", 0)
    st.session_state.setdefault("overview_date_order", DATE_ORDER_OPTIONS[0])
    st.session_state.setdefault("overview_from_date", earliest_data_date)
    st.session_state.setdefault("overview_to_date", latest_data_date)
    if not earliest_data_date <= st.session_state["overview_from_date"] <= latest_data_date:
        st.session_state["overview_from_date"] = earliest_data_date
    if not earliest_data_date <= st.session_state["overview_to_date"] <= latest_data_date:
        st.session_state["overview_to_date"] = latest_data_date
    if st.session_state["overview_from_date"] > st.session_state["overview_to_date"]:
        st.session_state["overview_from_date"] = earliest_data_date
        st.session_state["overview_to_date"] = latest_data_date

    with st.expander("Filter weeks", expanded=False):
        from_col, to_col = st.columns(2)
        with from_col:
            selected_start_date = st.date_input(
                "From",
                min_value=earliest_data_date,
                max_value=latest_data_date,
                format="MM/DD/YYYY",
                key="overview_from_date",
                on_change=_from_date_changed,
                help="Start of the date filter.",
            )
        with to_col:
            selected_end_date = st.date_input(
                "To",
                min_value=earliest_data_date,
                max_value=latest_data_date,
                format="MM/DD/YYYY",
                key="overview_to_date",
                on_change=_to_date_changed,
                help="End of the date filter.",
            )

        order_col, page_size_col = st.columns(2)
        with order_col:
            date_order = st.selectbox(
                "Date order",
                DATE_ORDER_OPTIONS,
                key="overview_date_order",
                on_change=_reset_week_page,
            )
        with page_size_col:
            rows_per_page = st.selectbox(
                "Rows per page",
                ROWS_PER_PAGE_OPTIONS,
                key="overview_rows_per_page",
                on_change=_reset_week_page,
            )

    records = [
        record
        for record in records
        if selected_start_date <= available_dates[record[0]] <= selected_end_date
    ]
    records.sort(
        key=lambda record: datetime.strptime(record[0], "%Y-%m-%d").date(),
        reverse=date_order == "Latest first",
    )

    total_rows = len(records)
    page_count = max(1, math.ceil(total_rows / rows_per_page))
    current_page = min(max(int(st.session_state["overview_page"]), 0), page_count - 1)
    st.session_state["overview_page"] = current_page
    start_index = current_page * rows_per_page
    end_index = min(start_index + rows_per_page, total_rows)

    summary_col, previous_col, next_col = st.columns([2.2, 1, 1])
    with summary_col:
        if total_rows:
            st.markdown(
                f'<div class="pagination-label">Showing {start_index + 1}–{end_index} of {total_rows:,} weeks</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="pagination-label">No weekly results available</div>', unsafe_allow_html=True)
    with previous_col:
        if st.button("Previous", key="overview_previous", disabled=current_page == 0, use_container_width=True):
            st.session_state["overview_page"] = max(0, current_page - 1)
            st.rerun()
    with next_col:
        if st.button("Next", key="overview_next", disabled=current_page >= page_count - 1, use_container_width=True):
            st.session_state["overview_page"] = min(page_count - 1, current_page + 1)
            st.rerun()

    table_rows = []
    for week, predicted, actual in records[start_index:end_index]:
        is_current_week = week == current_week
        try:
            week_date = datetime.strptime(week, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            week_date = None
        actual_is_available = actual is not None and np.isfinite(float(actual))
        if week_date is not None and week_date > today:
            actual_is_available = False
        actual_markup = (
            f'<span class="weekly-actual">{_money(float(actual))}</span>'
            if actual_is_available
            else '<span class="weekly-not-yet">Not yet</span>'
        )
        date_content = html.escape(format_week_date(week))
        if is_current_week:
            date_content += '<span class="weekly-current-badge">This week</span>'
        row_class = ' class="current-week"' if is_current_week else ""
        table_rows.append(
            f"<tr{row_class}><td><span class=\"weekly-table-date\">{date_content}</span></td>"
            f'<td><span class="weekly-forecast">{_money(float(predicted))}</span></td>'
            f"<td>{actual_markup}</td></tr>"
        )

    st.markdown(
        '<div class="weekly-table-wrap"><table class="weekly-table">'
        "<thead><tr><th scope=\"col\">Date</th><th scope=\"col\">Forecasted</th>"
        "<th scope=\"col\">Actual</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render(weeks: list[str], forecast: dict, active_model: str, show_graph: bool = False) -> None:
    selected_forecast = np.asarray(forecast[active_model], dtype=float)
    actual_values = np.asarray(forecast["actual"], dtype=float)
    starting_actual = float(actual_values[0])
    ending_forecast = float(selected_forecast[-1])
    horizon_change = ending_forecast - starting_actual
    usable_actuals = np.isfinite(actual_values)
    usable_actuals &= np.asarray(
        [datetime.strptime(week, "%Y-%m-%d").date() <= date.today() for week in weeks],
        dtype=bool,
    )
    average_deviation = (
        float(np.mean(np.abs(selected_forecast[usable_actuals] - actual_values[usable_actuals])))
        if np.any(usable_actuals)
        else float("nan")
    )
    change_text = "Rise" if horizon_change >= 0 else "Fall"

    st.markdown(
        f'<div class="page-intro">Rice price forecasts across {len(weeks)} weeks, from '
        f'{format_week_date(weeks[0])} to {format_week_date(weeks[-1])}.</div>',
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(4)
    with metric_columns[0]:
        metric_card(
            "Starting price",
            f"{_money(starting_actual)}/kg",
            format_week_date(weeks[0]),
            "The observed retail price at the beginning of the selected time period.",
        )
    with metric_columns[1]:
        metric_card(
            "Ending price",
            f"{_money(ending_forecast)}/kg",
            format_week_date(weeks[-1]),
            "The model's forecasted retail price at the end of the time period.",
        )
    with metric_columns[2]:
        metric_card(
            "Change over period",
            f"{'+' if horizon_change >= 0 else '-'}₱{abs(horizon_change):.2f}/kg"
            if np.isfinite(horizon_change)
            else "Not available",
            f"{change_text} from first to final week",
            "The total expected increase or decrease in price from the start to the end.",
        )
    with metric_columns[3]:
        metric_card(
            "Average difference",
            f"{_money(average_deviation)}/kg",
            "Average distance from observed prices",
            "How far off the predictions usually are from the actual recorded prices.",
        )

    if show_graph:
        st.subheader("Forecast and observed prices")
        st.caption("The green line shows observed prices. Blue and slate lines show the two forecasts.")
        figure = go.Figure()
        for series_key, series_name in (
            ("multivariate", "Multivariate forecast"),
            ("univariate", "Univariate baseline"),
        ):
            is_active = series_key == active_model
            figure.add_trace(
                go.Scatter(
                    x=weeks,
                    y=forecast[series_key],
                    name=series_name,
                    mode="lines+markers" if is_active else "lines",
                    line={"color": COLORS[series_key], "width": 3 if is_active else 1.8},
                    opacity=1 if is_active else 0.58,
                    hovertemplate="%{y:.2f} ₱/kg<extra>" + series_name + "</extra>",
                )
            )
        figure.add_trace(
            go.Scatter(
                x=weeks,
                y=forecast["actual"],
                name="Observed price",
                mode="lines",
                line={"color": COLORS["actual"], "width": 2.5, "dash": "dash"},
                hovertemplate="%{y:.2f} ₱/kg<extra>Observed price</extra>",
            )
        )
        all_prices = np.concatenate(
            [actual_values, np.asarray(forecast["multivariate"]), np.asarray(forecast["univariate"])]
        )
        y_padding = max(0.5, (float(all_prices.max()) - float(all_prices.min())) * 0.12)
        chart_layout(figure, 460)
        figure.update_layout(
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.08, "x": 0, "bgcolor": COLORS["white"]},
        )
        figure.update_yaxes(
            title_text="Retail price (PHP per kg)",
            tickprefix="₱",
            range=[float(all_prices.min()) - y_padding, float(all_prices.max()) + y_padding],
        )
        figure.update_xaxes(title_text="Week", tickangle=-30)
        render_chart(st.empty(), figure, "overview_technical_forecast")
    else:
        _render_week_table(weeks, forecast, active_model)
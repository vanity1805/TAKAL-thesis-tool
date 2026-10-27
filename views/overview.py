"""Overview screen with price summaries and a paginated weekly forecast table."""

from __future__ import annotations

import html
import math
from datetime import date, datetime
from typing import Callable, Sequence

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components import chart_layout, metric_card, render_chart, render_reading
from interpretation import overview_reading, peso
from utils import format_week_date

ROWS_PER_PAGE_OPTIONS = (10, 25, 50, 100)
DATE_ORDER_OPTIONS = ("Latest first", "Oldest first")


def _reset_week_page() -> None:
    st.session_state["overview_page"] = 0


def _horizon_changed() -> None:
    """Reset pagination when the user changes the forecast horizon dropdown."""
    st.session_state["overview_page"] = 0


def _render_select(
    label: str,
    options: Sequence[str | int],
    key: str,
    default: str | int,
    on_change: Callable[[], None] | None = None,
) -> str | int:
    """Render a real select box.

    A select box (not a popover of buttons) gives the open list, its position under
    the field, and its closing behaviour the same look as the SHAP dropdowns.
    """
    st.session_state.setdefault(key, default)
    return st.selectbox(label, options, key=key, on_change=on_change)


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


def _reset_overview_filters(earliest_data_date: date, latest_data_date: date) -> None:
    st.session_state["overview_page"] = 0
    st.session_state["overview_date_order"] = DATE_ORDER_OPTIONS[0]
    st.session_state["overview_rows_per_page"] = ROWS_PER_PAGE_OPTIONS[0]
    st.session_state["overview_from_date"] = earliest_data_date
    st.session_state["overview_to_date"] = latest_data_date


def _money(value: float | str | None) -> str:
    if value is None or isinstance(value, str):
        try:
            if value is None or str(value).strip() in ("", "-", "—"):
                return "-"
            val = float(value)
        except (TypeError, ValueError):
            return "-"
    else:
        val = float(value)
    if not np.isfinite(val):
        return "-"
    return f"₱{val:.2f}"


def _usable_actual(week: str, actual, today: date | None = None) -> bool:
    """Return True only when an actual price is a finite number and its date is not in the future."""
    today = today or date.today()
    try:
        week_date = datetime.strptime(week, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        week_date = None
    try:
        if actual is None or isinstance(actual, str) and actual.strip() in ("", "-", "—"):
            return False
        if not np.isfinite(float(actual)):
            return False
    except (TypeError, ValueError):
        return False
    return not (week_date is not None and week_date > today)


def _build_combined_series(
    weeks: list[str],
    horizon_forecast: dict,
    horizon: int,
    future_forecast: dict | None,
) -> tuple[list[str], dict[str, list[float | None]]]:
    """Combine historical test weeks with up to ``horizon`` dynamically appended future weeks.

    If actual data later becomes available for any of the future dates (either inside
    ``horizon_forecast['actual']`` or ``future_forecast['actual']``), it automatically
    populates that date's actual value so the "-" placeholder is replaced with the real number.
    """
    combined_weeks = list(weeks)
    combined_mv = [float(v) for v in horizon_forecast["multivariate"]]
    combined_uni = [float(v) for v in horizon_forecast["univariate"]]
    combined_actual: list[float | None] = [
        float(v) if v is not None and np.isfinite(float(v)) else None
        for v in horizon_forecast["actual"]
    ]

    # Map any known actual prices by ISO date so future rows automatically resolve if data exists
    actual_lookup: dict[str, float] = {}
    for w, act in zip(weeks, combined_actual):
        if act is not None and np.isfinite(act):
            actual_lookup[w] = float(act)

    if future_forecast:
        future_dates = list(future_forecast.get("horizon_dates") or [])[: int(horizon)]
        future_mv = list(future_forecast.get("multivariate_predictions") or [])[: int(horizon)]
        future_uni = list(future_forecast.get("univariate_predictions") or [])[: int(horizon)]
        future_actuals = future_forecast.get("actual") or {}

        if isinstance(future_actuals, dict):
            for f_date, f_val in future_actuals.items():
                try:
                    if f_val is not None and np.isfinite(float(f_val)):
                        actual_lookup[str(f_date)] = float(f_val)
                except (TypeError, ValueError):
                    pass
        elif isinstance(future_actuals, (list, tuple)):
            for f_date, f_val in zip(future_forecast.get("horizon_dates") or [], future_actuals):
                try:
                    if f_val is not None and np.isfinite(float(f_val)):
                        actual_lookup[str(f_date)] = float(f_val)
                except (TypeError, ValueError):
                    pass

        existing_dates = set(combined_weeks)
        for f_date, mv_val, uni_val in zip(future_dates, future_mv, future_uni):
            if f_date not in existing_dates:
                combined_weeks.append(f_date)
                combined_mv.append(float(mv_val))
                combined_uni.append(float(uni_val))
                combined_actual.append(actual_lookup.get(f_date, None))
                existing_dates.add(f_date)

    return combined_weeks, {
        "multivariate": combined_mv,
        "univariate": combined_uni,
        "actual": combined_actual,
    }


def _render_week_table(weeks: list[str], forecast: dict, active_model: str, horizon: int) -> None:
    records = list(zip(weeks, forecast[active_model], forecast["actual"]))
    records.sort(key=lambda record: record[0], reverse=True)

    available_dates = {
        week: datetime.strptime(week, "%Y-%m-%d").date()
        for week, _, _ in records
    }
    earliest_data_date = min(available_dates.values(), default=date(2025, 1, 1))
    latest_data_date = max(available_dates.values(), default=date(2026, 4, 30))
    today = date.today()

    st.markdown('<div class="section-subheading">Weekly forecast vs actual</div>', unsafe_allow_html=True)
    st.caption(
        f"Available dates: {format_week_date(earliest_data_date.isoformat())} to "
        f"{format_week_date(latest_data_date.isoformat())}."
    )

    st.session_state.setdefault("overview_rows_per_page", ROWS_PER_PAGE_OPTIONS[0])
    st.session_state.setdefault("overview_page", 0)
    st.session_state.setdefault("overview_date_order", DATE_ORDER_OPTIONS[0])
    st.session_state.setdefault("overview_from_date", earliest_data_date)
    st.session_state.setdefault("overview_to_date", latest_data_date)

    # Whenever the date range or selected horizon changes, widen the filter range
    # so newly appended future horizon rows are immediately visible in the table.
    data_range = (earliest_data_date, latest_data_date, int(horizon))
    if st.session_state.get("overview_data_range") != data_range:
        st.session_state["overview_from_date"] = earliest_data_date
        st.session_state["overview_to_date"] = latest_data_date
        st.session_state["overview_page"] = 0
        st.session_state["overview_data_range"] = data_range

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
            date_order = _render_select(
                "Date order",
                DATE_ORDER_OPTIONS,
                "overview_date_order",
                DATE_ORDER_OPTIONS[0],
                on_change=_reset_week_page,
            )
        with page_size_col:
            rows_per_page = _render_select(
                "Rows per page",
                ROWS_PER_PAGE_OPTIONS,
                "overview_rows_per_page",
                ROWS_PER_PAGE_OPTIONS[0],
                on_change=_reset_week_page,
            )

        st.button(
            "Back to default",
            key="overview_default_button",
            on_click=_reset_overview_filters,
            args=(earliest_data_date, latest_data_date),
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

    table_rows = []
    for week, predicted, actual in records[start_index:end_index]:
        actual_is_available = _usable_actual(week, actual, today)
        actual_markup = (
            f'<span class="weekly-actual">{_money(float(actual))}</span>'
            if actual_is_available
            else '<span class="weekly-not-yet">-</span>'
        )
        if actual_is_available and predicted is not None and np.isfinite(float(predicted)):
            difference = float(predicted) - float(actual)
            css = "flat" if round(difference, 2) == 0 else ("up" if difference > 0 else "down")
            difference_markup = f'<span class="weekly-diff-{css}">{peso(difference, signed=True)}</span>'
        else:
            difference_markup = '<span class="weekly-not-yet">-</span>'

        date_content = html.escape(format_week_date(week))
        table_rows.append(
            f'<tr><td class="col-center col-label"><span class="weekly-table-date">{date_content}</span></td>'
            f'<td class="col-center"><span class="weekly-forecast">{_money(predicted)}</span></td>'
            f'<td class="col-center">{actual_markup}</td>'
            f'<td class="col-center">{difference_markup}</td></tr>'
        )

    st.markdown(
        '<div class="weekly-table-wrap"><table class="weekly-table">'
        '<thead><tr><th scope="col" class="col-center">Date</th>'
        '<th scope="col" class="col-center">Forecasted</th>'
        '<th scope="col" class="col-center">Actual</th>'
        '<th scope="col" class="col-center">Difference'
        '<span class="th-sub">Forecasted − Actual</span></th></tr></thead><tbody>'
        + "".join(table_rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )

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

    if records:
        reading_weeks = [week for week, _, _ in records]
        reading_forecast = [float(predicted) for _, predicted, _ in records]
        reading_actual = [
            float(actual) if _usable_actual(week, actual, today) else float("nan")
            for week, _, actual in records
        ]
        render_reading(
            f"Reading the table: {len(records):,} weeks in the selected date range",
            overview_reading(reading_weeks, reading_forecast, reading_actual, active_model, horizon),
        )


def render(
    weeks: list[str],
    forecast: dict,
    active_model: str,
    show_graph: bool = False,
    future_forecast: dict | None = None,
) -> None:
    horizon = int(
        _render_select(
            "Forecast Horizon (Weeks Ahead)",
            tuple(range(1, 13)),
            "overview_horizon",
            1,
            on_change=_horizon_changed,
        )
    )
    raw_horizon_forecast = forecast[str(horizon)]
    combined_weeks, horizon_forecast = _build_combined_series(
        weeks, raw_horizon_forecast, horizon, future_forecast
    )

    selected_forecast = np.asarray(horizon_forecast[active_model], dtype=float)
    raw_actuals = [
        float(val) if val is not None and np.isfinite(float(val)) else float("nan")
        for val in horizon_forecast["actual"]
    ]
    actual_values = np.asarray(raw_actuals, dtype=float)

    today = date.today()
    usable_actuals = np.asarray(
        [_usable_actual(week, val, today) for week, val in zip(combined_weeks, actual_values)],
        dtype=bool,
    )
    plotted_actuals = np.where(usable_actuals, actual_values, np.nan)

    starting_actual = float(plotted_actuals[0]) if len(plotted_actuals) else float("nan")
    ending_forecast = float(selected_forecast[-1]) if len(selected_forecast) else float("nan")
    horizon_change = (
        ending_forecast - starting_actual
        if np.isfinite(starting_actual) and np.isfinite(ending_forecast)
        else float("nan")
    )

    average_deviation = (
        float(np.mean(np.abs(selected_forecast[usable_actuals] - actual_values[usable_actuals])))
        if np.any(usable_actuals)
        else float("nan")
    )
    change_text = (
        f"{'Rise' if horizon_change >= 0 else 'Fall'} from first to final week"
        if np.isfinite(horizon_change)
        else "Waiting on official price release"
    )

    change_pct = (
        horizon_change / starting_actual * 100
        if np.isfinite(horizon_change) and np.isfinite(starting_actual) and starting_actual != 0
        else float("nan")
    )
    change_badge = (
        f"{'+' if change_pct >= 0 else '-'}{abs(change_pct):.1f}%"
        if np.isfinite(change_pct)
        else None
    )

    future_added = len(combined_weeks) - len(weeks)
    future_note = (
        f" plus {future_added} forecasted future {'week' if future_added == 1 else 'weeks'} "
        f"(through {format_week_date(combined_weeks[-1])})"
        if future_added > 0
        else ""
    )

    st.markdown(
        f'<div class="page-intro">{horizon}-week-ahead <strong>Regular-Milled</strong> retail rice price forecasts in the '
        f'<strong>National Capital Region (NCR)</strong> across the {len(weeks)}-week held-out test set '
        f'({format_week_date(weeks[0])} to {format_week_date(weeks[-1])}){future_note}.</div>',
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    with metric_columns[0]:
        metric_card(
            "Starting observed price",
            f"{_money(starting_actual)}/kg" if np.isfinite(starting_actual) else "-",
            format_week_date(combined_weeks[0]),
            "The actual observed composite retail price at the start of the period.",
            badge="Baseline",
            always_dark=False,
        )
    with metric_columns[1]:
        metric_card(
            "Ending forecast",
            f"{_money(ending_forecast)}/kg" if np.isfinite(ending_forecast) else "-",
            format_week_date(combined_weeks[-1]),
            f"The Stacked LSTM model's {horizon}-week-ahead forecast for the final week shown ({format_week_date(combined_weeks[-1])}).",
            badge="LSTM",
            always_dark=False,
        )
    with metric_columns[2]:
        metric_card(
            "Change over period",
            f"{'+' if horizon_change >= 0 else '-'} {abs(horizon_change):.2f}/kg"
            if np.isfinite(horizon_change)
            else "-",
            change_text,
            "The net difference between the starting observed price and the final week's forecasted price. The badge shows the same change as a percentage of the starting price.",
            badge=change_badge,
            always_dark=False,
        )
    with metric_columns[3]:
        metric_card(
            "Average deviation (MAE)",
            f"{_money(average_deviation)}/kg" if np.isfinite(average_deviation) else "-",
            "Mean absolute error across observed weeks",
            "Mean Absolute Error (MAE): average absolute difference between forecasted and observed prices on weeks where official data is available.",
            badge="Error",
            always_dark=False,
        )

    if show_graph:
        st.markdown('<div class="section-subheading">Forecast and observed prices</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Black line shows observed prices. Mint and sage lines indicate model forecasts.</div>',
            unsafe_allow_html=True,
        )

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=combined_weeks,
                y=horizon_forecast["univariate"],
                name="Univariate baseline",
                mode="lines",
                line={"color": "#6E998B", "width": 1.8, "dash": "dot"},
                hovertemplate="%{y:.2f} ₱/kg<extra>Univariate baseline</extra>",
            )
        )
        is_active = active_model == "multivariate"
        figure.add_trace(
            go.Scatter(
                x=combined_weeks,
                y=horizon_forecast["multivariate"],
                name="Multivariate forecast",
                mode="lines+markers" if is_active else "lines",
                marker={"size": 4.5, "color": "#1BB385"},
                line={"color": "#1BB385", "width": 2.6},
                fill="tozeroy",
                fillcolor="rgba(37, 201, 152, 0.08)",
                hovertemplate="%{y:.2f} ₱/kg<extra>Multivariate forecast</extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=combined_weeks,
                y=plotted_actuals.tolist(),
                name="Observed price",
                mode="lines",
                connectgaps=False,
                line={"color": "#000000", "width": 2.4},
                hovertemplate="%{y:.2f} ₱/kg<extra>Observed price</extra>",
            )
        )

        all_prices = np.concatenate(
            [
                plotted_actuals[np.isfinite(plotted_actuals)],
                np.asarray(horizon_forecast["multivariate"], dtype=float),
                np.asarray(horizon_forecast["univariate"], dtype=float),
            ]
        )
        finite_prices = all_prices[np.isfinite(all_prices)]
        if len(finite_prices):
            min_price, max_price = float(np.nanmin(finite_prices)), float(np.nanmax(finite_prices))
        else:
            min_price, max_price = 0.0, 100.0
        y_padding = max(0.5, (max_price - min_price) * 0.15)

        chart_layout(figure, 440)
        figure.update_layout(
            hovermode="x unified",
            legend={
                "orientation": "h",
                "y": 1.02,
                "yanchor": "bottom",
                "x": 0,
                "xanchor": "left",
                "font": {"size": 12, "color": "#0B221B"},
            },
            margin={"t": 80, "r": 15, "b": 30, "l": 10},
        )
        figure.update_yaxes(
            title_text="Retail price (PHP / kg)",
            tickprefix="₱",
            range=[min_price - y_padding, max_price + y_padding],
        )
        figure.update_xaxes(
            title_text="Week",
            tickangle=-30,
        )

        with st.container(key="chart_card_forecast"):
            render_chart(st.empty(), figure, "overview_technical_forecast")

        # Add the reading box under the chart in technical details mode as well
        render_reading(
            f"Reading the {horizon}-week-ahead forecast ({len(combined_weeks)} weeks shown)",
            overview_reading(
                combined_weeks,
                horizon_forecast[active_model],
                plotted_actuals.tolist(),
                active_model,
                horizon,
            ),
        )
    else:
        _render_week_table(combined_weeks, horizon_forecast, active_model, horizon)
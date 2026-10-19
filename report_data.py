"""Assemble the content of an exported TAKAL report.

``build_report`` turns the saved results into a plain, format-neutral structure:
sections -> blocks (key figures, tables, charts, readings). ``report_pdf`` and
``report_excel`` only decide how to draw it, so both files always say the same thing.

This module does not import Streamlit.
"""

from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import pandas as pd

from interpretation import (
    MODEL_NAMES,
    direction_text,
    dm_reading,
    importance_reading,
    importance_rows,
    local_reading,
    long_date,
    metrics_reading,
    overview_reading,
    peso,
)

SECTION_ORDER = ("Overview", "Benchmarking", "Explainability")

SECTION_BLURBS = {
    "Overview": "Weekly forecast versus the actual price, and how far apart they were.",
    "Benchmarking": "How accurate each model is, and whether the two models truly differ.",
    "Explainability": "Which indicators the model relied on, and how they moved the forecast.",
}

MODEL_TITLES = {"multivariate": "Multivariate model", "univariate": "Univariate model"}

GLOSSARY = [
    ("Forecast horizon", "How many weeks ahead a forecast looks. A 12-week-ahead forecast for a given week was made 12 weeks earlier."),
    ("Test weeks", "Weeks the models never saw while learning. Accuracy is measured on these, so it shows how the models do on new data."),
    ("Multivariate model", "A forecasting model that uses past rice prices plus economic indicators (exchange rate, imports, stockpile, palay production, diesel price)."),
    ("Univariate model", "A baseline model that uses only past rice prices."),
    ("Difference", "Forecasted price minus actual price, in PHP per kg. Positive = the forecast was too high; negative = too low."),
    ("MAE", "Mean Absolute Error: the average size of the miss, in PHP per kg. Lower is better."),
    ("RMSE", "Root Mean Squared Error: like MAE but punishes large misses more. Lower is better."),
    ("MAPE", "Mean Absolute Percentage Error: the average miss as a percentage of the actual price. Lower is better."),
    ("R²", "How much of the price movement the forecasts follow. 1 is a perfect match; 0 is no better than guessing the average; below 0 is worse than that."),
    ("Diebold–Mariano (DM) test", "A statistical test of whether two models' forecast errors really differ or whether the gap could be luck."),
    ("p-value", "The chance of seeing a gap this large if the two models were equally good. Below the threshold (alpha) means the gap is unlikely to be luck."),
    ("Null hypothesis (H₀)", "The starting assumption that the two models are equally accurate. 'Retained' means the test found no evidence against it; 'rejected' means it found evidence of a difference."),
    ("SHAP", "A method that splits a forecast into the amount each input indicator added or subtracted. It explains how the model works; it does not prove cause and effect."),
    ("Baseline (SHAP)", "The model's average forecast. Each indicator's contribution is measured from this starting point."),
    ("Lag", "A past value used as an input, such as 'Rice price 2 weeks ago'."),
]

# column kinds understood by the PDF and Excel writers
#   text, date, peso, peso_signed, dec3, dec3_signed, dec4, dec6, pct0, pct1, int, effect


def _nice_date(iso: str) -> str:
    return long_date(iso)


def _usable(actual: float, week: str, today: date) -> bool:
    if actual is None or not np.isfinite(actual):
        return False
    try:
        return datetime.strptime(week, "%Y-%m-%d").date() <= today
    except ValueError:
        return True


def _overview_section(results: dict, model: str, horizon: int, today: date) -> dict:
    weeks = results["weeks"]
    block = results["forecast"][str(horizon)]
    forecast = np.asarray(block[model], dtype=float)
    other = "univariate" if model == "multivariate" else "multivariate"
    other_forecast = np.asarray(block[other], dtype=float)
    actual = np.asarray(block["actual"], dtype=float)
    usable = np.asarray([_usable(a, w, today) for a, w in zip(actual, weeks)])
    actual_used = np.where(usable, actual, np.nan)

    start_actual = float(actual[0])
    end_forecast = float(forecast[-1])
    change = end_forecast - start_actual
    mae = float(np.mean(np.abs(forecast[usable] - actual[usable]))) if usable.any() else float("nan")
    pct = change / start_actual * 100 if start_actual else float("nan")

    cards = [
        ("Starting observed price", f"{peso(start_actual)}/kg", _nice_date(weeks[0])),
        ("Ending forecast", f"{peso(end_forecast)}/kg", _nice_date(weeks[-1])),
        ("Change over period", f"{peso(change, signed=True)}/kg", f"{pct:+.1f}% from first to final week"),
        ("Average deviation (MAE)", f"{peso(mae)}/kg", "Average miss across test weeks"),
    ]
    diff = forecast - actual_used
    table = pd.DataFrame(
        {
            "Date": [datetime.strptime(w, "%Y-%m-%d").date() for w in weeks],
            "Forecasted (PHP/kg)": forecast,
            "Actual (PHP/kg)": actual_used,
            "Difference (PHP/kg)": diff,
        }
    )
    return {
        "key": "Overview",
        "title": "Overview",
        "intro": (
            f"{horizon}-week-ahead forecasts of the regular-milled retail rice price in the National Capital "
            f"Region, for {len(weeks)} test weeks ({_nice_date(weeks[0])} to {_nice_date(weeks[-1])}). "
            f"Figures below use the {MODEL_NAMES[model]}. The Difference column shows how far each forecast "
            "was from the actual price."
        ),
        "blocks": [
            {"type": "cards", "items": cards},
            {
                "type": "chart",
                "kind": "forecast",
                "title": "Forecast and actual prices",
                "caption": "The black line is the actual price. The coloured lines are the two models' forecasts "
                           "for the same weeks.",
                "weeks": list(weeks),
                "actual": actual_used.tolist(),
                "series": {model: forecast.tolist(), other: other_forecast.tolist()},
                "active": model,
            },
            {
                "type": "chart",
                "kind": "difference",
                "title": "Difference between forecast and actual",
                "caption": "Bars above zero are weeks the forecast was too high; bars below zero are weeks it was too low.",
                "weeks": list(weeks),
                "values": diff.tolist(),
            },
            {
                "type": "table",
                "id": "overview_weekly",
                "title": f"Weekly forecast versus actual ({MODEL_TITLES[model]}, {horizon} weeks ahead)",
                "df": table,
                "kinds": {
                    "Date": "date",
                    "Forecasted (PHP/kg)": "peso",
                    "Actual (PHP/kg)": "peso",
                    "Difference (PHP/kg)": "peso_signed",
                },
                "note": "Weeks are listed oldest to newest. Difference = Forecasted − Actual.",
                "reading_title": "Reading the weekly table",
                "reading": overview_reading(weeks, forecast, actual_used, model, horizon),
            },
        ],
    }


def _benchmark_section(results: dict, model: str) -> dict:
    metrics, dm = results["metrics"], results["dm_test"]
    order = [model, "univariate" if model == "multivariate" else "multivariate"]
    horizons = [*(str(h) for h in range(1, 13)), "overall"]
    blocks: list[dict] = []
    for position, name in enumerate(order):
        rows = [
            {
                "Forecast distance": "Overall" if h == "overall" else f"Week {h}",
                "RMSE": float(metrics[h][name]["rmse"]),
                "MAE": float(metrics[h][name]["mae"]),
                "MAPE (%)": float(metrics[h][name]["mape"]),
                "R²": float(metrics[h][name]["r2"]),
            }
            for h in horizons
        ]
        reading = metrics_reading(metrics, name)
        if position:  # the 'how to read' explanation is only needed once
            reading = reading[1:]
        blocks.append(
            {
                "type": "table",
                "id": f"accuracy_{name}",
                "title": f"Forecast accuracy by distance ahead: {MODEL_TITLES[name]}",
                "df": pd.DataFrame(rows),
                "kinds": {"Forecast distance": "text", "RMSE": "dec4", "MAE": "dec4", "MAPE (%)": "dec4", "R²": "dec4"},
                "note": "RMSE and MAE are in PHP per kg; MAPE is a percentage. Lower error is better; higher R² is better.",
                "reading_title": f"Reading the accuracy table ({MODEL_TITLES[name]})",
                "reading": reading,
            }
        )
    dm_rows = []
    for h in horizons:
        row = dm[h]
        significant = bool(row.get("significant", float(row["p_value"]) < float(row["alpha"])))
        dm_rows.append(
            {
                "Forecast distance": "Overall" if h == "overall" else f"Week {h}",
                "DM statistic": float(row["statistic"]),
                "p-value": float(row["p_value"]),
                "Alpha": float(row["alpha"]),
                "Significant?": "Yes" if significant else "No",
                "Lower-error model": MODEL_TITLES.get(row.get("better"), "-") if significant else "No clear winner",
                "Decision": row["decision"],
            }
        )
    blocks.append(
        {
            "type": "table",
            "id": "diebold_mariano",
            "title": "Do the two models really differ? Diebold–Mariano test",
            "df": pd.DataFrame(dm_rows),
            "kinds": {
                "Forecast distance": "text",
                "DM statistic": "dec4",
                "p-value": "dec6",
                "Alpha": "dec3",
                "Significant?": "text",
                "Lower-error model": "text",
                "Decision": "text",
            },
            "note": "'Retain H0' means no significant difference was found; 'Reject H0' means a significant difference was found.",
            "reading_title": "Reading the Diebold–Mariano table",
            "reading": dm_reading(dm),
        }
    )
    n = len(results["weeks"])
    return {
        "key": "Benchmarking",
        "title": "Benchmarking",
        "intro": (
            f"These measures compare each model's forecasts with the actual price on the same {n} held-out "
            "weeks. Both models are shown so they can be compared side by side. Lower error values mean the "
            "forecasts were closer to the actual prices."
        ),
        "blocks": blocks,
    }


def _explain_section(results: dict, model: str, horizon: int, week: str | None, label_fn) -> dict:
    weeks = results["weeks"]
    shap_model = results["shap"][str(horizon)][model]
    labels = {n: label_fn(n) for n in set(shap_model["features"]) | set(shap_model["detail_features"])}
    ranked = importance_rows(shap_model["features"], shap_model["bar"])
    named = [{**r, "name": labels[r["name"]]} for r in ranked]

    def direction(name: str) -> str:
        point = shap_model["beeswarm"].get(name)
        return direction_text(point["feature_value"], point["shap"]) if point else "No clear pattern"

    importance_df = pd.DataFrame(
        [
            {
                "Rank": i,
                "Indicator": labels[r["name"]],
                "Average impact (PHP/kg)": r["impact"],
                "Share of total": r["share"],
                "Typical direction": direction(r["name"]),
            }
            for i, r in enumerate(ranked, start=1)
        ]
    )
    blocks: list[dict] = [
        {
            "type": "chart",
            "kind": "importance",
            "title": "Which indicators mattered most",
            "caption": "A longer bar means the indicator moved the forecasts more, on average, across the test weeks.",
            "labels": [r["name"] for r in named],
            "values": [r["impact"] for r in named],
            "model": model,
        },
        {
            "type": "table",
            "id": "importance",
            "title": f"Indicator importance ({MODEL_TITLES[model]}, {horizon} weeks ahead)",
            "df": importance_df,
            "kinds": {
                "Rank": "int",
                "Indicator": "text",
                "Average impact (PHP/kg)": "dec3",
                "Share of total": "pct0",
                "Typical direction": "text",
            },
            "note": "Average impact is the mean absolute SHAP value across the test weeks.",
            "reading_title": "Reading the importance table",
            "reading": importance_reading(named, model, horizon),
        },
    ]

    if week and week in weeks:
        entry = shap_model["local"][weeks.index(week)]
        baseline = float(shap_model["base_value"])
        predicted = float(entry["predicted"])
        ordered = sorted(entry["contributions"].items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        running = baseline
        steps = []
        for i, (name, value) in enumerate(ordered, start=1):
            value = float(value)
            running += value
            steps.append(
                {
                    "Step": i,
                    "Indicator": label_fn(name, week),
                    "Contribution (PHP/kg)": value,
                    "Effect": "Raises the estimate" if value > 0 else "Lowers the estimate" if value < 0 else "No change",
                    "Estimate after this step (PHP/kg)": running,
                }
            )
        actual = float(results["forecast"][str(horizon)]["actual"][weeks.index(week)])
        blocks += [
            {
                "type": "heading",
                "text": f"Explaining one week: {_nice_date(week)}",
                "caption": "The tables above describe the whole test period. This part opens up a single forecast "
                           "to show how each indicator contributed to it.",
            },
            {
                "type": "cards",
                "items": [
                    ("Baseline", f"{peso(baseline)}/kg", "The model's average forecast"),
                    ("Forecast for the week", f"{peso(predicted)}/kg", _nice_date(week)),
                    ("Actual price", f"{peso(actual)}/kg", _nice_date(week)),
                    ("Difference", f"{peso(predicted - actual, signed=True)}/kg", "Forecasted − Actual"),
                ],
            },
            {
                "type": "chart",
                "kind": "waterfall",
                "title": f"How the forecast for {_nice_date(week)} was built",
                "caption": "Starting from the baseline, each bar shows how much one indicator raised (green) or lowered (blue) the estimate.",
                "labels": [s["Indicator"] for s in steps],
                "values": [s["Contribution (PHP/kg)"] for s in steps],
                "baseline": baseline,
                "predicted": predicted,
            },
            {
                "type": "table",
                "id": "local_week",
                "title": f"Single-week explanation: week of {_nice_date(week)} ({MODEL_TITLES[model]}, {horizon} weeks ahead)",
                "df": pd.DataFrame(steps),
                "kinds": {
                    "Step": "int",
                    "Indicator": "text",
                    "Contribution (PHP/kg)": "dec3_signed",
                    "Effect": "effect",
                    "Estimate after this step (PHP/kg)": "dec3",
                },
                "note": f"The running estimate starts at the baseline of {peso(baseline)}/kg and ends at the forecast of {peso(predicted)}/kg.",
                "reading_title": "Reading the single-week table",
                "reading": local_reading(
                    [(s["Indicator"], s["Contribution (PHP/kg)"]) for s in steps], baseline, predicted
                ),
            },
        ]
    return {
        "key": "Explainability",
        "title": "Explainability (SHAP)",
        "intro": (
            f"These explanations describe how the {MODEL_NAMES[model]} used its input indicators for "
            f"{horizon}-week-ahead forecasts. They explain the model's calculations; they do not prove that an "
            "indicator caused a market-price change."
        ),
        "blocks": blocks,
    }


def _friendly_note(note: str) -> str:
    """Swap raw column names in the pipeline's data note for the dashboard's labels."""
    try:
        from model import config

        pairs = sorted(config.MULTIVARIATE_FEATURES, key=lambda pair: -len(pair[0]))
    except Exception:  # the note is still useful with raw names
        return note
    for column, label in pairs:
        note = note.replace(column, label)
    return note


def _plain(text: str) -> str:
    """Replace em dashes: "-" in titles and placeholders, "," inside sentences."""
    if "\u2014" not in text:
        return text
    if text.strip() == "\u2014":
        return "-"
    if len(text.split()) >= 5:
        text = re.sub(r"\s*\u2014\s*", ", ", text)
    else:
        text = re.sub(r"\s*\u2014\s*", " - ", text)
    return text.replace("\u2014", "-")


def _scrub(obj):
    """Apply _plain to every string value in the report structure."""
    if isinstance(obj, str):
        return _plain(obj)
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_scrub(v) for v in obj)
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].map(lambda v: _scrub(v) if isinstance(v, str) else v)
        return out
    return obj


def build_report(
    results: dict,
    sections: list[str] | tuple[str, ...],
    model: str,
    horizon: int,
    local_week: str | None,
    label_fn,
    today: date | None = None,
) -> dict:
    """Return the report structure for the chosen sections (in dashboard order)."""
    today = today or date.today()
    chosen = [s for s in SECTION_ORDER if s in sections]
    if not chosen:
        raise ValueError("Select at least one section to export.")
    meta = results.get("meta") or {}
    weeks = results["weeks"]
    built: list[dict] = []
    for name in chosen:
        if name == "Overview":
            built.append(_overview_section(results, model, horizon, today))
        elif name == "Benchmarking":
            built.append(_benchmark_section(results, model))
        else:
            built.append(_explain_section(results, model, horizon, local_week, label_fn))
    info = [
        ("Report created", datetime.now().strftime("%B %d, %Y, %I:%M %p").replace(" 0", " ")),
        ("What is forecast", "Weekly retail price of regular-milled rice (PHP per kg), National Capital Region"),
        ("Test period", f"{_nice_date(weeks[0])} to {_nice_date(weeks[-1])} ({len(weeks)} weeks)"),
        ("Model shown", MODEL_TITLES[model]),
    ]
    if "Overview" in chosen or "Explainability" in chosen:
        info.append(("Forecast distance", f"{horizon} weeks ahead"))
    if meta.get("data_range"):
        info.append(("Data covers", f"{_nice_date(meta['data_range'][0])} to {_nice_date(meta['data_range'][1])}"))
    return _scrub({
        "title": "TAKAL Rice Price Forecast Report",
        "subtitle": "Regular-milled rice · National Capital Region · Weekly retail price",
        "info": info,
        "sections": built,
        "glossary": GLOSSARY,
        "data_note": _friendly_note(meta.get("data_note", "")),
    })
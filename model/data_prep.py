# =============================================================================
# Program      : TAKAL data preparation (model/data_prep.py)
# Description  : Loads the NCR weekly rice price workbook, checks it, fills the
#                few sanctioned gaps and hands a clean table to the pipeline.
# Author       : TAKAL Thesis Team (contact: chavezearljustineb@gmail.com)
# System Fit   : First stage of the offline pipeline (python -m model.run_all).
#                Its output feeds preprocessing.py. A failure here is reported
#                as DatasetLoadError, which the Streamlit dashboard shows as a
#                formatted "data could not be loaded" message.
# Date Written : 2026
# Date Revised : 2026-10-27
# Purpose      : Make sure the 330-week macroeconomic dataset (rice price,
#                diesel, USD/PHP, palay production, stockpile, imports) is
#                complete, weekly and chronologically ordered before any model
#                sees it.
# Algorithms   : Chronological sort; duplicate-date and weekly-gap checks;
#                bounded forward-fill of trailing gaps (thesis-sanctioned
#                imputation); leading-gap fill from the first observed value
#                (never back-fill interior gaps, which would leak the future).
# Data         : pandas DataFrame (one row per week); NumPy boolean masks used
# Structures     to find missing tails; a dict {column: weeks_filled} stored in
#                frame.attrs["imputed_weeks"]; lists of column names.
# =============================================================================
"""Data loading, imputation, and validation for the TAKAL pipeline.

Loads the master dataset, verifies the target column, forward-fills the trailing
gaps (thesis-sanctioned imputation), and validates that every multivariate
feature is fully populated before modelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .errors import DatasetLoadError


def load_dataset() -> pd.DataFrame:
    """Load the Clean Table sheet, sorted chronologically with a datetime index column.

    Raises ``DatasetLoadError`` (never a bare pandas / OS error) when the workbook
    is missing, cannot be opened, lacks the expected sheet or has no date column.
    """
    try:
        weekly_data = pd.read_excel(config.DATASET_PATH, sheet_name=config.DATASET_SHEET)
    except FileNotFoundError as error:
        raise DatasetLoadError(
            f"The NCR rice price workbook was not found at {config.DATASET_PATH}.",
            hint="Place THESIS-MASTER_DATASET.xlsx in the project folder, then run "
                 "`python -m model.run_all` again.",
        ) from error
    except PermissionError as error:
        raise DatasetLoadError(
            "The NCR rice price workbook could not be opened (permission denied).",
            hint="Close the file in Excel and check that you are allowed to read it.",
        ) from error
    except (OSError, ValueError, KeyError, ImportError) as error:
        # ValueError / KeyError: missing sheet or a file that is not a real .xlsx.
        raise DatasetLoadError(
            f"The NCR rice price workbook could not be read: {type(error).__name__}: {error}",
            hint=f"Check that the file is a valid .xlsx workbook with a sheet named "
                 f"'{config.DATASET_SHEET}'.",
        ) from error

    weekly_data.columns = [str(column).strip() for column in weekly_data.columns]
    if config.DATE_COL not in weekly_data.columns:
        raise DatasetLoadError(
            f"Expected a '{config.DATE_COL}' column; found {list(weekly_data.columns)}.",
            hint="Restore the original column headers of the 'Clean Table' sheet.",
        )
    try:
        weekly_data[config.DATE_COL] = pd.to_datetime(weekly_data[config.DATE_COL])
    except (ValueError, TypeError) as error:
        raise DatasetLoadError(
            f"The '{config.DATE_COL}' column contains values that are not dates: {error}",
            hint="Make sure every row in the date column holds a real date.",
        ) from error
    return weekly_data.sort_values(config.DATE_COL).reset_index(drop=True)


def _trim_unpublished_tail(weekly_data: pd.DataFrame) -> pd.DataFrame:
    """Drop trailing rows whose target price has not been published yet.

    New weeks are often typed in before the price is known. Those rows cannot be
    used as targets, so they are removed from the end (never from the middle).
    """
    has_price = weekly_data[config.TARGET_COL].notna().to_numpy()
    if not has_price.any():
        raise ValueError("Target column has no values; cannot train.")
    last_valid_row = int(np.flatnonzero(has_price)[-1])
    dropped_rows = len(weekly_data) - 1 - last_valid_row
    if dropped_rows:
        print(f"[data] Ignoring {dropped_rows} trailing row(s) with no target price yet.")
    return weekly_data.iloc[: last_valid_row + 1].copy()


def _check_weekly_spacing(dates: pd.Series) -> None:
    """Reject duplicate dates and warn when rows are not exactly one week apart."""
    if dates.duplicated().any():
        duplicate_dates = dates[dates.duplicated()].dt.date.tolist()[:5]
        raise ValueError(f"Duplicate dates in the dataset: {duplicate_dates}")
    gaps_in_days = dates.diff().dt.days.iloc[1:]
    irregular_gaps = gaps_in_days[gaps_in_days != config.EXPECTED_WEEKLY_STEP_DAYS]
    if len(irregular_gaps):
        print(f"[data] WARNING: {len(irregular_gaps)} row(s) are not exactly "
              f"{config.EXPECTED_WEEKLY_STEP_DAYS} days after the previous one "
              f"(first at {dates.loc[irregular_gaps.index[0]].date()}). "
              "Windows assume weekly steps.")


def prepare(weekly_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a modelling-ready frame for any dataset length.

    Steps: verify columns, drop rows after the last published target price,
    check weekly spacing, forward-fill bounded trailing gaps, and validate that
    every multivariate feature is fully populated. The number of forward-filled
    trailing weeks per column is stored in ``frame.attrs["imputed_weeks"]``.
    """
    if weekly_data is None:
        weekly_data = load_dataset()

    if config.TARGET_COL not in weekly_data.columns:
        raise ValueError(f"Target column '{config.TARGET_COL}' not found in dataset.")
    feature_cols = [column for column, _ in config.MULTIVARIATE_FEATURES]
    missing_cols = [column for column in feature_cols if column not in weekly_data.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required column(s): {missing_cols}")

    weekly_data = _trim_unpublished_tail(weekly_data)
    if weekly_data[config.TARGET_COL].isna().any():
        raise ValueError("Target column has missing values inside the series; "
                         "fill or remove those weeks before training.")

    columns_to_keep = [config.DATE_COL] + list(dict.fromkeys(feature_cols))
    weekly_data = weekly_data[columns_to_keep].copy()
    _check_weekly_spacing(weekly_data[config.DATE_COL])

    imputed_weeks: dict[str, int] = {}
    for column in config.FORWARD_FILL_COLS:
        if column not in weekly_data.columns:
            continue
        is_missing = weekly_data[column].isna().to_numpy()
        trailing_missing = 0
        while trailing_missing < len(is_missing) and is_missing[len(is_missing) - 1 - trailing_missing]:
            trailing_missing += 1
        if trailing_missing > config.MAX_TRAILING_FFILL_WEEKS:
            raise ValueError(
                f"'{column}' has {trailing_missing} missing trailing weeks; the limit for "
                f"forward-filling is {config.MAX_TRAILING_FFILL_WEEKS}. Add the "
                "real values or raise config.MAX_TRAILING_FFILL_WEEKS knowingly."
            )
        if trailing_missing:
            imputed_weeks[column] = trailing_missing
        weekly_data[column] = weekly_data[column].ffill()

    # Leading NaNs (nothing earlier to carry forward) are filled from the first
    # observed value. Only the leading stretch is touched: back-filling an interior
    # gap would copy a FUTURE value into the past, which is look-ahead leakage.
    for column in feature_cols:
        first_valid_row = weekly_data[column].first_valid_index()
        if first_valid_row is not None and first_valid_row > 0:
            weekly_data.loc[: first_valid_row - 1, column] = weekly_data.loc[first_valid_row, column]

    still_missing = {
        column: int(weekly_data[column].isna().sum())
        for column in feature_cols
        if weekly_data[column].isna().any()
    }
    if still_missing:
        raise ValueError(
            f"Features still contain missing values after imputation: {still_missing}. "
            "Fill these columns in the master dataset before training "
            "(only columns in config.FORWARD_FILL_COLS are carried forward)."
        )

    weekly_data = weekly_data.reset_index(drop=True)
    weekly_data.attrs["imputed_weeks"] = imputed_weeks
    return weekly_data


if __name__ == "__main__":
    prepared_data = prepare()
    print(f"Prepared {len(prepared_data)} rows: "
          f"{prepared_data[config.DATE_COL].iloc[0].date()} -> "
          f"{prepared_data[config.DATE_COL].iloc[-1].date()}")
    print("Columns:", list(prepared_data.columns))

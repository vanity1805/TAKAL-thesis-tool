"""Data loading, imputation, and validation for the TAKAL pipeline.

Loads the master dataset, verifies the target column, forward-fills the trailing
gaps (thesis-sanctioned imputation), and validates that every multivariate
feature is fully populated before modelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_dataset() -> pd.DataFrame:
    """Load the Clean Table sheet, sorted chronologically with a datetime index column."""
    df = pd.read_excel(config.DATASET_PATH, sheet_name=config.DATASET_SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    if config.DATE_COL not in df.columns:
        raise ValueError(f"Expected a '{config.DATE_COL}' column; found {list(df.columns)}")
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    df = df.sort_values(config.DATE_COL).reset_index(drop=True)
    return df


def _trim_unpublished_tail(df: pd.DataFrame) -> pd.DataFrame:
    """Drop trailing rows whose target price has not been published yet.

    New weeks are often typed in before the price is known. Those rows cannot be
    used as targets, so they are removed from the end (never from the middle).
    """
    valid = df[config.TARGET_COL].notna().to_numpy()
    if not valid.any():
        raise ValueError("Target column has no values; cannot train.")
    last_valid = int(np.flatnonzero(valid)[-1])
    dropped = len(df) - 1 - last_valid
    if dropped:
        print(f"[data] Ignoring {dropped} trailing row(s) with no target price yet.")
    return df.iloc[: last_valid + 1].copy()


def _check_weekly_spacing(dates: pd.Series) -> None:
    if dates.duplicated().any():
        dup = dates[dates.duplicated()].dt.date.tolist()[:5]
        raise ValueError(f"Duplicate dates in the dataset: {dup}")
    gaps = dates.diff().dt.days.iloc[1:]
    bad = gaps[gaps != config.EXPECTED_WEEKLY_STEP_DAYS]
    if len(bad):
        print(f"[data] WARNING: {len(bad)} row(s) are not exactly "
              f"{config.EXPECTED_WEEKLY_STEP_DAYS} days after the previous one "
              f"(first at {dates.loc[bad.index[0]].date()}). Windows assume weekly steps.")


def prepare(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a modelling-ready frame for any dataset length.

    Steps: verify columns, drop rows after the last published target price,
    check weekly spacing, forward-fill bounded trailing gaps, and validate that
    every multivariate feature is fully populated. The number of forward-filled
    trailing weeks per column is stored in ``frame.attrs["imputed_weeks"]``.
    """
    if df is None:
        df = load_dataset()

    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COL}' not found in dataset.")
    feature_cols = [col for col, _ in config.MULTIVARIATE_FEATURES]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required column(s): {missing_cols}")

    df = _trim_unpublished_tail(df)
    if df[config.TARGET_COL].isna().any():
        raise ValueError("Target column has missing values inside the series; "
                         "fill or remove those weeks before training.")

    keep = [config.DATE_COL] + list(dict.fromkeys(feature_cols))
    df = df[keep].copy()
    _check_weekly_spacing(df[config.DATE_COL])

    imputed: dict[str, int] = {}
    for col in config.FORWARD_FILL_COLS:
        if col not in df.columns:
            continue
        isna = df[col].isna().to_numpy()
        trailing = 0
        while trailing < len(isna) and isna[len(isna) - 1 - trailing]:
            trailing += 1
        if trailing > config.MAX_TRAILING_FFILL_WEEKS:
            raise ValueError(
                f"'{col}' has {trailing} missing trailing weeks; the limit for "
                f"forward-filling is {config.MAX_TRAILING_FFILL_WEEKS}. Add the "
                "real values or raise config.MAX_TRAILING_FFILL_WEEKS knowingly."
            )
        if trailing:
            imputed[col] = trailing
        df[col] = df[col].ffill()

    # Leading NaNs (nothing earlier to carry forward) are filled from the first
    # observed value. Only the leading stretch is touched: back-filling an interior
    # gap would copy a FUTURE value into the past, which is look-ahead leakage.
    for col in feature_cols:
        first = df[col].first_valid_index()
        if first is not None and first > 0:
            df.loc[: first - 1, col] = df.loc[first, col]

    missing = {c: int(df[c].isna().sum()) for c in feature_cols if df[c].isna().any()}
    if missing:
        raise ValueError(
            f"Features still contain missing values after imputation: {missing}. "
            "Fill these columns in the master dataset before training "
            "(only columns in config.FORWARD_FILL_COLS are carried forward)."
        )

    df = df.reset_index(drop=True)
    df.attrs["imputed_weeks"] = imputed
    return df


if __name__ == "__main__":
    frame = prepare()
    print(f"Prepared {len(frame)} rows: "
          f"{frame[config.DATE_COL].iloc[0].date()} -> {frame[config.DATE_COL].iloc[-1].date()}")
    print("Columns:", list(frame.columns))

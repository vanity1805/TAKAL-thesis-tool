"""Data loading, imputation, and validation for the TAKAL pipeline.

Loads the master dataset, verifies the target column, forward-fills the trailing
gaps (thesis-sanctioned imputation), and validates that every multivariate
feature is fully populated before modelling.
"""

from __future__ import annotations

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


def prepare(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a modelling-ready frame: target verified, trailing gaps forward-filled,
    and all multivariate features validated as fully populated."""
    if df is None:
        df = load_dataset()

    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COL}' not found in dataset.")
    if df[config.TARGET_COL].isna().any():
        raise ValueError("Target column contains missing values; cannot train.")

    feature_cols = [col for col, _ in config.MULTIVARIATE_FEATURES]
    keep = [config.DATE_COL] + list(dict.fromkeys(feature_cols))
    df = df[keep].copy()

    # Forward-fill trailing/interior gaps (e.g. the 9 trailing rice_import_mt weeks).
    for col in config.FORWARD_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Any remaining leading NaNs (nothing to carry forward from) are back-filled.
    df[feature_cols] = df[feature_cols].bfill()

    missing = {c: int(df[c].isna().sum()) for c in feature_cols if df[c].isna().any()}
    if missing:
        raise ValueError(
            f"Features still contain missing values after imputation: {missing}. "
            "Fill these columns in the master dataset before training."
        )

    return df.reset_index(drop=True)


if __name__ == "__main__":
    frame = prepare()
    print(f"Prepared {len(frame)} rows: "
          f"{frame[config.DATE_COL].iloc[0].date()} -> {frame[config.DATE_COL].iloc[-1].date()}")
    print("Columns:", list(frame.columns))

"""Central configuration for the TAKAL rice-price forecasting pipeline.

All paths, feature definitions, and hyperparameters live here so the rest of the
pipeline stays declarative. Values follow the thesis methodology (Stacked LSTM,
Min-Max normalization, 80/20 chronological split, light tuning of the lookback L).
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT, "THESIS-MASTER_DATASET.xlsx")
DATASET_SHEET = "Clean Table"
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
RESULTS_JSON = os.path.join(ARTIFACTS_DIR, "results.json")

# --------------------------------------------------------------------------- #
# Columns (dataset uses lowercase names; target column keeps its original name)
# --------------------------------------------------------------------------- #
DATE_COL = "date"
TARGET_COL = "rice_price_php_kg average"

# Multivariate feature set: (dataset column, dashboard display name).
# "Retail Rice Price Lag" is the target price itself used as an autoregressive
# input feature (its lagged values live in the lookback window).
MULTIVARIATE_FEATURES: list[tuple[str, str]] = [
    ("diesel_price_php_liter", "Diesel Fuel Price"),
    ("usdphp_exchange_rate", "USD/PHP Exchange Rate"),
    ("rice_stockpile_mt", "Rice Stockpile Inventory"),
    ("rice_import_mt", "Rice Import Volume"),
    (TARGET_COL, "Retail Rice Price Lag"),
    ("palay_production_mt", "Palay Production"),
]

# Columns that require forward-fill imputation of trailing gaps (user-confirmed,
# thesis-sanctioned). rice_import_mt has 9 missing weeks at the end of the series.
FORWARD_FILL_COLS = ["rice_import_mt", "usdphp_exchange_rate", "diesel_price_php_liter"]

# --------------------------------------------------------------------------- #
# Split / windowing
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.20            # chronological hold-out (~66 of 330 weeks)
VAL_FRACTION = 0.15         # last fraction of valid 12-target windows, before an 11-window purge
L_GRID = [4, 8, 12, 16]    # candidate lookback windows (lightly tuned)
FORECAST_HORIZON = 12

# --------------------------------------------------------------------------- #
# Stacked LSTM hyperparameters (fixed per thesis; only L is tuned)
# --------------------------------------------------------------------------- #
LSTM_UNITS = [64, 32, 16]   # three stacked LSTM layers
DROPOUT_RATE = 0.2          # applied after the first two LSTM layers
LEARNING_RATE = 1e-3        # Adam
MAX_EPOCHS = 300
BATCH_SIZE = 16
EARLY_STOPPING_PATIENCE = 20

SEED = 42
ALPHA = 0.05                # significance level for the Diebold-Mariano test

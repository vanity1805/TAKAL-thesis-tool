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
TEST_SIZE = 0.20            # chronological hold-out: the last 20% of rows, whatever the dataset size
VAL_FRACTION = 0.15         # last fraction of valid 12-target windows, before an 11-window purge
# Candidate lookback windows (weeks). Short windows are included on purpose: with
# only ~190 training windows and a 2023 price spike plus a 2026 diesel spike in the
# data, long windows (16-26 weeks) mostly feed the LSTM stale shock history.
# A grid whose best value sits on its edge is under-explored, so the grid is
# extended downward from the original [4, 8, 12, 16].
L_GRID = [2, 3, 4, 6, 8, 12]

# Lookback selection is repeated over several seeds and the validation losses are
# averaged, because a single LSTM run is noisy (see model/seed_robustness.py).
# L is chosen on the mean of the multivariate AND univariate validation losses so
# the shared L does not favour either side of the experiment.
SELECTION_SEEDS = [42, 7, 123]

# Shock guard: clamp the *scaled* inputs to the training range [0, 1]. Min-Max
# scalers are fit on training rows only, so a test-period value outside the
# training range (e.g. diesel at 150 PHP/L vs a training max of ~90) would
# otherwise reach the network as a scaled value of ~2, which the LSTM never saw.
CLIP_INPUTS_TO_TRAIN_RANGE = False
FORECAST_HORIZON = 12

# Gradient samples per SHAP value for the single forward-looking input window (the
# test-set explanations keep SHAP's default of 200). One window is cheap to explain
# precisely, and more samples shrink the approximation error that
# explain._complete_to_sum would otherwise have to redistribute (~0.6 -> ~0.06 PHP/kg).
FUTURE_SHAP_SAMPLES = 5000

# --------------------------------------------------------------------------- #
# Growing-dataset guards (new weeks can be appended to the Excel sheet)
# --------------------------------------------------------------------------- #
# A feature that has not been published yet for the newest weeks may be carried
# forward from its last value, but only for a limited stretch so stale data cannot
# quietly feed the model. Beyond this many trailing weeks the pipeline stops.
MAX_TRAILING_FFILL_WEEKS = 26
EXPECTED_WEEKLY_STEP_DAYS = 7   # rows must be one week apart (gaps raise a warning)
MIN_TEST_WEEKS = 30             # refuse to evaluate on fewer test weeks than this

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

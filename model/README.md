# TAKAL model pipeline

Offline Python pipeline that trains the thesis **Stacked LSTM** forecasting
models on `THESIS-MASTER_DATASET.xlsx`, evaluates them, computes SHAP
explanations, and writes a single `artifacts/results.json` that the Streamlit
dashboard (`../app.py`) loads.

## What it produces

`artifacts/results.json` with:

- **forecast** — actual vs. multivariate vs. univariate prices for each of the
  12 forecast horizons, all aligned to the same 66 held-out target weeks.
- **metrics** — RMSE, MAE, MAPE, R² for each model and each horizon, plus an
  overall row pooling all 12 × 66 errors.
- **dm_test** — horizon-specific h-step Diebold-Mariano tests and an overall
  test of the per-week mean squared-error loss differential.
- **shap** — horizon-specific global bar (mean|SHAP|), beeswarm, per-week local
  waterfall contributions, and feature dependence data for both models.

## Methodology (from the thesis)

- **Model:** three stacked LSTM layers, Dropout 0.2 after the first two, Dense(12),
  Adam optimizer, MSE loss, Early Stopping. See `config.py`.
- **Two variants:** multivariate (6 economic features) and univariate (price only).
- **Features (multivariate):** Diesel Fuel Price, USD/PHP Exchange Rate, Rice
  Stockpile Inventory, Rice Import Volume, Retail Rice Price Lag, Palay Production.
- **Preprocessing:** Min-Max normalization fit on the training set only; each
  sliding lookback window predicts the next 12 prices. Training and validation
  targets stay before the test split, with an 11-window purge between them.
- **Split:** 80/20 chronological (no shuffle); ~66-week hold-out test set.
- **Lookback L:** lightly tuned over `config.L_GRID` (chosen by lowest validation
  loss); the chosen value and the full grid are recorded in the results file
  (`meta.lookback_L`, `meta.lookback_grid_val_loss`). L is kept constant across
  both models (controlled experiment).
- **SHAP:** `GradientExplainer` on a scalar-output wrapper for each horizon;
  multivariate attributions are aggregated across the lookback window into one
  value per feature ("Vector SHAP"), while the univariate model also reports
  per-lag explanations and an aggregate price attribution.

## Data handling

- `rice_import_mt`: 9 trailing weeks are **forward-filled** with the last observed
  value (5049 MT), per the thesis's stated imputation method.
- All six multivariate feature columns must be populated; the pipeline raises a
  clear error otherwise.

## Setup

Use Python 3.11–3.13.

```powershell
pip install -r model/requirements.txt   # modelling deps (tensorflow, shap, ...)
pip install -r requirements.txt          # dashboard deps (streamlit, plotly)
```

## Run

From the `FINAL TOOL` folder (the parent of `model/`):

```powershell
python -m model.run_all      # trains, evaluates, explains, writes artifacts/results.json
streamlit run app.py         # view the dashboard with the real results
```

Re-run `python -m model.run_all` whenever the dataset changes; the dashboard
picks up the new `results.json` automatically (its cache is keyed on the file's
modification time).

## Adding new weeks of data

The pipeline is size-agnostic: nothing depends on the current 330 rows.

1. Append the new weekly rows to the `Clean Table` sheet of `THESIS-MASTER_DATASET.xlsx`
   (same column names, one row per week, dates 7 days apart).
2. Run `python -m model.run_all`, then `streamlit run app.py`.

What adjusts automatically: the 80/20 chronological split, the number of training,
validation and test windows, the scalers (refit on the new training rows), the
test-week count shown in the dashboard, and the data note in `meta.data_note`.

Built-in guards (see `config.py`):

- Trailing rows with no target price yet are ignored, not an error.
- A feature missing for the newest weeks is forward-filled only if it is listed in
  `FORWARD_FILL_COLS`, and only up to `MAX_TRAILING_FFILL_WEEKS`; other gaps stop the run.
- Duplicate dates stop the run; non-weekly spacing prints a warning.
- Fewer than `MIN_TEST_WEEKS` test weeks stops the run.
- Test-period inputs outside the training range are listed in the data note.

Note that appending data moves the 80/20 cutoff, so the test period (and every
metric) changes. Fix the data cut-off date and sources before running, and report
what comes out. Update the study period and split sizes quoted in the paper.

## Module map

| File                 | Responsibility                                             |
|----------------------|-----------------------------------------------------------|
| `config.py`          | Paths, feature list, split ratios, LSTM hyperparameters.  |
| `data_prep.py`       | Load dataset, verify target, forward-fill, validate.      |
| `preprocessing.py`   | Min-Max scaling, windowing, chronological split.          |
| `train.py`           | Build/train Stacked LSTM; light lookback (L) tuning.      |
| `evaluate.py`        | RMSE/MAE/MAPE/R² and the Diebold-Mariano test.            |
| `explain.py`         | SHAP global/local/dependence outputs.                     |
| `export_results.py`  | Assemble everything into `artifacts/results.json`.        |
| `run_all.py`         | One-command end-to-end runner.                            |

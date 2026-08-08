# TAKAL model pipeline

Offline Python pipeline that trains the thesis **Stacked LSTM** forecasting
models on `THESIS-MASTER_DATASET.xlsx`, evaluates them, computes SHAP
explanations, and writes a single `artifacts/results.json` that the Streamlit
dashboard (`../app.py`) loads.

## What it produces

`artifacts/results.json` with:

- **forecast** — actual vs. multivariate vs. univariate prices over the held-out
  test horizon (~66 weeks), plus the week dates.
- **metrics** — RMSE, MAE, MAPE, R² per model.
- **dm_test** — Diebold-Mariano statistic, p-value, and verdict.
- **shap** — global bar (mean|SHAP|), beeswarm, per-week local waterfall
  contributions, and feature dependence data for both models.

## Methodology (from the thesis)

- **Model:** three stacked LSTM layers, Dropout 0.2 after the first two, Dense(1),
  Adam optimizer, MSE loss, Early Stopping. See `config.py`.
- **Two variants:** multivariate (6 economic features) and univariate (price only).
- **Features (multivariate):** Diesel Fuel Price, USD/PHP Exchange Rate, Rice
  Stockpile Inventory, Rice Import Volume, Retail Rice Price Lag, Palay Production.
- **Preprocessing:** Min-Max normalization fit on the training set only; sliding
  lookback window.
- **Split:** 80/20 chronological (no shuffle); ~66-week hold-out test set.
- **Lookback L:** lightly tuned over `config.L_GRID` (chosen by lowest validation
  loss); the chosen value and the full grid are recorded in the results file
  (`meta.lookback_L`, `meta.lookback_grid_val_loss`). L is kept constant across
  both models (controlled experiment).
- **SHAP:** `GradientExplainer`; multivariate attributions are aggregated across
  the lookback window into one value per feature ("Vector SHAP"), while the
  univariate model keeps each lag (Price(t-1) … Price(t-L)) as its own player.

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

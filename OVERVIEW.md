# TAKAL — How, What & Why

**Time-Series Analysis & Knowledge Attribution for Local Rice (TAKAL)** is a tool
that **forecasts weekly retail rice prices in Metro Manila (NCR) and explains
*why* each prediction was made** — turning a "black-box" AI model into a
transparent decision-support instrument for food-security planning.

> One line: *TAKAL predicts next-week rice prices from economic indicators and
> shows which forces (fuel, exchange rate, stockpiles, imports, harvests, and
> recent prices) drove each forecast.*

---

## What it is

TAKAL has two parts working together:

1. **A forecasting engine** (offline Python pipeline) that learns from six years
   of weekly data and predicts the retail price of Regular-Milled rice.
2. **An interactive dashboard** (the Streamlit app you open in a browser) that
   displays those forecasts and their explanations.

### The two models
- **Multivariate model** *(the main model)* — uses six economic indicators to
  forecast the price.
- **Univariate model** *(the baseline)* — uses only the rice price's own recent
  history. It exists so we can prove the economic indicators actually help.

### What the dashboard shows — three tabs
| Tab | What you see |
|-----|--------------|
| **📈 Price Forecast** | Predicted vs. actual prices across the held-out test period, with quick summary cards. |
| **📊 Benchmarking** | Accuracy scores (RMSE, MAE, MAPE, R²), a model-comparison table, and a statistical test proving which model is better. |
| **💡 Explainability** | Which factors matter most (SHAP), a per-week breakdown of *why* a given price was predicted, and how each factor influences the forecast. |

### Inputs and outputs
- **Input:** `THESIS-MASTER_DATASET.xlsx` — 330 weekly rows (Jan 2020 → Apr 2026)
  of rice prices and economic indicators.
- **Output:** `artifacts/results.json` — the predictions, accuracy scores, and
  explanations that the dashboard reads.

---

## Why it matters

**Rice is a staple, and its price swings hurt people.** For low-income households
whose rice consumption barely changes with price, sudden increases force painful
budget trade-offs. Anticipating price movements helps agencies and planners react
before a spike becomes a crisis.

**Why use economic indicators (multivariate) instead of price history alone?**
Rice prices don't move in a vacuum — fuel costs, the peso's exchange rate,
harvest volumes, imports, and government stockpiles all push and pull them. The
multivariate model captures these forces; the univariate baseline shows how much
accuracy you *gain* by including them. In this build, the multivariate model is
**statistically significantly more accurate** (see results below).

**Why explainability (SHAP)?** A deep-learning model like an LSTM is a
"black box" — it can predict a price spike without telling you *why*. For
public-policy decisions, that is not enough. TAKAL uses **SHAP** to attribute
each forecast to specific drivers, so a stakeholder can see, for example, that a
predicted increase was driven mostly by the recent price trend and the exchange
rate. This transforms an opaque forecast into an **auditable, trustworthy**
recommendation.

---

## How it works

### The big picture
```
THESIS-MASTER_DATASET.xlsx
        │   (offline pipeline: the model/ folder)
        ▼
  clean → normalize → window → train two LSTMs → evaluate → explain (SHAP)
        │
        ▼   writes one file
   artifacts/results.json
        │   (loaded when the app starts)
        ▼
   app.py  →  the dashboard you view in a browser
```

The forecasting runs **once, offline** (it does not predict live). Whenever the
dataset changes, you re-run the pipeline and the dashboard automatically picks up
the new results.

### How to run it
From the `FINAL TOOL` folder:

```powershell
# 1. Install dependencies (first time only)
pip install -r requirements.txt          # dashboard: streamlit, plotly
pip install -r model/requirements.txt    # modelling: tensorflow, shap, ...

# 2. Train the models and generate results
python -m model.run_all

# 3. Open the dashboard
streamlit run app.py
```

`python -m model.run_all` trains both models, evaluates them, computes the SHAP
explanations, and writes `artifacts/results.json`. It finishes in well under a
minute on a normal laptop (CPU only).

---

## Technical details

*This section is for developers and reviewers who want the methodology and code
layout. General readers can stop above.*

### Methodology (from the thesis, Chapter 3)
- **Model — Stacked LSTM:** three sequential LSTM layers (64 → 32 → 16 units),
  **Dropout 0.2** after the first two layers, a Dense(1) output, **Adam**
  optimizer, **MSE** loss, and **Early Stopping** on validation loss.
- **Target:** `rice_price_php_kg average` (the mean of local and imported price).
- **Multivariate features (6):** Diesel Fuel Price, USD/PHP Exchange Rate, Rice
  Stockpile Inventory, Rice Import Volume, Retail Rice Price Lag (the price's own
  recent values), and Palay Production. The univariate model uses the price only.
- **Preprocessing:** **Min-Max normalization** fit on the **training set only**
  (no look-ahead), then a sliding **lookback window** turns the series into
  supervised `(samples, timesteps, features)` tensors.
- **Split:** **80/20 chronological** (no shuffling) — the last ~66 weeks are the
  held-out test set.
- **Lookback window `L`:** *lightly tuned* over `{4, 8, 12, 16}`; the value with
  the lowest validation loss is chosen and used for **both** models (kept constant
  for a controlled comparison). The grid and choice are recorded in the results
  file (`meta.lookback_L`, `meta.lookback_grid_val_loss`).
- **Metrics:** RMSE, MAE, MAPE, R² on the inverse-transformed (PHP/kg) test set.
- **Significance test:** **Diebold-Mariano** compares the two models' forecast
  errors to confirm the difference is statistically significant.
- **Explainability — SHAP (`GradientExplainer`):** for the multivariate model,
  per-timestep attributions are aggregated across the lookback window into one
  value per feature ("**Vector SHAP**"); the univariate model keeps each lag
  (Price(t-1) … Price(t-L)) as its own contributor. SHAP values are converted
  from the model's normalized output to **PHP/kg** so the charts are readable.

### Data handling
- `rice_import_mt` has 9 missing trailing weeks (Mar–Apr 2026); these are
  **forward-filled** with the last observed value (5049 MT), matching the thesis's
  stated imputation method. These weeks fall in the test set, so model *training*
  is unaffected.
- All six feature columns must be populated or the pipeline stops with a clear
  error.

### Results (current run)
| Model | RMSE | MAE | MAPE | R² |
|-------|-----:|----:|-----:|---:|
| **Multivariate** | **0.64** | **0.51** | **1.27 %** | **0.86** |
| Univariate (baseline) | 0.82 | 0.56 | 1.37 % | 0.76 |

- **Chosen lookback `L` = 4**; test horizon = 66 weeks (2025-01-27 → 2026-04-27).
- **Diebold-Mariano:** statistic = −2.05, **p = 0.044 (< 0.05)** → the multivariate
  model is **significantly more accurate** than the baseline.
- **Top SHAP drivers (multivariate):** Retail Rice Price Lag, then USD/PHP
  Exchange Rate and Rice Stockpile Inventory.

*(Values come from `artifacts/results.json`; re-running the pipeline regenerates
them.)*

### Code layout (`model/`)
| File | Responsibility |
|------|----------------|
| `config.py` | Paths, feature list, split ratios, LSTM hyperparameters. |
| `data_prep.py` | Load dataset, verify target, forward-fill, validate. |
| `preprocessing.py` | Min-Max scaling, windowing, chronological split. |
| `train.py` | Build/train the Stacked LSTM; light lookback (`L`) tuning. |
| `evaluate.py` | RMSE/MAE/MAPE/R² and the Diebold-Mariano test. |
| `explain.py` | SHAP global / local / dependence outputs. |
| `export_results.py` | Assemble everything into `artifacts/results.json`. |
| `run_all.py` | One-command end-to-end runner. |

`app.py` (the dashboard) loads `results.json` through a cached loader keyed on the
file's modification time — so re-running the pipeline refreshes the UI with no
code changes.

---

## Limitations & notes
- **Small dataset (~330 weekly rows).** Deep models can overfit limited data; the
  architecture mitigates this with dropout and early stopping, but accuracy should
  be read with that in mind.
- **Offline, not live.** TAKAL forecasts from a pre-compiled dataset; it does not
  ingest real-time feeds. New data requires re-running `python -m model.run_all`.
- **Imputed imports.** The final 9 weeks of import volume are forward-filled (see
  above).
- **Data-quality caveat.** Recent diesel prices (Mar–Apr 2026, up to ~₱154/L) are
  unusually high versus a typical ₱50–70/L range — worth verifying at the source.

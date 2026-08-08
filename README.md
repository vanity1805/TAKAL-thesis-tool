# Time-Series Analysis & Knowledge Attribution for Local Rice (TAKAL)

An interactive **Streamlit** dashboard that visualizes the outputs of two Stacked
LSTM models (multivariate vs. univariate) for forecasting weekly Regular-Milled
retail rice prices in the National Capital Region (NCR), with SHAP-based
explainability.

## Modules
- **Price Forecast** — 12-week forecast vs. actual prices, with summary cards.
- **Benchmarking** — RMSE / MAE / MAPE / R² metrics, comparison table, and the
  Diebold-Mariano significance test.
- **Explainability** — global SHAP (bar / beeswarm), a local SHAP waterfall
  per forecast week, and a feature dependence plot.

A global model selector in the header switches all visualizations between the
multivariate and univariate models.

## Run locally
```bash
pip install -r requirements.txt          # dashboard deps (streamlit, plotly)
pip install -r model/requirements.txt    # modelling deps (tensorflow, shap, ...)

python -m model.run_all                  # train models + write artifacts/results.json
streamlit run app.py                     # view the dashboard
```

## Data source
The dashboard renders **real** Stacked LSTM outputs loaded from
`artifacts/results.json`, produced by the offline pipeline in `model/`
(see `model/README.md`). If that file is missing, `app.py` shows an error
prompting you to run `python -m model.run_all` first. Re-run the pipeline
whenever `THESIS-MASTER_DATASET.xlsx` changes; the dashboard refreshes
automatically.


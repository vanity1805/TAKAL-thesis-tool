"""TAKAL — Time-Series Analysis & Knowledge Attribution for Local Rice.

Offline modelling pipeline that trains the thesis Stacked LSTM (multivariate vs.
univariate), evaluates it, computes SHAP explanations, and exports a single
``artifacts/results.json`` consumed by the Streamlit dashboard (``app.py``).
"""

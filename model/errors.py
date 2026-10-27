# =============================================================================
# Program      : TAKAL pipeline error types (model/errors.py)
# Description  : Small exception hierarchy shared by the TAKAL modelling
#                pipeline, plus a context manager that labels crashes with the
#                pipeline stage in which they happened.
# Author       : TAKAL Thesis Team (contact: chavezearljustineb@gmail.com)
# System Fit   : Lowest layer of the offline pipeline (python -m model.run_all).
#                data_prep, predict_future, explain and export_results raise
#                these errors; run_all.py catches them and writes
#                artifacts/pipeline_status.json, which the Streamlit dashboard
#                (app.py) turns into a formatted on-screen message.
# Date Written : 2026
# Date Revised : 2026-10-26
# Purpose      : Give every expected failure (dataset cannot be loaded, Stacked
#                LSTM inference fails, SHAP attribution fails) a precise type, a
#                human-readable message, and a hint about how to fix it.
# Algorithms   : None. The stage context manager is a plain try/except wrapper.
# Data         : Exception objects carry three string attributes
# Structures     (message, stage, hint). Nothing else is stored.
# =============================================================================
"""Exception types and the stage-labelling helper for the TAKAL pipeline."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator


class TakalPipelineError(Exception):
    """Base class for every expected, explainable pipeline failure."""

    default_stage = "Pipeline"

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage or self.default_stage
        self.hint = hint


class DatasetLoadError(TakalPipelineError):
    """The NCR rice price workbook is missing, unreadable, or fails validation."""

    default_stage = "Loading the NCR rice price dataset"


class ForecastInferenceError(TakalPipelineError):
    """The 3-layer Stacked LSTM could not produce a valid forecast."""

    default_stage = "Stacked LSTM inference"


class ShapAttributionError(TakalPipelineError):
    """SHAP could not attribute a forecast to its input features."""

    default_stage = "SHAP attribution"


@contextlib.contextmanager
def pipeline_stage(stage_name: str) -> Iterator[None]:
    """Label any failure inside the ``with`` block with ``stage_name``.

    Errors that are already ``TakalPipelineError`` keep their own, more specific
    stage. Any other exception is re-raised as a ``TakalPipelineError`` whose
    ``__cause__`` is the original exception, so no traceback detail is lost.
    ``KeyboardInterrupt`` and ``SystemExit`` are deliberately not intercepted.
    """
    try:
        yield
    except TakalPipelineError:
        raise
    except Exception as error:
        raise TakalPipelineError(
            f"{type(error).__name__}: {error}",
            stage=stage_name,
        ) from error

"""Deterministic forecast error measures.

Errors are forecast minus actual, so a positive bias means over-forecasting.
Hours where either value is missing are left out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCORE_COLUMNS = ("hours", "mae_kw", "rmse_kw", "mbe_kw", "nmae_pct", "nrmse_pct", "skill")
NOT_MODELS = ("actual", "daylight")


def _errors(forecast, actual):
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if forecast.shape != actual.shape:
        raise ValueError("forecast and actual must have the same shape")
    paired = ~(np.isnan(forecast) | np.isnan(actual))
    return forecast[paired] - actual[paired]


def mae(forecast, actual) -> float:
    errors = _errors(forecast, actual)
    return float(np.abs(errors).mean()) if errors.size else float("nan")


def rmse(forecast, actual) -> float:
    errors = _errors(forecast, actual)
    return float(np.sqrt((errors**2).mean())) if errors.size else float("nan")


def mbe(forecast, actual) -> float:
    errors = _errors(forecast, actual)
    return float(errors.mean()) if errors.size else float("nan")


def skill_score(forecast, reference, actual) -> float:
    """1 - RMSE(forecast) / RMSE(reference), on the hours where all three exist."""
    forecast = np.asarray(forecast, dtype=float)
    reference = np.asarray(reference, dtype=float)
    actual = np.asarray(actual, dtype=float)
    common = ~(np.isnan(forecast) | np.isnan(reference) | np.isnan(actual))
    base = rmse(reference[common], actual[common])
    if not base > 0:
        return float("nan")
    return 1.0 - rmse(forecast[common], actual[common]) / base


def score_table(forecasts: pd.DataFrame, capacity_kw: float, reference: str = "persistence", daylight_only: bool = True) -> pd.DataFrame:
    """One row of scores per model in a backtest result."""
    models = [name for name in forecasts.columns if name not in NOT_MODELS]
    if reference not in models:
        raise ValueError(f"reference model {reference!r} is not in the backtest")
    frame = forecasts[forecasts["daylight"].astype(bool)] if daylight_only else forecasts
    actual = frame["actual"].to_numpy(dtype=float)
    rows = {}
    for name in models:
        values = frame[name].to_numpy(dtype=float)
        rows[name] = {
            "hours": int((~(np.isnan(values) | np.isnan(actual))).sum()),
            "mae_kw": mae(values, actual),
            "rmse_kw": rmse(values, actual),
            "mbe_kw": mbe(values, actual),
            "nmae_pct": 100.0 * mae(values, actual) / capacity_kw,
            "nrmse_pct": 100.0 * rmse(values, actual) / capacity_kw,
            "skill": skill_score(values, frame[reference].to_numpy(dtype=float), actual),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")[list(SCORE_COLUMNS)]
    table.index.name = "model"
    return table

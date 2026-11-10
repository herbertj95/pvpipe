"""Rolling-origin evaluation of day-ahead forecasts."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

WINDOWS = ("expanding", "rolling")


@dataclass(frozen=True)
class BacktestConfig:
    train_days: int = 30  # history required before the first forecast day
    retrain_every: int = 7  # days between refits
    window: str = "expanding"
    rolling_days: int = 90  # length of the training window when window == "rolling"

    def __post_init__(self):
        if self.train_days < 1:
            raise ValueError("train_days must be at least 1")
        if self.retrain_every < 1:
            raise ValueError("retrain_every must be at least 1")
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {', '.join(WINDOWS)}")
        if self.rolling_days < 1:
            raise ValueError("rolling_days must be at least 1")


def training_rows(features: pd.DataFrame, issue_time: pd.Timestamp, config: BacktestConfig) -> pd.Series:
    """Rows whose measurement was known when the forecast was issued."""
    known = features.index < issue_time
    if config.window == "rolling":
        known &= features.index >= issue_time - pd.Timedelta(days=config.rolling_days)
    return pd.Series(known, index=features.index)


def run_backtest(features: pd.DataFrame, target: pd.Series, models: dict, config: BacktestConfig | None = None) -> pd.DataFrame:
    """Forecast every day after the first ``train_days`` and return all forecasts.

    The result has one column per model plus ``actual`` and ``daylight``.
    Models are refitted every ``retrain_every`` days on the rows known at the
    issue time of the day being forecast.
    """
    config = config or BacktestConfig()
    target = target.reindex(features.index)
    days = features.index.normalize().unique()
    forecast_days = days[config.train_days :]
    if len(forecast_days) == 0:
        raise ValueError(f"need more than {config.train_days} days of data, got {len(days)}")
    by_day = features.index.normalize()
    pieces = []
    for position, day in enumerate(forecast_days):
        rows = features[by_day == day]
        if position % config.retrain_every == 0:
            mask = training_rows(features, rows["issue_time"].iloc[0], config).to_numpy()
            for model in models.values():
                model.fit(features[mask], target[mask])
        piece = pd.DataFrame({name: model.predict(rows) for name, model in models.items()}, index=rows.index)
        pieces.append(piece)
    forecasts = pd.concat(pieces)
    forecasts["actual"] = target.reindex(forecasts.index)
    forecasts["daylight"] = features["daylight"].reindex(forecasts.index)
    return forecasts

"""Inputs of the forecast models.

The pipeline makes day-ahead forecasts: the forecast for every hour of UTC day
D is issued on day D-1 at ``issue_hour`` UTC. A feature for target hour ``t``
may only use production measured before that issue time. The weather forecast
for ``t`` is assumed to be available at issue time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pvpipe.clearsky import clear_sky_frame
from pvpipe.plant import simulate_power
from pvpipe.solar import require_utc, solar_time_hours

DAY = pd.Timedelta(days=1)
MIN_CLEAR_FRACTION = 0.05  # below this share of capacity the clear-sky index is not defined
MAX_CLEAR_SKY_INDEX = 1.3
RECENT_DAYS = 3

FEATURE_COLUMNS = (
    "ghi_fc",
    "temp_fc",
    "cloud_fc",
    "ghi_clear",
    "cos_zenith",
    "kc_fc",
    "physical_kw",
    "power_clear_kw",
    "power_lag_kw",
    "kc_lag",
    "power_recent_kw",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "daylight",
    "issue_time",
)


def issue_times(index: pd.DatetimeIndex, issue_hour: int = 12, lead_days: int = 1) -> pd.DatetimeIndex:
    """When the forecast for each target hour is made."""
    index = require_utc(index)
    return index.normalize() - lead_days * DAY + pd.Timedelta(hours=issue_hour)


def lag_days(index: pd.DatetimeIndex, issue_hour: int = 12, lead_days: int = 1) -> np.ndarray:
    """Days back to the newest same-hour measurement known at issue time."""
    index = require_utc(index)
    return np.where(index.hour < issue_hour, lead_days, lead_days + 1)


def _shifted(series: pd.Series, index: pd.DatetimeIndex, days) -> np.ndarray:
    source = index - pd.to_timedelta(days, unit="D")
    return series.reindex(source).to_numpy()


def build_features(
    production: pd.Series,
    forecast: pd.DataFrame,
    site,
    issue_hour: int = 12,
    lead_days: int = 1,
) -> pd.DataFrame:
    """Feature table with one row per hour of the weather forecast."""
    if not 0 <= issue_hour <= 23:
        raise ValueError("issue_hour must be between 0 and 23")
    if lead_days < 1:
        raise ValueError("lead_days must be at least 1")
    index = require_utc(forecast.index)
    sky = clear_sky_frame(index, site)
    ghi_clear = sky["ghi_clear"].to_numpy()
    ghi_fc = np.clip(forecast["ghi"].to_numpy(dtype=float), 0.0, None)
    temp_fc = forecast["temp_air"].to_numpy(dtype=float)

    power_clear = simulate_power(index, ghi_clear, temp_fc, site)
    physical = simulate_power(index, ghi_fc, temp_fc, site)

    back = lag_days(index, issue_hour, lead_days)
    power_lag = _shifted(production, index, back)
    clear_lag = _shifted(power_clear, index, back)
    defined = clear_lag > MIN_CLEAR_FRACTION * site.capacity_kw
    with np.errstate(divide="ignore", invalid="ignore"):
        kc_lag = np.where(defined, power_lag / np.where(defined, clear_lag, 1.0), np.nan)
    kc_lag = np.clip(kc_lag, 0.0, MAX_CLEAR_SKY_INDEX)

    recent = np.vstack([_shifted(production, index, back + extra) for extra in range(RECENT_DAYS)])
    known = ~np.isnan(recent)
    counts = known.sum(axis=0)
    power_recent = np.where(counts > 0, np.where(known, recent, 0.0).sum(axis=0) / np.maximum(counts, 1), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        kc_fc = np.where(ghi_clear > 1.0, ghi_fc / np.where(ghi_clear > 1.0, ghi_clear, 1.0), 0.0)

    hour_angle = 2.0 * np.pi * solar_time_hours(index + pd.Timedelta(minutes=30), site.longitude) / 24.0
    year_angle = 2.0 * np.pi * sky["day_of_year"].to_numpy() / 365.0

    features = pd.DataFrame(
        {
            "ghi_fc": ghi_fc,
            "temp_fc": temp_fc,
            "cloud_fc": forecast["cloud_cover"].to_numpy(dtype=float),
            "ghi_clear": ghi_clear,
            "cos_zenith": np.clip(np.cos(np.radians(sky["zenith"].to_numpy())), 0.0, None),
            "kc_fc": np.clip(kc_fc, 0.0, MAX_CLEAR_SKY_INDEX),
            "physical_kw": physical.to_numpy(),
            "power_clear_kw": power_clear.to_numpy(),
            "power_lag_kw": power_lag,
            "kc_lag": kc_lag,
            "power_recent_kw": power_recent,
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "doy_sin": np.sin(year_angle),
            "doy_cos": np.cos(year_angle),
            "daylight": sky["elevation"].to_numpy() > 0.0,
            "issue_time": issue_times(index, issue_hour, lead_days),
        },
        index=index,
    )
    return features[list(FEATURE_COLUMNS)]

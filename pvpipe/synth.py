"""Synthetic weather and production for tests, examples and demos.

Nothing here is measured. Cloud cover is a smooth random process, irradiance
follows from the clear-sky model and the cloud cover, and production is the
plant model applied to that weather plus noise. The same seed always gives the
same series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pvpipe.clearsky import clear_sky_frame
from pvpipe.io import Dataset
from pvpipe.plant import simulate_power
from pvpipe.solar import solar_time_hours

CLOUD_PERSISTENCE = 0.97  # hour-to-hour correlation of the latent cloud process


def _ar1(rng, n, phi):
    values = np.empty(n)
    values[0] = rng.normal()
    shocks = rng.normal(size=n) * np.sqrt(1.0 - phi**2)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + shocks[i]
    return values


def cloud_to_clearness(cloud_cover):
    """Ratio of GHI to clear-sky GHI for a cloud fraction (Kasten and Czeplak, 1980)."""
    return 1.0 - 0.75 * np.clip(np.asarray(cloud_cover, dtype=float), 0.0, 1.0) ** 3.4


def make_weather(site, start, days: int, seed: int = 0) -> pd.DataFrame:
    """Hourly 'true' weather: ghi (W/m2), temp_air (degrees C), cloud_cover (0-1)."""
    if days <= 0:
        raise ValueError("days must be positive")
    first = pd.Timestamp(start)
    first = first.tz_localize("UTC") if first.tzinfo is None else first.tz_convert("UTC")
    index = pd.date_range(first.normalize(), periods=days * 24, freq="1h")
    rng = np.random.default_rng(seed)
    sky = clear_sky_frame(index, site)
    latent = _ar1(rng, len(index), CLOUD_PERSISTENCE)
    cloud = 1.0 / (1.0 + np.exp(-(1.6 * latent - 0.2)))
    ghi = sky["ghi_clear"].to_numpy() * cloud_to_clearness(cloud)
    day_of_year = sky["day_of_year"].to_numpy()
    solar_hour = solar_time_hours(index, site.longitude)
    seasonal = 9.0 * np.cos(2.0 * np.pi * (day_of_year - 200.0) / 365.0)
    daily = 4.0 * np.cos(2.0 * np.pi * (solar_hour - 15.0) / 24.0)
    temp = 11.0 + seasonal + daily + 1.5 * _ar1(rng, len(index), 0.9)
    return pd.DataFrame({"ghi": ghi, "temp_air": temp, "cloud_cover": cloud}, index=index.rename("timestamp"))


def make_forecast(weather: pd.DataFrame, site, seed: int = 0, cloud_error: float = 0.18) -> pd.DataFrame:
    """A weather forecast for the same hours: the truth with smooth errors added."""
    rng = np.random.default_rng(seed + 1)
    n = len(weather)
    cloud = np.clip(weather["cloud_cover"].to_numpy() + cloud_error * _ar1(rng, n, 0.9), 0.0, 1.0)
    sky = clear_sky_frame(weather.index, site)
    ghi = sky["ghi_clear"].to_numpy() * cloud_to_clearness(cloud)
    temp = weather["temp_air"].to_numpy() + rng.normal(scale=1.2, size=n)
    return pd.DataFrame({"ghi": ghi, "temp_air": temp, "cloud_cover": cloud}, index=weather.index)


def make_production(weather: pd.DataFrame, site, seed: int = 0, noise: float = 0.04) -> pd.Series:
    """Metered AC power in kW for the given weather."""
    rng = np.random.default_rng(seed + 2)
    power = simulate_power(weather.index, weather["ghi"], weather["temp_air"], site)
    noisy = power.to_numpy() * (1.0 + noise * rng.normal(size=len(power)))
    return pd.Series(np.clip(noisy, 0.0, site.ac_limit).round(3), index=weather.index, name="power_kw")


def add_data_problems(production: pd.Series, site, seed: int = 0) -> pd.Series:
    """Return a copy with the kinds of defect that real meter data has.

    A two-day gap (rows removed), a daytime run of repeated values, one reading
    above the inverter limit and one reading at night.
    """
    rng = np.random.default_rng(seed + 3)
    series = production.copy()
    days = series.index.normalize().unique()
    if len(days) < 12:
        raise ValueError("need at least 12 days to place the defects")
    picks = rng.choice(np.arange(2, len(days) - 2), size=4, replace=False)
    gap_day, stuck_day, spike_day, night_day = (days[i] for i in sorted(picks))

    stuck = series.loc[stuck_day : stuck_day + pd.Timedelta(hours=23)]
    daytime = stuck[stuck > 0.05 * site.capacity_kw]
    if len(daytime) >= 5:
        frozen = min(float(daytime.iloc[0]), 0.9 * site.ac_limit)
        series.loc[daytime.index[:5]] = round(frozen, 3)

    spike = series.loc[spike_day : spike_day + pd.Timedelta(hours=23)]
    series.loc[spike.idxmax()] = round(1.3 * site.ac_limit, 3)

    night = series.loc[night_day : night_day + pd.Timedelta(hours=23)]
    series.loc[night[night == 0].index[0]] = round(0.2 * site.capacity_kw, 3)

    gap = (series.index >= gap_day) & (series.index < gap_day + pd.Timedelta(days=2))
    return series[~gap]


def make_dataset(site, start, days: int, seed: int = 0, dirty: bool = False) -> Dataset:
    """Production, a weather forecast and the weather that really occurred."""
    observed = make_weather(site, start, days, seed)
    forecast = make_forecast(observed, site, seed)
    production = make_production(observed, site, seed)
    if dirty:
        production = add_data_problems(production, site, seed)
    return Dataset(production, forecast, observed)

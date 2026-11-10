"""Solar geometry.

All functions take timezone-aware timestamps and work in UTC. Angles are
returned in degrees. Azimuth is measured clockwise from north.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOLAR_CONSTANT = 1361.0  # W/m2


def require_utc(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(times, pd.DatetimeIndex):
        raise TypeError("expected a pandas DatetimeIndex")
    if times.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    return times.tz_convert("UTC")


def day_angle(day_of_year):
    """Fractional year in radians (Spencer, 1971)."""
    return 2.0 * np.pi * (np.asarray(day_of_year, dtype=float) - 1.0) / 365.0


def declination(day_of_year):
    """Solar declination in radians."""
    g = day_angle(day_of_year)
    return (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )


def equation_of_time(day_of_year):
    """Equation of time in minutes."""
    g = day_angle(day_of_year)
    return 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )


def extraterrestrial_normal(day_of_year):
    """Irradiance on a plane normal to the sun at the top of the atmosphere, W/m2."""
    doy = np.asarray(day_of_year, dtype=float)
    return SOLAR_CONSTANT * (1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0))


def solar_time_hours(times: pd.DatetimeIndex, longitude: float):
    """Apparent solar time in hours, in [0, 24)."""
    times = require_utc(times)
    clock = times.hour + times.minute / 60.0 + times.second / 3600.0
    solar = clock + longitude / 15.0 + equation_of_time(times.day_of_year) / 60.0
    return np.mod(np.asarray(solar, dtype=float), 24.0)


def solar_position(times: pd.DatetimeIndex, latitude: float, longitude: float) -> pd.DataFrame:
    """Zenith, elevation and azimuth of the sun at each instant."""
    times = require_utc(times)
    hour_angle = np.radians(15.0 * (solar_time_hours(times, longitude) - 12.0))
    lat = np.radians(latitude)
    dec = declination(times.day_of_year)
    cos_zenith = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle)
    zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    from_south = np.arctan2(np.sin(hour_angle), np.cos(hour_angle) * np.sin(lat) - np.tan(dec) * np.cos(lat))
    azimuth = np.mod(np.degrees(from_south) + 180.0, 360.0)
    return pd.DataFrame(
        {"zenith": zenith, "elevation": 90.0 - zenith, "azimuth": azimuth},
        index=times,
    )

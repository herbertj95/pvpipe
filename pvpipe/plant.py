"""From irradiance on the horizontal to AC power of a fixed-tilt plant."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pvpipe.clearsky import clear_sky_frame, erbs_decomposition

STC_IRRADIANCE = 1000.0  # W/m2
STC_CELL_TEMP = 25.0  # degrees C


def cos_incidence(zenith, azimuth, tilt, surface_azimuth):
    """Cosine of the angle between the sun and the module normal."""
    zenith = np.radians(np.asarray(zenith, dtype=float))
    azimuth = np.radians(np.asarray(azimuth, dtype=float))
    tilt = np.radians(tilt)
    surface_azimuth = np.radians(surface_azimuth)
    return np.cos(zenith) * np.cos(tilt) + np.sin(zenith) * np.sin(tilt) * np.cos(azimuth - surface_azimuth)


def plane_of_array(ghi, dni, dhi, zenith, azimuth, tilt, surface_azimuth, albedo=0.2):
    """Irradiance on the module plane in W/m2, isotropic sky model."""
    beam = np.asarray(dni, dtype=float) * np.clip(cos_incidence(zenith, azimuth, tilt, surface_azimuth), 0.0, None)
    tilt_rad = np.radians(tilt)
    sky = np.asarray(dhi, dtype=float) * (1.0 + np.cos(tilt_rad)) / 2.0
    ground = np.asarray(ghi, dtype=float) * albedo * (1.0 - np.cos(tilt_rad)) / 2.0
    return beam + sky + ground


def cell_temperature(poa, temp_air, noct=45.0):
    """Cell temperature in degrees C from the NOCT model."""
    return np.asarray(temp_air, dtype=float) + np.asarray(poa, dtype=float) / 800.0 * (noct - 20.0)


def dc_power(poa, cell_temp, capacity_kw, temp_coeff=-0.004):
    """DC power in kW, linear in irradiance with a temperature correction."""
    derating = 1.0 + temp_coeff * (np.asarray(cell_temp, dtype=float) - STC_CELL_TEMP)
    return capacity_kw * np.asarray(poa, dtype=float) / STC_IRRADIANCE * derating


def ac_power(dc_kw, site):
    """AC power in kW after system losses and inverter clipping."""
    return np.clip(np.asarray(dc_kw, dtype=float) * (1.0 - site.system_losses), 0.0, site.ac_limit)


def simulate_power(index: pd.DatetimeIndex, ghi, temp_air, site) -> pd.Series:
    """AC power for intervals starting at ``index`` given GHI and air temperature."""
    sky = clear_sky_frame(index, site)
    zenith = sky["zenith"].to_numpy()
    ghi = np.clip(np.asarray(ghi, dtype=float), 0.0, None)
    dni, dhi = erbs_decomposition(ghi, zenith, sky["day_of_year"].to_numpy())
    poa = plane_of_array(ghi, dni, dhi, zenith, sky["azimuth"].to_numpy(), site.tilt, site.azimuth, site.albedo)
    cells = cell_temperature(poa, temp_air, site.noct)
    power = ac_power(dc_power(poa, cells, site.capacity_kw, site.temp_coeff), site)
    return pd.Series(power, index=sky.index, name="power_kw")

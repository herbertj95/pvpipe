"""Clear-sky irradiance and the split of global irradiance into beam and diffuse."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pvpipe.solar import extraterrestrial_normal, require_utc, solar_position

STEP = pd.Timedelta(hours=1)
LOW_SUN_ZENITH = 87.0  # above this the beam component is not resolved


def haurwitz_ghi(zenith):
    """Clear-sky global horizontal irradiance in W/m2 (Haurwitz, 1945)."""
    cos_zenith = np.cos(np.radians(np.asarray(zenith, dtype=float)))
    sun_up = cos_zenith > 0
    safe = np.where(sun_up, cos_zenith, 1.0)
    return np.where(sun_up, 1098.0 * safe * np.exp(-0.059 / safe), 0.0)


def erbs_decomposition(ghi, zenith, day_of_year):
    """Split GHI into (DNI, DHI) with the Erbs et al. (1982) correlation."""
    ghi = np.clip(np.asarray(ghi, dtype=float), 0.0, None)
    zenith = np.asarray(zenith, dtype=float)
    cos_zenith = np.cos(np.radians(zenith))
    extra = extraterrestrial_normal(day_of_year)
    horizontal_extra = extra * np.clip(cos_zenith, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(horizontal_extra > 1.0, ghi / horizontal_extra, 0.0)
    kt = np.clip(kt, 0.0, 1.0)
    fraction = np.where(
        kt <= 0.22,
        1.0 - 0.09 * kt,
        np.where(
            kt <= 0.80,
            0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
            0.165,
        ),
    )
    low_sun = zenith > LOW_SUN_ZENITH
    dhi = np.where(low_sun, ghi, fraction * ghi)
    safe_cos = np.where(low_sun, 1.0, cos_zenith)
    dni = np.where(low_sun, 0.0, (ghi - dhi) / safe_cos)
    dni = np.clip(dni, 0.0, extra)
    return dni, dhi


def clear_sky_frame(index: pd.DatetimeIndex, site, step: pd.Timedelta = STEP) -> pd.DataFrame:
    """Sun position and clear-sky GHI for intervals that start at ``index``.

    Each timestamp labels the start of an interval of length ``step``. The sun
    position is taken at the middle of the interval.
    """
    index = require_utc(index)
    middle = index + step / 2
    position = solar_position(middle, site.latitude, site.longitude)
    zenith = position["zenith"].to_numpy()
    return pd.DataFrame(
        {
            "zenith": zenith,
            "elevation": position["elevation"].to_numpy(),
            "azimuth": position["azimuth"].to_numpy(),
            "day_of_year": middle.day_of_year.to_numpy(),
            "ghi_clear": haurwitz_ghi(zenith),
        },
        index=index,
    )

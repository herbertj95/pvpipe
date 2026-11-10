"""Quality checks on metered production."""
from __future__ import annotations

import pandas as pd

from pvpipe.clearsky import clear_sky_frame

FLAG_NAMES = ("missing", "negative", "above_limit", "night_production", "stuck")
LIMIT_TOLERANCE = 1.05  # readings up to 5 % above the inverter rating are accepted
NIGHT_ELEVATION = -3.0  # degrees; below this the sun cannot produce power
NIGHT_THRESHOLD = 0.01  # fraction of capacity that counts as production at night
STUCK_MIN_RUN = 4  # equal consecutive readings before a sensor counts as stuck
CLIPPING_BAND = 0.98  # readings this close to the inverter rating may repeat legitimately


def hourly_range(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Every hour from the first to the last timestamp of ``index``."""
    if len(index) == 0:
        return index
    return pd.date_range(index.min(), index.max(), freq="1h", name=index.name)


def flag(production: pd.Series, site) -> pd.DataFrame:
    """One boolean column per check, on the full hourly range of the series."""
    full = hourly_range(production.index)
    values = production.reindex(full)
    sky = clear_sky_frame(full, site)
    flags = pd.DataFrame(index=full)
    flags["missing"] = values.isna()
    flags["negative"] = values < 0
    flags["above_limit"] = values > LIMIT_TOLERANCE * site.ac_limit
    flags["night_production"] = (sky["elevation"] < NIGHT_ELEVATION) & (values > NIGHT_THRESHOLD * site.capacity_kw)
    run_id = (values != values.shift()).cumsum()
    run_length = values.groupby(run_id).transform("size")
    repeatable = (values > 0) & (values < CLIPPING_BAND * site.ac_limit)
    flags["stuck"] = repeatable & (run_length >= STUCK_MIN_RUN)
    return flags[list(FLAG_NAMES)]


def clean(production: pd.Series, flags: pd.DataFrame) -> pd.Series:
    """The series on the full hourly range with every flagged reading set to NaN."""
    values = production.reindex(flags.index)
    return values.mask(flags.any(axis=1)).rename(production.name)


def summary(flags: pd.DataFrame) -> pd.Series:
    """Number of hours per flag, plus ``flagged`` (any flag) and ``hours``."""
    counts = flags.sum().astype(int)
    counts["flagged"] = int(flags.any(axis=1).sum())
    counts["hours"] = len(flags)
    return counts

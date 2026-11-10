"""Reference forecasts that need little or no training."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pvpipe.models.base import Forecaster


class Persistence(Forecaster):
    """Tomorrow looks like the newest same-hour measurement known at issue time."""

    name = "persistence"

    def _predict(self, features):
        return features["power_lag_kw"].to_numpy()


class SmartPersistence(Forecaster):
    """Persistence of the clear-sky index instead of the power.

    The ratio of measured power to clear-sky power is carried forward and
    applied to the clear-sky power of the target hour, which removes the
    change in sun position between the two days. Falls back to plain
    persistence where the ratio is not defined.
    """

    name = "smart_persistence"

    def _predict(self, features):
        scaled = features["kc_lag"].to_numpy() * features["power_clear_kw"].to_numpy()
        return np.where(np.isnan(scaled), features["power_lag_kw"].to_numpy(), scaled)


class Climatology(Forecaster):
    """Mean measured power per calendar month and hour of day."""

    name = "climatology"
    needs_fit = True

    def _fit(self, features, target):
        known = target.dropna()
        self._by_month_hour = known.groupby([known.index.month, known.index.hour]).mean()
        self._by_hour = known.groupby(known.index.hour).mean()

    def _predict(self, features):
        index = features.index
        keys = pd.MultiIndex.from_arrays([index.month, index.hour])
        values = self._by_month_hour.reindex(keys).to_numpy()
        fallback = self._by_hour.reindex(index.hour).to_numpy()
        return np.where(np.isnan(values), fallback, values)


class Physical(Forecaster):
    """The plant model driven by the weather forecast."""

    name = "physical"

    def _predict(self, features):
        return features["physical_kw"].to_numpy()

"""Linear model with an L2 penalty, fitted in closed form."""
from __future__ import annotations

import numpy as np

from pvpipe.models.base import Forecaster, NotEnoughData

POWER_INPUTS = ("physical_kw", "power_clear_kw", "power_lag_kw", "power_recent_kw")
OTHER_INPUTS = ("kc_fc", "cloud_fc", "temp_fc", "cos_zenith", "hour_sin", "hour_cos")


class Ridge(Forecaster):
    """Ridge regression on daylight hours.

    Power inputs and the target are divided by the plant capacity, all inputs
    are standardised, and the intercept is not penalised. Gaps in the lagged
    inputs are filled with the training mean of that input.
    """

    name = "ridge"
    needs_fit = True

    def __init__(self, site, alpha: float = 1.0, min_rows: int = 48):
        super().__init__(site)
        if alpha < 0:
            raise ValueError("alpha must not be negative")
        self.alpha = alpha
        self.min_rows = min_rows

    def _design(self, features):
        power = features[list(POWER_INPUTS)].to_numpy(dtype=float) / self.site.capacity_kw
        other = features[list(OTHER_INPUTS)].to_numpy(dtype=float)
        return np.hstack([power, other])

    def _fit(self, features, target):
        design = self._design(features)
        y = target.to_numpy(dtype=float) / self.site.capacity_kw
        rows = features["daylight"].to_numpy() & ~np.isnan(y) & ~np.isnan(design).any(axis=1)
        if rows.sum() < self.min_rows:
            raise NotEnoughData(f"ridge needs {self.min_rows} complete daylight hours, got {int(rows.sum())}")
        x = design[rows]
        self._mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self._scale = np.where(scale > 0, scale, 1.0)
        z = (x - self._mean) / self._scale
        self._intercept = y[rows].mean()
        gram = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.coefficients_ = np.linalg.solve(gram, z.T @ (y[rows] - self._intercept))

    def _predict(self, features):
        design = self._design(features)
        design = np.where(np.isnan(design), self._mean, design)
        z = (design - self._mean) / self._scale
        return (self._intercept + z @ self.coefficients_) * self.site.capacity_kw

"""What every forecast model has in common."""
from __future__ import annotations

import numpy as np
import pandas as pd


class NotFitted(RuntimeError):
    """predict() was called before fit()."""


class NotEnoughData(ValueError):
    """The training data cannot support a fit."""


class Forecaster:
    """Base class. Subclasses implement ``_fit`` (optional) and ``_predict``.

    ``fit`` receives the feature table and the measured power for the same
    hours; ``predict`` returns power in kW for the rows of a feature table.
    Predictions are always zero when the sun is down and never leave the range
    the plant can deliver.
    """

    name = "forecaster"
    needs_fit = False

    def __init__(self, site):
        self.site = site
        self._fitted = False

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "Forecaster":
        self._fit(features, target.reindex(features.index))
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.needs_fit and not self._fitted:
            raise NotFitted(f"{self.name} must be fitted before it can predict")
        raw = np.asarray(self._predict(features), dtype=float)
        bounded = np.clip(raw, 0.0, self.site.ac_limit)
        bounded = np.where(features["daylight"].to_numpy(), bounded, 0.0)
        return pd.Series(bounded, index=features.index, name=self.name)

    def _fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        return None

    def _predict(self, features: pd.DataFrame):
        raise NotImplementedError

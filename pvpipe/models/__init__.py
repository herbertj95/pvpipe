"""Forecast models and the registry the command line uses."""
from __future__ import annotations

from pvpipe.models.base import Forecaster, NotEnoughData, NotFitted
from pvpipe.models.reference import Climatology, Persistence, Physical, SmartPersistence
from pvpipe.models.ridge import Ridge

MODELS = {
    model.name: model
    for model in (Persistence, SmartPersistence, Climatology, Physical, Ridge)
}

__all__ = ["MODELS", "Forecaster", "NotEnoughData", "NotFitted", "make_model", "make_models"]


def make_model(name: str, site, **params) -> Forecaster:
    try:
        factory = MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(sorted(MODELS))}") from None
    return factory(site, **params)


def make_models(names, site) -> dict[str, Forecaster]:
    return {name: make_model(name, site) for name in names}

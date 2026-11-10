"""Site description."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields


class ConfigError(ValueError):
    """The site file is missing a value or holds one that makes no sense."""


@dataclass(frozen=True)
class SiteConfig:
    """A single fixed-tilt PV plant.

    Angles are in degrees. ``azimuth`` is the direction the modules face,
    clockwise from north (180 = south). Powers are in kW.
    """

    name: str
    latitude: float
    longitude: float
    capacity_kw: float  # DC nameplate at standard test conditions
    tilt: float = 30.0
    azimuth: float = 180.0
    ac_limit_kw: float | None = None  # inverter rating; None means no clipping below capacity
    temp_coeff: float = -0.004  # relative power change per degree C of cell temperature
    system_losses: float = 0.14  # wiring, soiling, mismatch, inverter; fraction of DC power
    albedo: float = 0.2
    noct: float = 45.0  # nominal operating cell temperature, degrees C

    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ConfigError(f"latitude out of range: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ConfigError(f"longitude out of range: {self.longitude}")
        if self.capacity_kw <= 0:
            raise ConfigError("capacity_kw must be positive")
        if not 0 <= self.tilt <= 90:
            raise ConfigError(f"tilt out of range: {self.tilt}")
        if not 0 <= self.azimuth < 360:
            raise ConfigError(f"azimuth out of range: {self.azimuth}")
        if self.ac_limit_kw is not None and self.ac_limit_kw <= 0:
            raise ConfigError("ac_limit_kw must be positive")
        if not 0 <= self.system_losses < 1:
            raise ConfigError(f"system_losses out of range: {self.system_losses}")

    @property
    def ac_limit(self) -> float:
        """Largest AC power the plant can deliver."""
        return self.capacity_kw if self.ac_limit_kw is None else self.ac_limit_kw


def load_site(path) -> SiteConfig:
    """Read a site from a TOML file with a ``[site]`` table."""
    with open(path, "rb") as handle:
        try:
            document = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    table = document.get("site")
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: no [site] table")
    known = {field.name for field in fields(SiteConfig)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ConfigError(f"{path}: unknown keys: {', '.join(unknown)}")
    try:
        return SiteConfig(**table)
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

"""Reading and writing hourly time series as CSV.

Every file has a ``timestamp`` column in ISO 8601 with an explicit UTC offset.
A timestamp labels the start of the hour it describes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PRODUCTION_FILE = "production.csv"
FORECAST_FILE = "weather_forecast.csv"
OBSERVED_FILE = "weather_observed.csv"
WEATHER_COLUMNS = ("ghi", "temp_air", "cloud_cover")
HAS_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


class DataError(ValueError):
    """A data file cannot be used as it is."""


@dataclass
class Dataset:
    production: pd.Series  # power_kw
    weather_forecast: pd.DataFrame
    weather_observed: pd.DataFrame | None = None


def read_frame(path, columns) -> pd.DataFrame:
    """Read ``columns`` from a CSV into a frame indexed by UTC timestamp."""
    path = Path(path)
    try:
        raw = pd.read_csv(path, dtype={"timestamp": str})
    except pd.errors.EmptyDataError:
        raise DataError(f"{path.name}: file is empty") from None
    missing = [name for name in ("timestamp", *columns) if name not in raw.columns]
    if missing:
        raise DataError(f"{path.name}: missing columns: {', '.join(missing)}")
    stamps = raw["timestamp"].fillna("").str.strip()
    without_offset = ~stamps.str.contains(HAS_OFFSET)
    if without_offset.any():
        line = int(without_offset.idxmax()) + 2
        raise DataError(f"{path.name} line {line}: timestamp has no UTC offset: {stamps[without_offset.idxmax()]!r}")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True, format="ISO8601"))
    except ValueError as exc:
        raise DataError(f"{path.name}: {exc}") from None
    frame = raw[list(columns)].apply(pd.to_numeric, errors="coerce")
    frame.index = index.rename("timestamp")
    if frame.index.has_duplicates:
        first = frame.index[frame.index.duplicated()][0]
        raise DataError(f"{path.name}: duplicate timestamp {first.isoformat()}")
    return frame.sort_index()


def write_frame(frame: pd.DataFrame, path, decimals: int = 3) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.round(decimals).copy()
    out.insert(0, "timestamp", out.index.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
    out.to_csv(path, index=False, lineterminator="\n")


def read_dataset(directory) -> Dataset:
    directory = Path(directory)
    if not directory.is_dir():
        raise DataError(f"{directory}: not a directory")
    production = read_frame(directory / PRODUCTION_FILE, ["power_kw"])["power_kw"]
    forecast = read_frame(directory / FORECAST_FILE, WEATHER_COLUMNS)
    observed_path = directory / OBSERVED_FILE
    observed = read_frame(observed_path, WEATHER_COLUMNS) if observed_path.exists() else None
    return Dataset(production, forecast, observed)


def write_dataset(dataset: Dataset, directory) -> None:
    directory = Path(directory)
    write_frame(dataset.production.to_frame("power_kw"), directory / PRODUCTION_FILE)
    write_frame(dataset.weather_forecast, directory / FORECAST_FILE)
    if dataset.weather_observed is not None:
        write_frame(dataset.weather_observed, directory / OBSERVED_FILE)

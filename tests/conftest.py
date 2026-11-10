from pathlib import Path

import pandas as pd
import pytest

from pvpipe.config import SiteConfig, load_site
from pvpipe.features import build_features
from pvpipe.synth import make_dataset

ROOT = Path(__file__).resolve().parent.parent
DEMO_SITE = ROOT / "configs" / "demo_site.toml"


@pytest.fixture(scope="session")
def site() -> SiteConfig:
    return load_site(DEMO_SITE)


@pytest.fixture(scope="session")
def dataset(site):
    """Ninety days of clean synthetic data starting in March."""
    return make_dataset(site, "2024-03-01", days=90, seed=11)


@pytest.fixture(scope="session")
def features(site, dataset):
    return build_features(dataset.production, dataset.weather_forecast, site)


@pytest.fixture
def hours():
    def make(start="2024-06-01", periods=24):
        return pd.date_range(start, periods=periods, freq="1h", tz="UTC")

    return make

import pytest

from pvpipe.config import ConfigError, SiteConfig, load_site
from tests.conftest import DEMO_SITE


def test_demo_site_loads():
    site = load_site(DEMO_SITE)
    assert site.name == "demo-rooftop"
    assert site.capacity_kw == 500.0
    assert site.ac_limit == 400.0


def test_ac_limit_defaults_to_capacity():
    assert SiteConfig("s", 10.0, 20.0, 80.0).ac_limit == 80.0


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"latitude": 95.0}, "latitude"),
        ({"longitude": 200.0}, "longitude"),
        ({"capacity_kw": 0.0}, "capacity_kw"),
        ({"tilt": 120.0}, "tilt"),
        ({"azimuth": 360.0}, "azimuth"),
        ({"ac_limit_kw": -1.0}, "ac_limit_kw"),
        ({"system_losses": 1.0}, "system_losses"),
    ],
)
def test_values_that_make_no_sense_are_rejected(overrides, message):
    values = {"name": "s", "latitude": 10.0, "longitude": 20.0, "capacity_kw": 80.0, **overrides}
    with pytest.raises(ConfigError, match=message):
        SiteConfig(**values)


def test_unknown_key_is_named(tmp_path):
    path = tmp_path / "site.toml"
    path.write_text('[site]\nname = "s"\nlatitude = 1.0\nlongitude = 2.0\ncapacity_kw = 3.0\ntracker = true\n')
    with pytest.raises(ConfigError, match="unknown keys: tracker"):
        load_site(path)


def test_missing_required_value(tmp_path):
    path = tmp_path / "site.toml"
    path.write_text('[site]\nname = "s"\nlatitude = 1.0\n')
    with pytest.raises(ConfigError, match="longitude"):
        load_site(path)


def test_file_without_site_table(tmp_path):
    path = tmp_path / "site.toml"
    path.write_text('name = "s"\n')
    with pytest.raises(ConfigError, match=r"no \[site\] table"):
        load_site(path)

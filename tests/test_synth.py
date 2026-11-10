import numpy as np
import pandas as pd
import pytest

from pvpipe.clearsky import clear_sky_frame
from pvpipe.synth import add_data_problems, cloud_to_clearness, make_dataset, make_weather


def test_same_seed_gives_the_same_data(site):
    first = make_dataset(site, "2024-05-01", days=10, seed=3)
    second = make_dataset(site, "2024-05-01", days=10, seed=3)
    pd.testing.assert_series_equal(first.production, second.production)
    pd.testing.assert_frame_equal(first.weather_forecast, second.weather_forecast)


def test_different_seeds_differ(site):
    first = make_dataset(site, "2024-05-01", days=10, seed=3)
    second = make_dataset(site, "2024-05-01", days=10, seed=4)
    assert not first.production.equals(second.production)


def test_weather_covers_whole_days_in_utc(site):
    weather = make_weather(site, "2024-05-01", days=3)
    assert len(weather) == 72
    assert str(weather.index.tz) == "UTC"
    assert weather.index[0] == pd.Timestamp("2024-05-01T00:00:00Z")


def test_weather_is_physically_plausible(site, dataset):
    observed = dataset.weather_observed
    clear = clear_sky_frame(observed.index, site)["ghi_clear"]
    assert observed["cloud_cover"].between(0, 1).all()
    assert (observed["ghi"] >= 0).all()
    assert (observed["ghi"] <= clear + 1e-9).all()
    assert observed["temp_air"].between(-30, 45).all()


def test_clearness_runs_from_one_to_a_quarter():
    assert cloud_to_clearness(0.0) == 1.0
    assert cloud_to_clearness(1.0) == pytest.approx(0.25)
    assert cloud_to_clearness(2.0) == pytest.approx(0.25)


def test_production_respects_the_plant(site, dataset):
    production = dataset.production
    night = clear_sky_frame(production.index, site)["elevation"] < -3
    assert production.between(0, site.ac_limit).all()
    assert (production[night] == 0).all()
    assert production.max() > 0.5 * site.ac_limit


def test_forecast_differs_from_what_happened_but_not_wildly(dataset):
    error = dataset.weather_forecast["cloud_cover"] - dataset.weather_observed["cloud_cover"]
    assert error.abs().mean() > 0.02
    assert error.abs().mean() < 0.3
    assert np.corrcoef(dataset.weather_forecast["ghi"], dataset.weather_observed["ghi"])[0, 1] > 0.9


def test_dirty_data_has_a_gap_of_two_days(site):
    clean = make_dataset(site, "2024-05-01", days=30, seed=5)
    dirty = make_dataset(site, "2024-05-01", days=30, seed=5, dirty=True)
    assert len(clean.production) - len(dirty.production) == 48
    assert dirty.production.max() > site.ac_limit


def test_defects_need_enough_days(site):
    short = make_dataset(site, "2024-05-01", days=5, seed=5)
    with pytest.raises(ValueError, match="at least 12 days"):
        add_data_problems(short.production, site)

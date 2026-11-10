import numpy as np
import pandas as pd
import pytest

from pvpipe.features import FEATURE_COLUMNS, build_features, issue_times, lag_days
from pvpipe.plant import simulate_power


def test_one_row_per_forecast_hour_with_the_documented_columns(features, dataset):
    assert list(features.index) == list(dataset.weather_forecast.index)
    assert tuple(features.columns) == FEATURE_COLUMNS


def test_forecast_is_issued_at_noon_the_day_before(hours):
    index = hours("2024-06-10", 24)
    issued = issue_times(index)
    assert (issued == pd.Timestamp("2024-06-09T12:00:00Z")).all()


def test_issue_hour_and_lead_can_be_changed(hours):
    issued = issue_times(hours("2024-06-10", 2), issue_hour=6, lead_days=2)
    assert issued[0] == pd.Timestamp("2024-06-08T06:00:00Z")


def test_hours_after_the_issue_hour_reach_back_one_more_day(hours):
    back = lag_days(hours("2024-06-10", 24), issue_hour=12)
    assert list(back[:12]) == [1] * 12
    assert list(back[12:]) == [2] * 12


def test_lagged_power_is_the_measurement_from_that_many_days_before(features, dataset):
    stamp = pd.Timestamp("2024-04-10T09:00:00Z")
    assert features.loc[stamp, "power_lag_kw"] == dataset.production.loc[stamp - pd.Timedelta(days=1)]
    late = pd.Timestamp("2024-04-10T14:00:00Z")
    assert features.loc[late, "power_lag_kw"] == dataset.production.loc[late - pd.Timedelta(days=2)]


def test_no_feature_uses_production_from_after_the_issue_time(site, dataset):
    """Changing production at or after the issue time must not change any feature."""
    cut = pd.Timestamp("2024-04-09T12:00:00Z")  # issue time of 10 April
    changed = dataset.production.copy()
    changed[changed.index >= cut] = 123.0
    before = build_features(dataset.production, dataset.weather_forecast, site)
    after = build_features(changed, dataset.weather_forecast, site)
    day = before.index.normalize() == pd.Timestamp("2024-04-10", tz="UTC")
    pd.testing.assert_frame_equal(before[day], after[day])


def test_the_start_of_the_series_has_no_lag(features):
    assert np.isnan(features["power_lag_kw"].iloc[0])
    assert np.isnan(features["power_recent_kw"].iloc[0])


def test_recent_mean_skips_missing_days(site, dataset):
    stamp = pd.Timestamp("2024-04-10T09:00:00Z")
    holed = dataset.production.copy()
    holed[stamp - pd.Timedelta(days=2)] = np.nan
    features = build_features(holed, dataset.weather_forecast, site)
    expected = (holed[stamp - pd.Timedelta(days=1)] + holed[stamp - pd.Timedelta(days=3)]) / 2
    assert features.loc[stamp, "power_recent_kw"] == pytest.approx(expected)


def test_clear_sky_index_is_only_defined_in_proper_daylight(features):
    night = ~features["daylight"]
    assert features.loc[night, "kc_lag"].isna().all()
    assert features["kc_lag"].dropna().between(0, 1.3).all()
    assert (features.loc[night, "kc_fc"] == 0).all()


def test_physical_feature_is_the_plant_model_on_the_weather_forecast(site, dataset, features):
    forecast = dataset.weather_forecast
    expected = simulate_power(forecast.index, forecast["ghi"], forecast["temp_air"], site)
    assert features["physical_kw"].to_numpy() == pytest.approx(expected.to_numpy())
    # Clear-sky power bounds the forecast once the sun is well up. At low sun behind the
    # array a hazier sky has more diffuse light, so the bound does not hold there.
    high_sun = features["cos_zenith"] > 0.35
    assert (features.loc[high_sun, "power_clear_kw"] >= features.loc[high_sun, "physical_kw"] - 1e-9).all()


def test_cyclic_features_stay_on_the_unit_circle(features):
    assert (features["hour_sin"] ** 2 + features["hour_cos"] ** 2).to_numpy() == pytest.approx(1.0)
    assert (features["doy_sin"] ** 2 + features["doy_cos"] ** 2).to_numpy() == pytest.approx(1.0)


@pytest.mark.parametrize("kwargs, message", [({"issue_hour": 24}, "issue_hour"), ({"lead_days": 0}, "lead_days")])
def test_bad_settings_are_refused(site, dataset, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_features(dataset.production, dataset.weather_forecast, site, **kwargs)

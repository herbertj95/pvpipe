import numpy as np
import pandas as pd
import pytest

from pvpipe.features import build_features
from pvpipe.metrics import rmse
from pvpipe.models import MODELS, NotEnoughData, NotFitted, make_model, make_models
from pvpipe.models.base import Forecaster
from pvpipe.synth import make_production, make_weather

SPLIT = pd.Timestamp("2024-05-01T00:00:00Z")


@pytest.fixture(scope="module")
def split(features, dataset):
    target = dataset.production.reindex(features.index)
    train = features.index < SPLIT
    return features[train], target[train], features[~train], target[~train]


def test_registry_lists_every_model():
    assert sorted(MODELS) == ["climatology", "persistence", "physical", "ridge", "smart_persistence"]


def test_unknown_model_name(site):
    with pytest.raises(ValueError, match="unknown model 'arima'"):
        make_model("arima", site)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_predictions_are_zero_at_night_and_within_the_plant_limits(name, site, split):
    train_x, train_y, test_x, _ = split
    prediction = make_model(name, site).fit(train_x, train_y).predict(test_x)
    assert prediction.name == name
    assert list(prediction.index) == list(test_x.index)
    assert (prediction[~test_x["daylight"]] == 0).all()
    assert prediction.dropna().between(0, site.ac_limit).all()


def test_persistence_repeats_the_lagged_measurement(site, split):
    _, _, test_x, _ = split
    prediction = make_model("persistence", site).predict(test_x)
    day = test_x["daylight"]
    assert prediction[day].to_numpy() == pytest.approx(test_x.loc[day, "power_lag_kw"].to_numpy())


def test_smart_persistence_is_exact_under_a_cloudless_sky(site):
    """With no clouds and no noise, scaling by clear-sky power reproduces the next days."""
    weather = make_weather(site, "2024-03-01", days=20, seed=1)
    weather["cloud_cover"] = 0.0
    cloudless = build_features(pd.Series(dtype=float), weather, site)
    weather["ghi"] = cloudless["ghi_clear"]
    production = make_production(weather, site, noise=0.0)
    features = build_features(production, weather, site)
    rows = features[features["kc_lag"].notna()]
    smart = make_model("smart_persistence", site).predict(rows)
    plain = make_model("persistence", site).predict(rows)
    actual = production.reindex(rows.index)
    assert rmse(smart, actual) < 0.25 * rmse(plain, actual)
    assert rmse(smart, actual) < 0.02 * site.capacity_kw


def test_smart_persistence_falls_back_to_plain_persistence(site, split):
    _, _, test_x, _ = split
    rows = test_x[test_x["daylight"] & test_x["kc_lag"].isna() & test_x["power_lag_kw"].notna()]
    assert len(rows) > 0
    smart = make_model("smart_persistence", site).predict(rows)
    assert smart.to_numpy() == pytest.approx(np.clip(rows["power_lag_kw"].to_numpy(), 0, site.ac_limit))


def test_climatology_is_the_mean_of_the_training_hours(site, split):
    train_x, train_y, test_x, _ = split
    prediction = make_model("climatology", site).fit(train_x, train_y).predict(test_x)
    stamp = pd.Timestamp("2024-05-10T10:00:00Z")
    # May is not in the training data, so the hour-of-day mean is used
    assert prediction[stamp] == pytest.approx(train_y[train_y.index.hour == 10].mean())


def test_climatology_must_be_fitted_first(site, split):
    with pytest.raises(NotFitted, match="climatology"):
        make_model("climatology", site).predict(split[2])


def test_physical_returns_the_plant_model_feature(site, split):
    _, _, test_x, _ = split
    prediction = make_model("physical", site).predict(test_x)
    day = test_x["daylight"]
    assert prediction[day].to_numpy() == pytest.approx(test_x.loc[day, "physical_kw"].to_numpy())


def test_ridge_beats_persistence_on_unseen_days(site, split):
    train_x, train_y, test_x, test_y = split
    ridge = make_model("ridge", site).fit(train_x, train_y).predict(test_x)
    persistence = make_model("persistence", site).predict(test_x)
    day = test_x["daylight"]
    assert rmse(ridge[day], test_y[day]) < 0.8 * rmse(persistence[day], test_y[day])


def test_ridge_shrinks_with_a_larger_penalty(site, split):
    train_x, train_y, _, _ = split
    loose = make_model("ridge", site, alpha=0.01).fit(train_x, train_y)
    tight = make_model("ridge", site, alpha=1000.0).fit(train_x, train_y)
    assert np.linalg.norm(tight.coefficients_) < np.linalg.norm(loose.coefficients_)


def test_ridge_copes_with_missing_lags_at_prediction_time(site, split):
    train_x, train_y, test_x, _ = split
    holed = test_x.copy()
    holed[["power_lag_kw", "power_recent_kw"]] = np.nan
    prediction = make_model("ridge", site).fit(train_x, train_y).predict(holed)
    assert prediction.notna().all()


def test_ridge_needs_enough_training_hours(site, split):
    train_x, train_y, _, _ = split
    with pytest.raises(NotEnoughData, match="ridge needs 48"):
        make_model("ridge", site).fit(train_x.iloc[:30], train_y.iloc[:30])


def test_ridge_rejects_a_negative_penalty(site):
    with pytest.raises(ValueError, match="alpha"):
        make_model("ridge", site, alpha=-1.0)


def test_make_models_builds_one_instance_per_name(site):
    models = make_models(["physical", "ridge"], site)
    assert list(models) == ["physical", "ridge"]
    assert all(isinstance(model, Forecaster) for model in models.values())

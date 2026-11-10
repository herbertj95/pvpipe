import pandas as pd
import pytest

from pvpipe.backtest import BacktestConfig, run_backtest, training_rows
from pvpipe.models import make_models
from pvpipe.models.base import Forecaster


class Spy(Forecaster):
    """Records what it was trained on and predicts a constant."""

    name = "spy"
    needs_fit = True

    def __init__(self, site):
        super().__init__(site)
        self.fits = []

    def _fit(self, features, target):
        self.fits.append((features.index.min(), features.index.max(), len(features)))

    def _predict(self, features):
        return [1.0] * len(features)


@pytest.fixture
def target(features, dataset):
    return dataset.production.reindex(features.index)


def test_forecasts_start_after_the_training_days(site, features, target):
    result = run_backtest(features, target, make_models(["persistence"], site), BacktestConfig(train_days=30))
    assert result.index[0] == features.index[0] + pd.Timedelta(days=30)
    assert result.index[-1] == features.index[-1]
    assert list(result.columns) == ["persistence", "actual", "daylight"]
    assert len(result) == 60 * 24


def test_actual_column_is_the_measured_power(site, features, target):
    result = run_backtest(features, target, make_models(["physical"], site))
    pd.testing.assert_series_equal(result["actual"], target.reindex(result.index), check_names=False)


def test_models_are_refitted_on_schedule(site, features, target):
    spy = Spy(site)
    run_backtest(features, target, {"spy": spy}, BacktestConfig(train_days=30, retrain_every=10))
    assert len(spy.fits) == 6  # 60 forecast days, one fit every 10


def test_training_never_reaches_the_issue_time(site, features, target):
    spy = Spy(site)
    run_backtest(features, target, {"spy": spy}, BacktestConfig(train_days=30, retrain_every=1))
    first_day = features.index[0] + pd.Timedelta(days=30)
    for position, (_, newest, _) in enumerate(spy.fits):
        issue = first_day + pd.Timedelta(days=position - 1, hours=12)
        assert newest < issue
        assert newest == issue - pd.Timedelta(hours=1)


def test_expanding_window_grows_and_rolling_window_does_not(site, features, target):
    expanding, rolling = Spy(site), Spy(site)
    run_backtest(features, target, {"spy": expanding}, BacktestConfig(train_days=30, retrain_every=10))
    run_backtest(
        features,
        target,
        {"spy": rolling},
        BacktestConfig(train_days=30, retrain_every=10, window="rolling", rolling_days=20),
    )
    assert [rows for _, _, rows in expanding.fits] == sorted(rows for _, _, rows in expanding.fits)
    assert expanding.fits[-1][2] > expanding.fits[0][2]
    assert {rows for _, _, rows in rolling.fits} == {20 * 24}


def test_training_rows_for_a_rolling_window(features):
    issue = pd.Timestamp("2024-04-20T12:00:00Z")
    rows = training_rows(features, issue, BacktestConfig(window="rolling", rolling_days=5))
    chosen = features.index[rows.to_numpy()]
    assert chosen.min() == issue - pd.Timedelta(days=5)
    assert chosen.max() == issue - pd.Timedelta(hours=1)


def test_too_little_data(site, features, target):
    with pytest.raises(ValueError, match="need more than 90 days"):
        run_backtest(features, target, make_models(["persistence"], site), BacktestConfig(train_days=90))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"train_days": 0}, "train_days"),
        ({"retrain_every": 0}, "retrain_every"),
        ({"window": "sliding"}, "window"),
        ({"rolling_days": 0}, "rolling_days"),
    ],
)
def test_bad_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BacktestConfig(**kwargs)

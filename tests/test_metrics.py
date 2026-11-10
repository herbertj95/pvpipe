import math

import numpy as np
import pandas as pd
import pytest

from pvpipe.metrics import mae, mbe, rmse, score_table, skill_score

NAN = float("nan")


def test_known_values():
    forecast, actual = [10.0, 20.0, 30.0], [12.0, 20.0, 26.0]
    assert mae(forecast, actual) == pytest.approx(2.0)
    assert rmse(forecast, actual) == pytest.approx(math.sqrt(20 / 3))
    assert mbe(forecast, actual) == pytest.approx(2 / 3)


def test_positive_bias_means_over_forecasting():
    assert mbe([5.0, 5.0], [3.0, 3.0]) == 2.0


def test_hours_with_a_missing_value_are_left_out():
    assert mae([1.0, NAN, 5.0], [2.0, 9.0, NAN]) == 1.0


def test_no_common_hours_gives_nan():
    assert math.isnan(rmse([NAN], [1.0]))
    assert math.isnan(mae([], []))


def test_shapes_must_match():
    with pytest.raises(ValueError, match="same shape"):
        mae([1.0, 2.0], [1.0])


def test_skill_is_zero_for_the_reference_and_one_for_a_perfect_forecast():
    actual = np.array([1.0, 4.0, 2.0, 8.0])
    reference = actual + np.array([1.0, -1.0, 2.0, -2.0])
    assert skill_score(reference, reference, actual) == pytest.approx(0.0)
    assert skill_score(actual, reference, actual) == pytest.approx(1.0)


def test_skill_halves_when_the_error_halves():
    actual = np.zeros(4)
    assert skill_score(np.full(4, 1.0), np.full(4, 2.0), actual) == pytest.approx(0.5)


def test_skill_uses_only_hours_all_three_series_have():
    actual = np.array([0.0, 0.0, 0.0])
    reference = np.array([2.0, 2.0, NAN])
    forecast = np.array([1.0, 1.0, 50.0])
    assert skill_score(forecast, reference, actual) == pytest.approx(0.5)


def test_skill_against_a_perfect_reference_is_undefined():
    assert math.isnan(skill_score([1.0], [0.0], [0.0]))


@pytest.fixture
def forecasts():
    index = pd.date_range("2024-06-01", periods=4, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "persistence": [0.0, 40.0, 60.0, 9.0],
            "ridge": [0.0, 45.0, 55.0, 0.0],
            "actual": [0.0, 50.0, 50.0, 0.0],
            "daylight": [False, True, True, False],
        },
        index=index,
    )


def test_score_table_scores_daylight_hours_by_default(forecasts):
    table = score_table(forecasts, capacity_kw=100.0)
    assert list(table.index) == ["persistence", "ridge"]
    assert table.loc["persistence", "hours"] == 2
    assert table.loc["persistence", "mae_kw"] == pytest.approx(10.0)
    assert table.loc["ridge", "nmae_pct"] == pytest.approx(5.0)
    assert table.loc["ridge", "mbe_kw"] == pytest.approx(0.0)
    assert table.loc["persistence", "skill"] == pytest.approx(0.0)
    assert table.loc["ridge", "skill"] == pytest.approx(0.5)


def test_score_table_can_include_the_night(forecasts):
    table = score_table(forecasts, capacity_kw=100.0, daylight_only=False)
    assert table.loc["persistence", "hours"] == 4
    assert table.loc["persistence", "mbe_kw"] == pytest.approx(9.0 / 4)


def test_reference_must_be_in_the_backtest(forecasts):
    with pytest.raises(ValueError, match="reference model 'physical'"):
        score_table(forecasts, 100.0, reference="physical")

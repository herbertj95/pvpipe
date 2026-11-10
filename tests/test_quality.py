import numpy as np
import pandas as pd

from pvpipe import quality
from pvpipe.synth import make_dataset


def day_of_power(site, hours, values_by_hour):
    """A June day that is zero except for the given {hour: kW} readings."""
    index = hours("2024-06-10", 24)
    series = pd.Series(0.0, index=index, name="power_kw")
    for hour, value in values_by_hour.items():
        series.iloc[hour] = value
    return series


def test_clean_synthetic_data_raises_no_flags(site, dataset):
    assert not quality.flag(dataset.production, site).any().any()


def test_missing_hours_are_found_from_the_gap_in_the_index(site, hours):
    series = day_of_power(site, hours, {10: 100.0}).drop(hours("2024-06-10", 24)[5:8])
    flags = quality.flag(series, site)
    assert len(flags) == 24
    assert list(flags.index[flags["missing"]].hour) == [5, 6, 7]


def test_nan_counts_as_missing(site, hours):
    series = day_of_power(site, hours, {10: np.nan})
    assert quality.flag(series, site)["missing"].sum() == 1


def test_negative_reading(site, hours):
    flags = quality.flag(day_of_power(site, hours, {10: -2.0}), site)
    assert list(flags.index[flags["negative"]].hour) == [10]


def test_reading_above_the_inverter_limit(site, hours):
    within = quality.flag(day_of_power(site, hours, {11: 1.04 * site.ac_limit}), site)
    beyond = quality.flag(day_of_power(site, hours, {11: 1.06 * site.ac_limit}), site)
    assert not within["above_limit"].any()
    assert list(beyond.index[beyond["above_limit"]].hour) == [11]


def test_production_at_night(site, hours):
    flags = quality.flag(day_of_power(site, hours, {0: 50.0, 12: 50.0}), site)
    assert list(flags.index[flags["night_production"]].hour) == [0]


def test_tiny_night_readings_are_tolerated(site, hours):
    flags = quality.flag(day_of_power(site, hours, {0: 0.004 * site.capacity_kw}), site)
    assert not flags["night_production"].any()


def test_stuck_sensor_needs_four_equal_readings(site, hours):
    three = quality.flag(day_of_power(site, hours, {9: 120.0, 10: 120.0, 11: 120.0}), site)
    four = quality.flag(day_of_power(site, hours, {9: 120.0, 10: 120.0, 11: 120.0, 12: 120.0}), site)
    assert not three["stuck"].any()
    assert list(four.index[four["stuck"]].hour) == [9, 10, 11, 12]


def test_inverter_clipping_is_not_a_stuck_sensor(site, hours):
    clipped = {hour: site.ac_limit for hour in range(8, 15)}
    assert not quality.flag(day_of_power(site, hours, clipped), site)["stuck"].any()


def test_zero_at_night_is_not_a_stuck_sensor(site, hours):
    assert not quality.flag(day_of_power(site, hours, {12: 100.0}), site)["stuck"].any()


def test_clean_blanks_every_flagged_hour(site, hours):
    series = day_of_power(site, hours, {0: 50.0, 10: -2.0, 12: 100.0})
    cleaned = quality.clean(series, quality.flag(series, site))
    assert cleaned.isna().sum() == 2
    assert cleaned.iloc[12] == 100.0
    assert cleaned.name == "power_kw"


def test_summary_counts_hours_once_even_with_two_flags(site, hours):
    series = day_of_power(site, hours, {0: 2 * site.ac_limit})  # at night and above the limit
    counts = quality.summary(quality.flag(series, site))
    assert counts["above_limit"] == 1
    assert counts["night_production"] == 1
    assert counts["flagged"] == 1
    assert counts["hours"] == 24


def test_injected_defects_are_all_found(site):
    dirty = make_dataset(site, "2024-05-01", days=40, seed=9, dirty=True)
    counts = quality.summary(quality.flag(dirty.production, site))
    assert counts["missing"] == 48
    assert counts["above_limit"] == 1
    assert counts["night_production"] == 1
    assert counts["stuck"] == 5

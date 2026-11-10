import numpy as np
import pandas as pd
import pytest

from pvpipe.solar import (
    declination,
    equation_of_time,
    extraterrestrial_normal,
    solar_position,
    solar_time_hours,
)


def utc(*stamps):
    return pd.DatetimeIndex(stamps, tz="UTC")


def test_declination_at_the_solstices_and_equinox():
    assert np.degrees(declination(172)) == pytest.approx(23.45, abs=0.1)
    assert np.degrees(declination(355)) == pytest.approx(-23.45, abs=0.1)
    assert abs(np.degrees(declination(80))) < 1.0


def test_equation_of_time_extremes():
    assert equation_of_time(45) == pytest.approx(-14.2, abs=0.5)  # mid February
    assert equation_of_time(307) == pytest.approx(16.4, abs=0.5)  # early November


def test_extraterrestrial_irradiance_is_highest_in_january():
    assert extraterrestrial_normal(3) > extraterrestrial_normal(185)
    assert extraterrestrial_normal(3) == pytest.approx(1361 * 1.033, rel=0.002)


def test_sun_is_overhead_at_the_equator_at_equinox_noon():
    position = solar_position(utc("2024-03-20 12:07"), latitude=0.0, longitude=0.0)
    assert position["zenith"].iloc[0] < 1.5


def test_noon_elevation_matches_latitude_and_declination():
    # at solar noon the elevation is 90 - latitude + declination
    position = solar_position(utc("2024-06-21 12:00"), latitude=48.0, longitude=0.0)
    assert position["elevation"].iloc[0] == pytest.approx(90 - 48 + 23.44, abs=0.3)
    assert position["azimuth"].iloc[0] == pytest.approx(180.0, abs=2.0)


def test_sun_is_east_in_the_morning_and_west_in_the_evening():
    position = solar_position(utc("2024-06-21 06:00", "2024-06-21 18:00"), latitude=48.0, longitude=0.0)
    assert 45 < position["azimuth"].iloc[0] < 135
    assert 225 < position["azimuth"].iloc[1] < 315


def test_sun_is_down_at_midnight():
    position = solar_position(utc("2024-06-21 00:00"), latitude=48.0, longitude=0.0)
    assert position["elevation"].iloc[0] < 0


def test_longitude_shifts_solar_time_by_four_minutes_per_degree():
    times = utc("2024-04-15 12:00")
    shift = solar_time_hours(times, 15.0) - solar_time_hours(times, 0.0)
    assert shift[0] == pytest.approx(1.0)


def test_other_time_zones_give_the_same_position():
    in_utc = utc("2024-06-21 10:00")
    elsewhere = in_utc.tz_convert("Asia/Tokyo")
    assert solar_position(elsewhere, 48.0, 11.0)["zenith"].iloc[0] == pytest.approx(
        solar_position(in_utc, 48.0, 11.0)["zenith"].iloc[0]
    )


def test_naive_timestamps_are_refused():
    with pytest.raises(ValueError, match="timezone-aware"):
        solar_position(pd.DatetimeIndex(["2024-06-21 10:00"]), 48.0, 11.0)

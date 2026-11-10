import numpy as np
import pytest

from pvpipe.clearsky import clear_sky_frame, erbs_decomposition, haurwitz_ghi


def test_haurwitz_with_the_sun_overhead():
    assert haurwitz_ghi(0.0) == pytest.approx(1098 * np.exp(-0.059))


def test_haurwitz_is_zero_below_the_horizon():
    assert list(haurwitz_ghi([90.0, 100.0, 170.0])) == [0.0, 0.0, 0.0]


def test_haurwitz_falls_as_the_sun_sinks():
    values = haurwitz_ghi([10.0, 40.0, 70.0, 85.0])
    assert np.all(np.diff(values) < 0)


def test_erbs_components_add_up_to_ghi():
    ghi = np.array([100.0, 400.0, 800.0])
    zenith = np.array([70.0, 50.0, 30.0])
    dni, dhi = erbs_decomposition(ghi, zenith, 172)
    assert dhi + dni * np.cos(np.radians(zenith)) == pytest.approx(ghi)


def test_erbs_overcast_sky_is_nearly_all_diffuse():
    dni, dhi = erbs_decomposition(80.0, 40.0, 172)
    assert dhi / 80.0 > 0.95
    assert dni < 10.0


def test_erbs_clear_sky_is_mostly_beam():
    ghi = float(haurwitz_ghi(30.0))
    dni, dhi = erbs_decomposition(ghi, 30.0, 172)
    assert dhi / ghi < 0.25
    assert dni > 700.0


def test_erbs_low_sun_has_no_beam():
    dni, dhi = erbs_decomposition(20.0, 88.5, 172)
    assert dni == 0.0
    assert dhi == 20.0


def test_erbs_negative_ghi_is_treated_as_zero():
    dni, dhi = erbs_decomposition(-5.0, 40.0, 172)
    assert (dni, dhi) == (0.0, 0.0)


def test_clear_sky_frame_uses_the_middle_of_the_hour(site, hours):
    index = hours("2024-06-21", 24)
    frame = clear_sky_frame(index, site)
    assert list(frame.index) == list(index)
    # the site is at 11.6 E, so solar noon is near 11:15 UTC: the 11:00 hour is the brightest
    assert frame["ghi_clear"].idxmax() == index[11]
    assert frame["ghi_clear"].iloc[0] == 0.0
    assert set(frame.columns) == {"zenith", "elevation", "azimuth", "day_of_year", "ghi_clear"}

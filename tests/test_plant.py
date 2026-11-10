import numpy as np
import pytest

from pvpipe.clearsky import clear_sky_frame
from pvpipe.config import SiteConfig
from pvpipe.plant import (
    ac_power,
    cell_temperature,
    cos_incidence,
    dc_power,
    plane_of_array,
    simulate_power,
)


def test_horizontal_plane_receives_ghi():
    ghi, dni, dhi, zenith = 600.0, 500.0, 600.0 - 500.0 * np.cos(np.radians(40.0)), 40.0
    assert plane_of_array(ghi, dni, dhi, zenith, 180.0, tilt=0.0, surface_azimuth=180.0) == pytest.approx(ghi)


def test_incidence_is_normal_when_the_module_faces_the_sun():
    assert cos_incidence(zenith=30.0, azimuth=180.0, tilt=30.0, surface_azimuth=180.0) == pytest.approx(1.0)


def test_no_beam_from_behind_the_module():
    # sun low in the north, module facing south and tilted steeply
    poa = plane_of_array(ghi=50.0, dni=300.0, dhi=40.0, zenith=80.0, azimuth=0.0, tilt=60.0, surface_azimuth=180.0, albedo=0.0)
    assert poa == pytest.approx(40.0 * (1 + np.cos(np.radians(60.0))) / 2)


def test_ground_reflection_grows_with_tilt_and_albedo():
    flat = plane_of_array(500.0, 0.0, 0.0, 45.0, 180.0, tilt=0.0, surface_azimuth=180.0, albedo=0.5)
    steep = plane_of_array(500.0, 0.0, 0.0, 45.0, 180.0, tilt=90.0, surface_azimuth=180.0, albedo=0.5)
    assert flat == 0.0
    assert steep == pytest.approx(500.0 * 0.5 * 0.5)


def test_cell_temperature_at_noct_conditions():
    assert cell_temperature(poa=800.0, temp_air=20.0, noct=45.0) == pytest.approx(45.0)
    assert cell_temperature(poa=0.0, temp_air=7.0) == pytest.approx(7.0)


def test_dc_power_at_standard_test_conditions_is_the_nameplate():
    assert dc_power(poa=1000.0, cell_temp=25.0, capacity_kw=250.0) == pytest.approx(250.0)


def test_hot_cells_produce_less():
    assert dc_power(1000.0, 65.0, 100.0, temp_coeff=-0.004) == pytest.approx(84.0)


def test_losses_and_clipping():
    site = SiteConfig("s", 45.0, 0.0, capacity_kw=100.0, ac_limit_kw=70.0, system_losses=0.1)
    assert list(ac_power([50.0, 100.0, -3.0], site)) == [45.0, 70.0, 0.0]


def test_simulated_power_is_zero_at_night_and_bounded_by_day(site, hours):
    index = hours("2024-06-21", 24)
    sky = clear_sky_frame(index, site)
    power = simulate_power(index, sky["ghi_clear"], np.full(24, 20.0), site)
    assert power.iloc[0] == 0.0
    assert power.max() <= site.ac_limit
    assert power.max() > 0.6 * site.ac_limit
    assert power.idxmax().hour in (10, 11, 12)


def test_more_irradiance_never_means_less_power(site, hours):
    index = hours("2024-04-10", 24)
    sky = clear_sky_frame(index, site)
    dim = simulate_power(index, 0.3 * sky["ghi_clear"], np.full(24, 10.0), site)
    bright = simulate_power(index, 0.9 * sky["ghi_clear"], np.full(24, 10.0), site)
    assert (bright >= dim).all()


def test_south_facing_beats_north_facing(hours):
    index = hours("2024-03-20", 24)
    south = SiteConfig("s", 48.0, 11.0, 100.0, tilt=30.0, azimuth=180.0)
    north = SiteConfig("n", 48.0, 11.0, 100.0, tilt=30.0, azimuth=0.0)
    ghi = clear_sky_frame(index, south)["ghi_clear"]
    temp = np.full(24, 10.0)
    assert simulate_power(index, ghi, temp, south).sum() > 1.5 * simulate_power(index, ghi, temp, north).sum()

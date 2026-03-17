from os import path
import pytest

from autogalaxy.operate.lens_calc import LensCalc

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_profile_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


def test__1d__half_light_radius_from_light_profile(lp_0):
    assert lp_0.half_light_radius is not None


def test__1d__einstein_radius_from_mass_profile(mp_0, grid_2d_7x7):
    od = LensCalc.from_mass_obj(mp_0)
    einstein_radius = od.einstein_radius_from(grid=grid_2d_7x7)
    assert einstein_radius is not None


def test__2d__critical_curves_from_mass_obj(gal_x1_mp, grid_2d_7x7):
    od = LensCalc.from_mass_obj(gal_x1_mp)
    tc = od.tangential_critical_curve_list_from(grid=grid_2d_7x7)
    assert tc is not None
    assert len(tc) > 0


def test__2d__caustics_from_mass_obj(gal_x1_mp, grid_2d_7x7):
    od = LensCalc.from_mass_obj(gal_x1_mp)
    tc = od.tangential_caustic_list_from(grid=grid_2d_7x7)
    rc = od.radial_caustic_list_from(grid=grid_2d_7x7)
    assert tc is not None
    assert rc is not None


def test__mass_plotter__visuals_with_critical_curves(gal_x1_mp, grid_2d_7x7):
    from autogalaxy.plot.mass_plotter import MassPlotter

    plotter = MassPlotter(mass_obj=gal_x1_mp, grid=grid_2d_7x7)
    visuals = plotter.visuals_2d_with_critical_curves

    od = LensCalc.from_mass_obj(gal_x1_mp)
    expected_tc = od.tangential_critical_curve_list_from(grid=grid_2d_7x7)

    assert (
        visuals.tangential_critical_curves[0] == expected_tc[0]
    ).all()


def test__mass_plotter__visuals_with_caustics_plane_index_1(gal_x1_mp, grid_2d_7x7):
    from autogalaxy.plot.visuals.two_d import Visuals2D
    from autogalaxy.operate.lens_calc import LensCalc

    od = LensCalc.from_mass_obj(gal_x1_mp)
    visuals = Visuals2D().add_critical_curves_or_caustics(
        mass_obj=gal_x1_mp, grid=grid_2d_7x7, plane_index=1
    )

    assert (
        visuals.tangential_caustics[0]
        == od.tangential_caustic_list_from(grid=grid_2d_7x7)[0]
    ).all()
    assert (
        visuals.radial_caustics[0]
        == od.radial_caustic_list_from(grid=grid_2d_7x7)[0]
    ).all()

from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_mp_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


def test__figures_1d__all_are_output(
    mp_0,
    mp_1,
    sub_grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    include_1d_all,
    plot_path,
    plot_patch,
):

    mat_plot_1d = aplt.MatPlot1D(
        half_light_radius_axvline=aplt.HalfLightRadiusAXVLine(color="r"),
        einstein_radius_axvline=aplt.EinsteinRadiusAXVLine(color="r"),
        output=aplt.Output(plot_path, format="png"),
    )

    mass_profile_plotter = aplt.MassProfilePlotter(
        mass_profile=mp_0,
        grid=sub_grid_2d_7x7,
        include_1d=include_1d_all,
        mat_plot_1d=mat_plot_1d,
    )

    mass_profile_plotter.figures_1d(convergence=True, potential=True)

    assert path.join(plot_path, "convergence_1d.png") in plot_patch.paths
    assert path.join(plot_path, "potential_1d.png") in plot_patch.paths

    plot_patch.paths = []

    mp_offset_centre = ag.mp.SphIsothermal(centre=(5.0, 5.0))

    mass_profile_plotter = aplt.MassProfilePDFPlotter(
        mass_profile_pdf_list=[mp_0, mp_1, mp_0, mp_1, mp_0, mp_offset_centre],
        grid=sub_grid_2d_7x7,
        include_1d=include_1d_all,
        mat_plot_1d=mat_plot_1d,
    )

    mass_profile_plotter.figures_1d(convergence=True, potential=True)

    assert path.join(plot_path, "convergence_1d.png") in plot_patch.paths
    assert path.join(plot_path, "potential_1d.png") in plot_patch.paths


def test__figures_2d__all_are_output(
    mp_0,
    sub_grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    include_2d_all,
    plot_path,
    plot_patch,
):

    mass_profile_plotter = aplt.MassProfilePlotter(
        mass_profile=mp_0,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )
    mass_profile_plotter.figures_2d(
        convergence=True,
        potential=True,
        deflections_y=True,
        deflections_x=True,
        magnification=True,
    )

    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths
    assert path.join(plot_path, "potential_2d.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_y_2d.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_x_2d.png") in plot_patch.paths
    assert path.join(plot_path, "magnification_2d.png") in plot_patch.paths

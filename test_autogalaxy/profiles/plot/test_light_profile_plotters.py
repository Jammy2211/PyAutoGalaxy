from os import path

from autoconf import conf
import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_profile_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


def test__figures_1d__all_are_output(
    lp_0,
    lp_1,
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

    light_profile_plotter = aplt.LightProfilePlotter(
        light_profile=lp_0,
        grid=sub_grid_2d_7x7,
        include_1d=include_1d_all,
        mat_plot_1d=mat_plot_1d,
    )

    light_profile_plotter.figures_1d(image=True)

    assert path.join(plot_path, "image_1d.png") in plot_patch.paths

    plot_patch.paths = []

    lp_offset_centre = ag.lp.SphSersic(centre=(5.0, 5.0))

    light_profile_plotter = aplt.LightProfilePDFPlotter(
        light_profile_pdf_list=[lp_0, lp_1, lp_0, lp_1, lp_0, lp_offset_centre],
        grid=sub_grid_2d_7x7,
        include_1d=include_1d_all,
        mat_plot_1d=mat_plot_1d,
        sigma=2.0,
    )

    light_profile_plotter.figures_1d(image=True)

    assert path.join(plot_path, "image_1d.png") in plot_patch.paths


def test__figures_2d__all_are_output(
    lp_0,
    sub_grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    include_2d_all,
    plot_path,
    plot_patch,
):

    light_profile_plotter = aplt.LightProfilePlotter(
        light_profile=lp_0,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    light_profile_plotter.figures_2d(image=True)

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths

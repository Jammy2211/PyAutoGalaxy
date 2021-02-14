from os import path

import autogalaxy.plot as aplt
import pytest


@pytest.fixture(name="plot_path")
def make_fit_interferometer_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__fit_sub_plot_real_space(
    masked_interferometer_fit_7x7,
    masked_interferometer_fit_x2_galaxy_inversion_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(
        fit=masked_interferometer_fit_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_interferometer_plotter.subplot_fit_real_space()

    assert path.join(plot_path, "subplot_fit_real_space.png") in plot_patch.paths

    plot_patch.paths = []

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(
        fit=masked_interferometer_fit_x2_galaxy_inversion_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_interferometer_plotter.subplot_fit_real_space()

    assert path.join(plot_path, "subplot_fit_real_space.png") in plot_patch.paths

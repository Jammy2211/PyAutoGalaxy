from os import path

import autogalaxy.plot as aplt
import pytest


@pytest.fixture(name="plot_path")
def make_fit_dataset_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__fit_sub_plot_real_space(
    fit_interferometer_7x7,
    fit_interferometer_x2_galaxy_inversion_7x7,
    plot_path,
    plot_patch,
):
    fit_plotter = aplt.FitInterferometerPlotter(
        fit=fit_interferometer_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_plotter.subplot_fit_real_space()

    assert path.join(plot_path, "subplot_fit_real_space.png") in plot_patch.paths

    plot_patch.paths = []

    fit_plotter = aplt.FitInterferometerPlotter(
        fit=fit_interferometer_x2_galaxy_inversion_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_plotter.subplot_fit_real_space()

    assert path.join(plot_path, "subplot_fit_real_space.png") in plot_patch.paths

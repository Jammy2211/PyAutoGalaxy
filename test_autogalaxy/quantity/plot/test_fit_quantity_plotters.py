from os import path

import pytest

import autogalaxy.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_galaxy_fit_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "galaxy_fitting",
    )


def test__fit_sub_plot__all_types_of_galaxy_fit(
    fit_quantity_7x7_array_2d, include_2d_all, plot_patch, plot_path
):

    fit_quantity_plotter = aplt.FitQuantityPlotter(
        fit=fit_quantity_7x7_array_2d,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_quantity_plotter.subplot_fit_quantity()
    assert path.join(plot_path, "subplot_fit_quantity.png") in plot_patch.paths

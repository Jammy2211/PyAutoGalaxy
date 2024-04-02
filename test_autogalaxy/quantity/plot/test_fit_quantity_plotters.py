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


def test__fit_individuals__source_and_galaxy__dependent_on_input(
    fit_quantity_7x7_array_2d,
    fit_quantity_7x7_vector_yx_2d,
    include_2d_all,
    plot_path,
    plot_patch,
):
    fit_quantity_plotter = aplt.FitQuantityPlotter(
        fit=fit_quantity_7x7_array_2d,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_quantity_plotter.figures_2d(
        image=True,
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

    fit_quantity_plotter = aplt.FitQuantityPlotter(
        fit=fit_quantity_7x7_vector_yx_2d,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_quantity_plotter.figures_2d(
        image=True,
        noise_map=False,
    )

    assert path.join(plot_path, "data_y.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_y.png") not in plot_patch.paths

    assert path.join(plot_path, "data_x.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_x.png") not in plot_patch.paths


def test__fit_sub_plot__all_types_of_fit(
    fit_quantity_7x7_array_2d,
    fit_quantity_7x7_vector_yx_2d,
    include_2d_all,
    plot_patch,
    plot_path,
):
    fit_quantity_plotter = aplt.FitQuantityPlotter(
        fit=fit_quantity_7x7_array_2d,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_quantity_plotter.subplot_fit()
    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths

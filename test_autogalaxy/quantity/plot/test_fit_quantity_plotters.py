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


def test__fit_sub_plot__all_types_of_fit(
    fit_quantity_7x7_array_2d,
    fit_quantity_7x7_vector_yx_2d,
    plot_patch,
    plot_path,
):
    aplt.subplot_fit_quantity(
        fit=fit_quantity_7x7_array_2d,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths

from os import path

import autogalaxy.plot as aplt
import pytest


@pytest.fixture(name="plot_path")
def make_hyper_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "hyper_galaxies",
    )


def test__plot_hyper_images_of_galaxies(
    hyper_galaxy_image_path_dict_7x7, mask_2d_7x7, include_2d_all, plot_path, plot_patch
):

    hyper_plotter = aplt.AdaptPlotter(
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    hyper_plotter.subplot_hyper_images_of_galaxies(
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict_7x7
    )
    assert (
        path.join(plot_path, "subplot_hyper_images_of_galaxies.png") in plot_patch.paths
    )

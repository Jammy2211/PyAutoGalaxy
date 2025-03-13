from os import path

import autogalaxy.plot as aplt
import pytest


@pytest.fixture(name="plot_path")
def make_adapt_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "adapt",
    )


def test__plot_adapt_adapt_images(
    adapt_galaxy_name_image_dict_7x7, mask_2d_7x7, include_2d_all, plot_path, plot_patch
):
    adapt_plotter = aplt.AdaptPlotter(
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    adapt_plotter.subplot_adapt_images(
        adapt_galaxy_name_image_dict=adapt_galaxy_name_image_dict_7x7
    )
    assert path.join(plot_path, "subplot_adapt_images.png") in plot_patch.paths

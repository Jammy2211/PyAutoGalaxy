from os import path

import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "galaxies"
    )


def test__galaxies_sub_plot_output(galaxies_x2_7x7, grid_2d_7x7, plot_path, plot_patch):
    aplt.subplot_galaxies(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "galaxies.png") in plot_patch.paths

    aplt.subplot_galaxy_images(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "galaxy_images.png") in plot_patch.paths

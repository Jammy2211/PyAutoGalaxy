import shutil
from os import path
import pytest

import autogalaxy as ag

from autogalaxy.analysis.plotter_interface import PlotterInterface

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_interface_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__galaxies(
    masked_imaging_7x7, galaxies_7x7, include_2d_all, plot_path, plot_patch
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = PlotterInterface(image_path=plot_path)

    plotter_interface.galaxies(
        galaxies=galaxies_7x7,
        grid=masked_imaging_7x7.grids.lp,
    )

    assert path.join(plot_path, "subplot_galaxies.png") in plot_patch.paths


def test__galaxies_1d(
    masked_imaging_7x7, galaxies_7x7, include_2d_all, plot_path, plot_patch
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = PlotterInterface(image_path=plot_path)

    plotter_interface.galaxies_1d(
        galaxies=galaxies_7x7,
        grid=masked_imaging_7x7.grids.lp,
    )

    plot_path = path.join(plot_path, "galaxies_1d")

    # subplot


def test__inversion(
    masked_imaging_7x7,
    rectangular_inversion_7x7_3x3,
    include_2d_all,
    plot_path,
    plot_patch,
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = PlotterInterface(image_path=plot_path)

    plotter_interface.inversion(
        inversion=rectangular_inversion_7x7_3x3,
    )

    assert path.join(plot_path, "subplot_inversion_0.png") in plot_patch.paths


def test__adapt_images(
    masked_imaging_7x7,
    include_2d_all,
    adapt_galaxy_name_image_dict_7x7,
    fit_imaging_x2_galaxy_inversion_7x7,
    plot_path,
    plot_patch,
):
    plotter_interface = PlotterInterface(image_path=plot_path)

    adapt_images = ag.AdaptImages(
        galaxy_image_dict=adapt_galaxy_name_image_dict_7x7,
    )

    plotter_interface.adapt_images(
        adapt_images=adapt_images,
    )

    plot_path = path.join(plot_path)

    assert (
        path.join(plot_path, "subplot_adapt_images_of_galaxies.png") in plot_patch.paths
    )

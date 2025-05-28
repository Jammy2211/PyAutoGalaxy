from os import path
import pytest

import autogalaxy as ag

from autogalaxy.interferometer.model.plotter_interface import (
    PlotterInterfaceInterferometer,
)

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_interface_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__interferometer(interferometer_7, include_2d_all, plot_path, plot_patch):
    plotter_interface = PlotterInterfaceInterferometer(image_path=plot_path)

    plotter_interface.interferometer(dataset=interferometer_7)

    assert path.join(plot_path, "subplot_dataset.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(
        file_path=path.join(plot_path, "dataset.fits"), hdu=1
    )

    assert image.shape == (7, 2)


def test__fit_interferometer(
    interferometer_7,
    fit_interferometer_x2_galaxy_inversion_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):
    PlotterInterface = PlotterInterfaceInterferometer(image_path=plot_path)

    PlotterInterface.fit_interferometer(
        fit=fit_interferometer_x2_galaxy_inversion_7x7,
    )

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(
        file_path=path.join(plot_path, "model_galaxy_images.fits"), hdu=1
    )

    assert image.shape == (5, 5)

    image = ag.ndarray_via_fits_from(
        file_path=path.join(plot_path, "fit_dirty_images.fits"), hdu=1
    )

    assert image.shape == (5, 5)

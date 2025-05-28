import shutil
from os import path
import pytest

import autogalaxy as ag
from autogalaxy.imaging.model.plotter_interface import PlotterInterfaceImaging

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_interface_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__imaging(imaging_7x7, include_2d_all, plot_path, plot_patch):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = PlotterInterfaceImaging(image_path=plot_path)

    plotter_interface.imaging(dataset=imaging_7x7)

    assert path.join(plot_path, "subplot_dataset.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(
        file_path=path.join(plot_path, "dataset.fits"), hdu=1
    )

    assert image.shape == (7, 7)


def test__imaging_combined(imaging_7x7, plot_path, plot_patch):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceImaging(image_path=plot_path)

    visualizer.imaging_combined(dataset_list=[imaging_7x7, imaging_7x7])

    assert path.join(plot_path, "subplot_dataset_combined.png") in plot_patch.paths


def test__fit_imaging(
    masked_imaging_7x7,
    fit_imaging_x2_galaxy_inversion_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = PlotterInterfaceImaging(image_path=plot_path)

    plotter_interface.fit_imaging(
        fit=fit_imaging_x2_galaxy_inversion_7x7,
    )

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(file_path=path.join(plot_path, "fit.fits"), hdu=1)

    assert image.shape == (5, 5)

    image = ag.ndarray_via_fits_from(
        file_path=path.join(plot_path, "model_galaxy_images.fits"), hdu=1
    )

    assert image.shape == (5, 5)


def test__fit_imaging_combined(
    fit_imaging_x2_galaxy_inversion_7x7, plot_path, plot_patch
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceImaging(image_path=plot_path)

    visualizer.fit_imaging_combined(fit_list=2 * [fit_imaging_x2_galaxy_inversion_7x7])

    assert path.join(plot_path, "subplot_fit_combined.png") in plot_patch.paths

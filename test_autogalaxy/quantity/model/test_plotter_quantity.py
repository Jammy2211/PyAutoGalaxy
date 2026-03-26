import shutil
from os import path
import pytest

import autogalaxy as ag

from autogalaxy.quantity.model.plotter import PlotterQuantity

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__dataset(
    dataset_quantity_7x7_array_2d,
    plot_path,
    plot_patch,
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter = PlotterQuantity(image_path=plot_path)

    plotter.dataset_quantity(dataset=dataset_quantity_7x7_array_2d)

    image = ag.ndarray_via_fits_from(
        file_path=path.join(plot_path, "dataset.fits"), hdu=1
    )

    assert image.shape == (7, 7)


def test__fit_quantity(
    fit_quantity_7x7_array_2d,
    fit_quantity_7x7_vector_yx_2d,
    plot_path,
    plot_patch,
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter = PlotterQuantity(image_path=plot_path)

    plotter.fit_quantity(fit=fit_quantity_7x7_array_2d)

    assert path.join(plot_path, "fit.png") not in plot_patch.paths

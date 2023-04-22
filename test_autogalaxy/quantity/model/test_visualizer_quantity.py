import shutil
from os import path
import pytest

import autogalaxy as ag
from autoconf import conf
from autogalaxy.quantity.model.visualizer import VisualizerQuantity

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__visualizes_fit_quantity__uses_configs(
    fit_quantity_7x7_array_2d,
    fit_quantity_7x7_vector_yx_2d,
    include_2d_all,
    plot_path,
    plot_patch,
):

    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = VisualizerQuantity(visualize_path=plot_path)

    visualizer.visualize_fit_quantity(fit=fit_quantity_7x7_array_2d)

    plot_path = path.join(plot_path, "fit_quantity")

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

    image = ag.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=path.join(plot_path, "fits", "data.fits"), hdu=0
    )

    assert image.shape == (7, 7)

    plot_patch.paths = []

    visualizer.visualize_fit_quantity(fit=fit_quantity_7x7_vector_yx_2d)

    assert path.join(plot_path, "subplot_fit_y.png") in plot_patch.paths
    assert path.join(plot_path, "data_y.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_y.png") not in plot_patch.paths

    assert path.join(plot_path, "subplot_fit_x.png") in plot_patch.paths
    assert path.join(plot_path, "data_x.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_x.png") not in plot_patch.paths

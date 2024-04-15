import shutil
from os import path
import pytest

from autogalaxy.quantity.model.plotter_interface import PlotterInterfaceQuantity

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_interface_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__fit_quantity(
    fit_quantity_7x7_array_2d,
    fit_quantity_7x7_vector_yx_2d,
    include_2d_all,
    plot_path,
    plot_patch,
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    PlotterInterface = PlotterInterfaceQuantity(image_path=plot_path)

    PlotterInterface.fit_quantity(fit=fit_quantity_7x7_array_2d)

    plot_path = path.join(plot_path, "fit_quantity")

    assert path.join(plot_path, "subplot_fit.png") not in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

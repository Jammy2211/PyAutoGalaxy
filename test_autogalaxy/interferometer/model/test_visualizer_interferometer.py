from os import path
import pytest

from autogalaxy.interferometer.model.visualizer import VisualizerInterferometer

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__visualizes_fit_interferometer__uses_configs(
    interferometer_7,
    fit_interferometer_x2_galaxy_inversion_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):
    visualizer = VisualizerInterferometer(visualize_path=plot_path)

    visualizer.visualize_fit_interferometer(
        fit=fit_interferometer_x2_galaxy_inversion_7x7, during_analysis=True
    )

    plot_path = path.join(plot_path, "fit_dataset")

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

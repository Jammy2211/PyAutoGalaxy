from os import path

from autoconf import conf
import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_profile_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )

def test__figures_2d__all_are_output(
    lp_0,
    grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    light_profile_plotter = aplt.LightProfilePlotter(
        light_profile=lp_0,
        grid=grid_2d_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    light_profile_plotter.figures_2d(image=True)

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths

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


def test__all_quantities_are_output(
    lp_0,
    sub_grid_7x7,
    grid_irregular_grouped_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    ag.plot.LightProfile.image(
        light_profile=lp_0,
        grid=sub_grid_7x7,
        positions=grid_irregular_grouped_7x7,
        include_2d=include_2d_all,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths

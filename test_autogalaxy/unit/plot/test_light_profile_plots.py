import os
from os import path

from autoconf import conf
import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="profile_plotter_path")
def make_profile_plotter_setup():
    return "{}/files/plots/profiles/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files/plotter"), path.join(directory, "output")
    )


def test__all_quantities_are_output(
    lp_0, sub_grid_7x7, positions_7x7, include_all, profile_plotter_path, plot_patch
):

    ag.plot.LightProfile.image(
        light_profile=lp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(profile_plotter_path, format="png")),
    )

    assert profile_plotter_path + "image.png" in plot_patch.paths

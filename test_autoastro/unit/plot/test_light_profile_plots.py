import autoarray as aa
import autoastro.plot as aplt
import autoastro as aast
import pytest
import os
from os import path

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="profile_plotter_path")
def make_profile_plotter_setup():
    return "{}/files/plots/profiles/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        path.join(directory, "files/plotter"), path.join(directory, "output")
    )


def test__all_quantities_are_output(
    lp_0, sub_grid_7x7, positions_7x7, include_all, profile_plotter_path, plot_patch
):

    aast.plot.LightProfile.profile_image(
        light_profile=lp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(profile_plotter_path, format="png")),
    )

    assert profile_plotter_path + "profile_image.png" in plot_patch.paths

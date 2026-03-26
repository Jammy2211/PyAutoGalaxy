from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_profile_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


def test__subplot_image(
    lp_0,
    lp_1,
    grid_2d_7x7,
    plot_path,
    plot_patch,
):
    basis = ag.lp_basis.Basis(profile_list=[lp_0, lp_1])

    aplt.subplot_basis_image(
        basis=basis,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "basis_image.png") in plot_patch.paths

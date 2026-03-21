from os import path

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
    plot_path,
    plot_patch,
):
    aplt.plot_array(
        array=lp_0.image_2d_from(grid=grid_2d_7x7),
        title="Image",
        output_path=plot_path,
        output_filename="image_2d",
        output_format="png",
    )

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths

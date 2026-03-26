from os import path

import pytest

import autogalaxy.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_imaging_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__subplot_of_galaxy(fit_imaging_x2_galaxy_7x7, plot_path, plot_patch):
    aplt.subplot_fit_imaging_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )
    aplt.subplot_fit_imaging_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=1,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "of_galaxy_1.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.subplot_fit_imaging_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "of_galaxy_1.png") not in plot_patch.paths

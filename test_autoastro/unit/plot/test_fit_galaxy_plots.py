import autoarray as aa
import autoastro.plot as aplt
import autoastro as aast
import os

import pytest

from os import path

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="galaxy_fit_plotter_path")
def make_galaxy_fit_plotter_setup():
    return "{}/../../../test_files/plotting/galaxy_fitting/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


def test__fit_sub_plot__all_types_of_galaxy_fit(
    gal_fit_7x7_image,
    gal_fit_7x7_convergence,
    gal_fit_7x7_potential,
    gal_fit_7x7_deflections_y,
    gal_fit_7x7_deflections_x,
    positions_7x7,
    include_all,
    plot_patch,
    galaxy_fit_plotter_path,
):
    aplt.fit_galaxy.subplot_fit_galaxy(
        fit=gal_fit_7x7_image,
        positions=positions_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(path=galaxy_fit_plotter_path, format="png")
        ),
    )

    assert galaxy_fit_plotter_path + "subplot_fit_galaxy.png" in plot_patch.paths

    aplt.fit_galaxy.subplot_fit_galaxy(
        fit=gal_fit_7x7_convergence,
        positions=positions_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(path=galaxy_fit_plotter_path, format="png")
        ),
    )

    assert galaxy_fit_plotter_path + "subplot_fit_galaxy.png" in plot_patch.paths

    aplt.fit_galaxy.subplot_fit_galaxy(
        fit=gal_fit_7x7_potential,
        positions=positions_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(path=galaxy_fit_plotter_path, format="png")
        ),
    )

    assert galaxy_fit_plotter_path + "subplot_fit_galaxy.png" in plot_patch.paths

    aplt.fit_galaxy.subplot_fit_galaxy(
        fit=gal_fit_7x7_deflections_y,
        positions=positions_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(path=galaxy_fit_plotter_path, format="png")
        ),
    )

    assert galaxy_fit_plotter_path + "subplot_fit_galaxy.png" in plot_patch.paths

    aplt.fit_galaxy.subplot_fit_galaxy(
        fit=gal_fit_7x7_deflections_x,
        positions=positions_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(path=galaxy_fit_plotter_path, format="png")
        ),
    )

    assert galaxy_fit_plotter_path + "subplot_fit_galaxy.png" in plot_patch.paths

import autoarray as aa
import autoastro.plot as aplt
import autoastro as aast
import pytest
import os
from os import path

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="mp_plotter_path")
def make_mp_plotter_setup():
    return "{}/../../../test_files/plotting/mps/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


def test__all_quantities_are_output(
    mp_0, sub_grid_7x7, positions_7x7, include_all, mp_plotter_path, plot_patch
):

    aast.plot.mp.convergence(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(mp_plotter_path, format="png")),
    )

    assert mp_plotter_path + "convergence.png" in plot_patch.paths

    aast.plot.mp.potential(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(mp_plotter_path, format="png")),
    )

    assert mp_plotter_path + "potential.png" in plot_patch.paths

    aast.plot.mp.deflections_y(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(mp_plotter_path, format="png")),
    )

    assert mp_plotter_path + "deflections_y.png" in plot_patch.paths

    aast.plot.mp.deflections_x(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(mp_plotter_path, format="png")),
    )

    assert mp_plotter_path + "deflections_x.png" in plot_patch.paths

    aast.plot.mp.magnification(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(mp_plotter_path, format="png")),
    )

    assert mp_plotter_path + "magnification.png" in plot_patch.paths

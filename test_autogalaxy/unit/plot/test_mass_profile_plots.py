from os import path

from autoconf import conf
import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_mp_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files", "plotter"), path.join(directory, "output")
    )


def test__all_quantities_are_output(
    mp_0, sub_grid_7x7, positions_7x7, include_all, plot_path, plot_patch
):

    ag.plot.MassProfile.convergence(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "convergence.png") in plot_patch.paths

    ag.plot.MassProfile.potential(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "potential.png") in plot_patch.paths

    ag.plot.MassProfile.deflections_y(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "deflections_y.png") in plot_patch.paths

    ag.plot.MassProfile.deflections_x(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "deflections_x.png") in plot_patch.paths

    ag.plot.MassProfile.magnification(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "magnification.png") in plot_patch.paths

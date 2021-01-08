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


def test__all_quantities_are_output(
    mp_0,
    sub_grid_7x7,
    grid_irregular_grouped_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    mass_profile_plotter = aplt.MassProfilePlotter(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    mass_profile_plotter.figure_convergence()
    assert path.join(plot_path, "convergence.png") in plot_patch.paths

    mass_profile_plotter.figure_potential()

    assert path.join(plot_path, "potential.png") in plot_patch.paths

    mass_profile_plotter.figure_deflections_y()

    assert path.join(plot_path, "deflections_y.png") in plot_patch.paths

    mass_profile_plotter.figure_deflections_x()

    assert path.join(plot_path, "deflections_x.png") in plot_patch.paths

    mass_profile_plotter.figure_magnification()

    assert path.join(plot_path, "magnification.png") in plot_patch.paths
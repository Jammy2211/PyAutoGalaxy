from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_mp_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


def test__figures_2d__all_are_output(
    mp_0,
    grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    mass_profile_plotter = aplt.MassProfilePlotter(
        mass_profile=mp_0,
        grid=grid_2d_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )
    mass_profile_plotter.figures_2d(
        convergence=True,
        potential=True,
        deflections_y=True,
        deflections_x=True,
        magnification=True,
    )

    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths
    assert path.join(plot_path, "potential_2d.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_y_2d.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_x_2d.png") in plot_patch.paths
    assert path.join(plot_path, "magnification_2d.png") in plot_patch.paths

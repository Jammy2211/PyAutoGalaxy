from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_galaxy_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "galaxy"
    )


def test__figures_2d__all_are_output(
    gal_x1_lp_x1_mp,
    grid_2d_7x7,
    mask_2d_7x7,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp,
        grid=grid_2d_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )
    galaxy_plotter.figures_2d(image=True, convergence=True)

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths


def test__subplots_galaxy_quantities__all_are_output(
    gal_x1_lp_x1_mp,
    grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp,
        grid=grid_2d_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )
    galaxy_plotter.subplot_of_light_profiles(image=True)

    assert path.join(plot_path, "subplot_image.png") in plot_patch.paths

    galaxy_plotter.subplot_of_mass_profiles(
        convergence=True, potential=True, deflections_y=True, deflections_x=True
    )

    assert path.join(plot_path, "subplot_convergence.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_potential.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_deflections_y.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_deflections_x.png") in plot_patch.paths

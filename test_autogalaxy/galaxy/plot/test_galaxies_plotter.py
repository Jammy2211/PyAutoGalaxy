from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "galaxies"
    )


def test__all_individual_plotter__output_file_with_default_name(
    galaxies_7x7,
    grid_2d_7x7,
    mask_2d_7x7,
    grid_2d_irregular_7x7_list,
    include_2d_all,
    plot_path,
    plot_patch,
):
    plotter = aplt.GalaxiesPlotter(
        galaxies=galaxies_7x7,
        grid=grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    plotter.figures_2d(image=True, convergence=True)

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths


def test__figures_of_galaxies(
    galaxies_x2_7x7,
    grid_2d_7x7,
    mask_2d_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):
    plotter = aplt.GalaxiesPlotter(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    plotter.figures_2d_of_galaxies(image=True)

    assert path.join(plot_path, "image_2d_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "image_2d_of_galaxy_1.png") in plot_patch.paths

    plot_patch.paths = []

    plotter.figures_2d_of_galaxies(image=True, galaxy_index=0)

    assert path.join(plot_path, "image_2d_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "image_2d_of_galaxy_1.png") not in plot_patch.paths


def test__galaxies_sub_plot_output(
    galaxies_x2_7x7, grid_2d_7x7, include_2d_all, plot_path, plot_patch
):
    plotter = aplt.GalaxiesPlotter(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    plotter.subplot_galaxies()
    assert path.join(plot_path, "subplot_galaxies.png") in plot_patch.paths

    plotter.subplot_galaxy_images()
    assert path.join(plot_path, "subplot_galaxy_images.png") in plot_patch.paths

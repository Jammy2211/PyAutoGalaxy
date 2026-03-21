from os import path

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
    plot_path,
    plot_patch,
):
    aplt.plot_galaxies_image_2d(
        galaxies=galaxies_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    aplt.plot_galaxies_convergence_2d(
        galaxies=galaxies_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths


def test__figures_of_galaxies(
    galaxies_x2_7x7,
    grid_2d_7x7,
    mask_2d_7x7,
    plot_path,
    plot_patch,
):
    from autogalaxy.galaxy.plot.galaxies_plots import plot_image_2d_of_galaxy

    plot_image_2d_of_galaxy(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )
    plot_image_2d_of_galaxy(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        galaxy_index=1,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "image_2d_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "image_2d_of_galaxy_1.png") in plot_patch.paths

    plot_patch.paths = []

    plot_image_2d_of_galaxy(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "image_2d_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "image_2d_of_galaxy_1.png") not in plot_patch.paths


def test__galaxies_sub_plot_output(galaxies_x2_7x7, grid_2d_7x7, plot_path, plot_patch):
    aplt.subplot_galaxies(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_galaxies.png") in plot_patch.paths

    aplt.subplot_galaxy_images(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_galaxy_images.png") in plot_patch.paths

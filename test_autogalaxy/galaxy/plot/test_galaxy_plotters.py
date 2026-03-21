from os import path

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
    aplt.plot_galaxy_image_2d(
        galaxy=gal_x1_lp_x1_mp,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    aplt.plot_galaxy_convergence_2d(
        galaxy=gal_x1_lp_x1_mp,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths


def test__subplots_galaxy_quantities__all_are_output(
    gal_x1_lp_x1_mp,
    grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    aplt.subplot_galaxy_light_profiles(
        galaxy=gal_x1_lp_x1_mp,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "subplot_image.png") in plot_patch.paths

    aplt.subplot_galaxy_mass_profiles(
        galaxy=gal_x1_lp_x1_mp,
        grid=grid_2d_7x7,
        convergence=True,
        potential=True,
        deflections_y=True,
        deflections_x=True,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "subplot_convergence.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_potential.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_deflections_y.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_deflections_x.png") in plot_patch.paths

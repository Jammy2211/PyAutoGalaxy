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


def test__visuals_with_include_2d(gal_x1_lp_x1_mp, grid_7x7):

    visuals_2d = aplt.Visuals2D(vector_field=2)

    include = aplt.Include2D(
        origin=True,
        mask=True,
        border=True,
        light_profile_centres=True,
        mass_profile_centres=True,
        critical_curves=True,
        caustics=True,
    )

    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp, grid=grid_7x7, visuals_2d=visuals_2d, include_2d=include
    )

    assert galaxy_plotter.visuals_with_include_2d.origin.in_list == [(0.0, 0.0)]
    assert (galaxy_plotter.visuals_with_include_2d.mask == grid_7x7.mask).all()
    assert (
        galaxy_plotter.visuals_with_include_2d.border
        == grid_7x7.mask.border_grid_sub_1.slim_binned
    ).all()
    assert galaxy_plotter.visuals_with_include_2d.light_profile_centres.in_list == [
        gal_x1_lp_x1_mp.light_profile_0.centre
    ]
    assert galaxy_plotter.visuals_with_include_2d.mass_profile_centres.in_list == [
        gal_x1_lp_x1_mp.mass_profile_0.centre
    ]
    assert galaxy_plotter.visuals_with_include_2d.vector_field == 2

    include = aplt.Include2D(origin=False, mask=False, border=False)

    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp, grid=grid_7x7, visuals_2d=visuals_2d, include_2d=include
    )

    assert galaxy_plotter.visuals_with_include_2d.origin == None
    assert galaxy_plotter.visuals_with_include_2d.mask == None
    assert galaxy_plotter.visuals_with_include_2d.border == None
    assert galaxy_plotter.visuals_with_include_2d.vector_field == 2


def test__individual_images_are_output(
    gal_x1_lp_x1_mp,
    sub_grid_7x7,
    mask_7x7,
    grid_irregular_grouped_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )
    galaxy_plotter.figures(image=True, convergence=True)

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "convergence.png") in plot_patch.paths

    gal_x1_lp_x1_mp.hyper_galaxy = ag.HyperGalaxy()
    gal_x1_lp_x1_mp.hyper_model_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )
    gal_x1_lp_x1_mp.hyper_galaxy_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )

    galaxy_plotter.figures(contribution_map=True)
    assert path.join(plot_path, "contribution_map.png") in plot_patch.paths


def test__subplots_galaxy_quantities__all_are_output(
    gal_x1_lp_x1_mp,
    sub_grid_7x7,
    grid_irregular_grouped_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        include_2d=include_2d_all,
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

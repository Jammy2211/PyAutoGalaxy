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


def test__figures_1d__all_are_output(
    gal_x1_lp_x1_mp, sub_grid_2d_7x7, mask_2d_7x7, include_1d_all, plot_path, plot_patch
):

    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_2d_7x7,
        include_1d=include_1d_all,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )
    galaxy_plotter.figures_1d(image=True, convergence=True, potential=True)

    assert path.join(plot_path, "image_1d.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_1d.png") in plot_patch.paths
    assert path.join(plot_path, "potential_1d.png") in plot_patch.paths

    plot_patch.paths = []

    galaxy_plotter = aplt.GalaxyPDFPlotter(
        galaxy_pdf_list=[gal_x1_lp_x1_mp, gal_x1_lp_x1_mp, gal_x1_lp_x1_mp],
        grid=sub_grid_2d_7x7,
        include_1d=include_1d_all,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )
    galaxy_plotter.figures_1d(image=True, convergence=True, potential=True)

    assert path.join(plot_path, "image_1d.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_1d.png") in plot_patch.paths
    assert path.join(plot_path, "potential_1d.png") in plot_patch.paths


def test__figures_1d_decomposed__all_are_output(
    gal_x1_lp_x1_mp, sub_grid_2d_7x7, mask_2d_7x7, include_1d_all, plot_path, plot_patch
):

    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_2d_7x7,
        include_1d=include_1d_all,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )
    galaxy_plotter.figures_1d_decomposed(image=True, convergence=True, potential=True)

    assert path.join(plot_path, "image_1d_decomposed.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_1d_decomposed.png") in plot_patch.paths
    assert path.join(plot_path, "potential_1d_decomposed.png") in plot_patch.paths

    plot_patch.paths = []

    galaxy_plotter = aplt.GalaxyPDFPlotter(
        galaxy_pdf_list=[gal_x1_lp_x1_mp, gal_x1_lp_x1_mp, gal_x1_lp_x1_mp],
        grid=sub_grid_2d_7x7,
        include_1d=include_1d_all,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    galaxy_plotter.figures_1d_decomposed(image=True, convergence=True, potential=True)

    assert path.join(plot_path, "image_1d_decomposed.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_1d_decomposed.png") in plot_patch.paths
    assert path.join(plot_path, "potential_1d_decomposed.png") in plot_patch.paths


def test__figures_1d_decomposed__light_profiles_different_centres_making_offset_radial_grid(
    sub_grid_2d_7x7, mask_2d_7x7, include_1d_all, plot_path, plot_patch
):

    lp_0 = ag.lp.SphSersic(centre=(0.0, 0.0))
    lp_1 = ag.lp.SphSersic(centre=(1.0, 1.0))

    mp_0 = ag.mp.SphIsothermal(centre=(0.0, 0.0))
    mp_1 = ag.mp.SphIsothermal(centre=(1.0, 1.0))

    gal = ag.Galaxy(redshift=0.5, light_0=lp_0, light_1=lp_1, mass_0=mp_0, mass_1=mp_1)

    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal,
        grid=sub_grid_2d_7x7,
        include_1d=include_1d_all,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )
    galaxy_plotter.figures_1d_decomposed(image=True, convergence=True, potential=True)

    assert path.join(plot_path, "image_1d_decomposed.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_1d_decomposed.png") in plot_patch.paths
    assert path.join(plot_path, "potential_1d_decomposed.png") in plot_patch.paths


    lp_0 = ag.lp.SphSersic(centre=(0.0, 0.0))
    lp_1 = ag.lp.SphSersic(centre=(1.0, 1.0))

    mp_0 = ag.mp.SphIsothermal(centre=(0.0, 0.0))
    mp_1 = ag.mp.SphIsothermal(centre=(1.0, 1.0))

    gal_0 = ag.Galaxy(redshift=0.5, light_0=lp_0, mass_0=mp_0)
    gal_1 = ag.Galaxy(redshift=0.5, light_1=lp_1, mass_1=mp_1)

    galaxy_plotter = aplt.GalaxyPDFPlotter(
        galaxy_pdf_list=[gal_0, gal_1],
        grid=sub_grid_2d_7x7,
        include_1d=include_1d_all,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )
    galaxy_plotter.figures_1d_decomposed(image=True, convergence=True, potential=True)

    assert path.join(plot_path, "image_1d_decomposed.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_1d_decomposed.png") in plot_patch.paths
    assert path.join(plot_path, "potential_1d_decomposed.png") in plot_patch.paths


def test__figures_2d__all_are_output(
    gal_x1_lp_x1_mp,
    sub_grid_2d_7x7,
    mask_2d_7x7,
    grid_2d_irregular_7x7_list,
    include_2d_all,
    plot_path,
    plot_patch,
):

    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )
    galaxy_plotter.figures_2d(image=True, convergence=True)

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths

    gal_x1_lp_x1_mp.hyper_galaxy = ag.HyperGalaxy()
    gal_x1_lp_x1_mp.hyper_model_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )
    gal_x1_lp_x1_mp.hyper_galaxy_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )

    galaxy_plotter.figures_2d(contribution_map=True)
    assert path.join(plot_path, "contribution_map_2d.png") in plot_patch.paths


def test__subplots_galaxy_quantities__all_are_output(
    gal_x1_lp_x1_mp,
    sub_grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    include_2d_all,
    plot_path,
    plot_patch,
):

    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_2d_7x7,
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

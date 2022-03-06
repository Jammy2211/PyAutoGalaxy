from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plane_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "plane"
    )


def test__all_individual_plotter__output_file_with_default_name(
    plane_7x7,
    sub_grid_2d_7x7,
    mask_2d_7x7,
    grid_2d_irregular_7x7_list,
    include_2d_all,
    plot_path,
    plot_patch,
):

    plane_plotter = aplt.PlanePlotter(
        plane=plane_7x7,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    plane_plotter.figures_2d(
        image=True, plane_image=True, plane_grid=True, convergence=True
    )

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "plane_image.png") in plot_patch.paths
    assert path.join(plot_path, "plane_grid.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths

    plane_7x7.galaxies[0].hyper_galaxy = ag.HyperGalaxy()
    plane_7x7.galaxies[0].hyper_model_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )
    plane_7x7.galaxies[0].hyper_galaxy_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )

    plane_plotter.figures_2d(contribution_map=True)

    assert path.join(plot_path, "contribution_map_2d.png") in plot_patch.paths


def test__figures_of_galaxies(
    plane_x2_gal_7x7,
    sub_grid_2d_7x7,
    mask_2d_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    plane_plotter = aplt.PlanePlotter(
        plane=plane_x2_gal_7x7,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    plane_plotter.figures_2d_of_galaxies(image=True)

    assert path.join(plot_path, "image_2d_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "image_2d_of_galaxy_1.png") in plot_patch.paths

    plot_patch.paths = []

    plane_plotter.figures_2d_of_galaxies(image=True, galaxy_index=0)

    assert path.join(plot_path, "image_2d_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "image_2d_of_galaxy_1.png") not in plot_patch.paths


def test__plane_sub_plot_output(
    plane_x2_gal_7x7, sub_grid_2d_7x7, include_2d_all, plot_path, plot_patch
):

    plane_plotter = aplt.PlanePlotter(
        plane=plane_x2_gal_7x7,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    plane_plotter.subplot_plane()
    assert path.join(plot_path, "subplot_plane.png") in plot_patch.paths

    plane_plotter.subplot_galaxy_images()
    assert path.join(plot_path, "subplot_galaxy_images.png") in plot_patch.paths

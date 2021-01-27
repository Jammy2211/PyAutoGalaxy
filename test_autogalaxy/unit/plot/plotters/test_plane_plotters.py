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
    sub_grid_7x7,
    mask_7x7,
    grid_irregular_grouped_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    plane_plotter = aplt.PlanePlotter(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    plane_plotter.figures(image=True, plane_image=True, plane_grid=True)

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "plane_image.png") in plot_patch.paths
    assert path.join(plot_path, "plane_grid.png") in plot_patch.paths

    plane_7x7.galaxies[0].hyper_galaxy = ag.HyperGalaxy()
    plane_7x7.galaxies[0].hyper_model_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )
    plane_7x7.galaxies[0].hyper_galaxy_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )

    plane_plotter.figures(contribution_map=True)

    assert path.join(plot_path, "contribution_map.png") in plot_patch.paths

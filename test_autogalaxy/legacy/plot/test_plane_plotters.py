from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt
import autogalaxy.legacy.plot as aplt_legacy
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
    galaxy = ag.legacy.Galaxy(
        redshift=0.5,
        light_profile_0=ag.lp.SersicSph(
            intensity=1.0, effective_radius=2.0, sersic_index=2.0
        ),
    )

    plane_7x7 = ag.legacy.Plane(galaxies=[galaxy])

    plane_plotter = aplt_legacy.PlanePlotter(
        plane=plane_7x7,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    plane_plotter.figures_2d(
        image=True, plane_image=True, plane_grid=True, convergence=True
    )

    plane_7x7.galaxies[0].hyper_galaxy = ag.legacy.HyperGalaxy()
    plane_7x7.galaxies[0].adapt_model_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )
    plane_7x7.galaxies[0].adapt_galaxy_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )

    plane_plotter.figures_2d(contribution_map=True)

    assert path.join(plot_path, "contribution_map_2d.png") in plot_patch.paths

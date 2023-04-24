from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt
import autogalaxy.legacy.plot as aplt_legacy
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_galaxy_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "galaxy"
    )


def test__figures_2d__all_are_output(
    gal_x1_lp_x1_mp,
    sub_grid_2d_7x7,
    mask_2d_7x7,
    grid_2d_irregular_7x7_list,
    include_2d_all,
    plot_path,
    plot_patch,
):
    gal_x1_lp_x1_mp = ag.legacy.Galaxy(
        redshift=gal_x1_lp_x1_mp.redshift, light=ag.lp.SersicSph
    )

    galaxy_plotter = aplt_legacy.GalaxyPlotter(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    gal_x1_lp_x1_mp.hyper_galaxy = ag.legacy.HyperGalaxy()
    gal_x1_lp_x1_mp.adapt_model_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )
    gal_x1_lp_x1_mp.adapt_galaxy_image = ag.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )

    galaxy_plotter.figures_2d(contribution_map=True)
    assert path.join(plot_path, "contribution_map_2d.png") in plot_patch.paths

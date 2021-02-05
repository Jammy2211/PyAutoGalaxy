from os import path

import autogalaxy.plot as aplt
from autogalaxy.plot.plotters import lensing_obj_plotter as lo_plotter
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_mp_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


def test__visuals_with_include_2d(mp_0, grid_7x7):

    visuals_2d = aplt.Visuals2D(vector_field=2)

    include = aplt.Include2D(
        origin=True,
        mask=True,
        border=True,
        mass_profile_centres=True,
        critical_curves=True,
        caustics=True,
    )

    lensing_obj_plotter = lo_plotter.LensingObjPlotter(
        lensing_obj=mp_0, grid=grid_7x7, visuals_2d=visuals_2d, include_2d=include
    )

    assert lensing_obj_plotter.visuals_with_include_2d.origin.in_list == [(0.0, 0.0)]
    assert (lensing_obj_plotter.visuals_with_include_2d.mask == grid_7x7.mask).all()
    assert (
        lensing_obj_plotter.visuals_with_include_2d.border
        == grid_7x7.mask.border_grid_sub_1.slim_binned
    ).all()
    assert lensing_obj_plotter.visuals_with_include_2d.mass_profile_centres.in_list == [
        mp_0.centre
    ]
    assert lensing_obj_plotter.visuals_with_include_2d.vector_field == 2

    include = aplt.Include2D(origin=False, mask=False, border=False)

    lensing_obj_plotter = lo_plotter.LensingObjPlotter(
        lensing_obj=mp_0, grid=grid_7x7, visuals_2d=visuals_2d, include_2d=include
    )

    assert lensing_obj_plotter.visuals_with_include_2d.origin == None
    assert lensing_obj_plotter.visuals_with_include_2d.mask == None
    assert lensing_obj_plotter.visuals_with_include_2d.border == None
    assert lensing_obj_plotter.visuals_with_include_2d.vector_field == 2


def test__all_quantities_are_output(
    mp_0,
    sub_grid_7x7,
    grid_irregular_grouped_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    lensing_obj_plotter = lo_plotter.LensingObjPlotter(
        lensing_obj=mp_0,
        grid=sub_grid_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )
    lensing_obj_plotter.figures(
        convergence=True,
        potential=True,
        deflections_y=True,
        deflections_x=True,
        magnification=True,
    )

    assert path.join(plot_path, "convergence.png") in plot_patch.paths
    assert path.join(plot_path, "potential.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_y.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_x.png") in plot_patch.paths
    assert path.join(plot_path, "magnification.png") in plot_patch.paths

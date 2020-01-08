import autoarray as aa
import autoastro as aast
import os

import pytest
from os import path

from autoarray import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="galaxy_plotter_path")
def make_galaxy_plotter_setup():
    return "{}/../../../test_files/plotting/model_galaxy/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


def test__individual_images_are_output(
    gal_x1_lp_x1_mp,
    sub_grid_7x7,
    mask_7x7,
    positions_7x7,
    galaxy_plotter_path,
    plot_patch,
):

    aast.plot.galaxy.profile_image(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        plot_in_kpc=True,
        include_critical_curves=True,
        include_caustics=True,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "profile_image.png" in plot_patch.paths

    aast.plot.galaxy.convergence(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include_critical_curves=False,
        include_caustics=True,
        plot_in_kpc=True,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "convergence.png" in plot_patch.paths

    aast.plot.galaxy.potential(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include_critical_curves=False,
        include_caustics=True,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "potential.png" in plot_patch.paths

    aast.plot.galaxy.deflections_y(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include_critical_curves=False,
        include_caustics=True,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "deflections_y.png" in plot_patch.paths

    aast.plot.galaxy.deflections_x(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include_critical_curves=False,
        include_caustics=True,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "deflections_x.png" in plot_patch.paths

    aast.plot.galaxy.magnification(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include_critical_curves=False,
        include_caustics=True,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "magnification.png" in plot_patch.paths


def test__individual_galaxy_quantities__all_are_output(
    gal_x1_lp_x1_mp,
    sub_grid_7x7,
    mask_7x7,
    positions_7x7,
    galaxy_plotter_path,
    plot_patch,
):
    aast.plot.galaxy.profile_image_subplot(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "profile_image_subplot.png" in plot_patch.paths

    aast.plot.galaxy.convergence_subplot(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "convergence_subplot.png" in plot_patch.paths

    aast.plot.galaxy.potential_subplot(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "potential_subplot.png" in plot_patch.paths

    aast.plot.galaxy.deflections_y_subplot(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "deflections_y_subplot.png" in plot_patch.paths

    aast.plot.galaxy.deflections_x_subplot(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        array_plotter=aa.plotter.array(
            output_path=galaxy_plotter_path, output_format="png"
        ),
    )

    assert galaxy_plotter_path + "deflections_x_subplot.png" in plot_patch.paths

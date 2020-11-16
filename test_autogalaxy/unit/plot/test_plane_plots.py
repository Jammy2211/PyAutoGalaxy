from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt
import pytest
from autoconf import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plane_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "plane"
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files", "plotter"), path.join(directory, "output")
    )


def test__all_individual_plotters__output_file_with_default_name(
    plane_7x7, sub_grid_7x7, mask_7x7, positions_7x7, include_all, plot_path, plot_patch
):

    aplt.Plane.image(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths

    aplt.Plane.plane_image(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "plane_image.png") in plot_patch.paths

    aplt.Plane.convergence(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "convergence.png") in plot_patch.paths

    aplt.Plane.potential(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "potential.png") in plot_patch.paths

    aplt.Plane.deflections_y(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "deflections_y.png") in plot_patch.paths

    aplt.Plane.deflections_x(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "deflections_x.png") in plot_patch.paths

    aplt.Plane.magnification(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "magnification.png") in plot_patch.paths

    aplt.Plane.plane_grid(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "plane_grid.png") in plot_patch.paths

    plane_7x7.galaxies[0].hyper_galaxy = ag.HyperGalaxy()
    plane_7x7.galaxies[0].hyper_model_image = ag.Array.ones(
        shape_2d=(7, 7), pixel_scales=0.1
    )
    plane_7x7.galaxies[0].hyper_galaxy_image = ag.Array.ones(
        shape_2d=(7, 7), pixel_scales=0.1
    )

    aplt.Plane.contribution_map(
        plane=plane_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "contribution_map.png") in plot_patch.paths

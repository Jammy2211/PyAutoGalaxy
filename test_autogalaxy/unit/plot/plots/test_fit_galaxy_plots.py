from os import path

import pytest

import autogalaxy.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_galaxy_fit_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "galaxy_fitting",
    )


def test__fit_sub_plot__all_types_of_galaxy_fit(
    gal_fit_7x7_image,
    gal_fit_7x7_convergence,
    gal_fit_7x7_potential,
    gal_fit_7x7_deflections_y,
    gal_fit_7x7_deflections_x,
    include_2d_all,
    plot_patch,
    plot_path,
):
    aplt.FitGalaxy.subplot_fit_galaxy(
        fit=gal_fit_7x7_image,
        include_2d=include_2d_all,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_fit_galaxy.png") in plot_patch.paths

    aplt.FitGalaxy.subplot_fit_galaxy(
        fit=gal_fit_7x7_convergence,
        include_2d=include_2d_all,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_fit_galaxy.png") in plot_patch.paths

    aplt.FitGalaxy.subplot_fit_galaxy(
        fit=gal_fit_7x7_potential,
        include_2d=include_2d_all,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_fit_galaxy.png") in plot_patch.paths

    aplt.FitGalaxy.subplot_fit_galaxy(
        fit=gal_fit_7x7_deflections_y,
        include_2d=include_2d_all,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_fit_galaxy.png") in plot_patch.paths

    aplt.FitGalaxy.subplot_fit_galaxy(
        fit=gal_fit_7x7_deflections_x,
        include_2d=include_2d_all,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_fit_galaxy.png") in plot_patch.paths

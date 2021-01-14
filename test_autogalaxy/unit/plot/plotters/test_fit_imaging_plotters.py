from os import path

import pytest

import autogalaxy.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_imaging_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__fit_individuals__source_and_galaxy__dependent_on_input(
    masked_imaging_fit_x2_galaxy_7x7, include_2d_all, plot_path, plot_patch
):

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=masked_imaging_fit_x2_galaxy_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_imaging_plotter.figures(
        image=True,
        noise_map=False,
        signal_to_noise_map=False,
        model_image=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "model_image.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths


def test__figures_of_galaxies(
    masked_imaging_fit_x2_galaxy_7x7, include_2d_all, plot_path, plot_patch
):

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=masked_imaging_fit_x2_galaxy_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_imaging_plotter.figures_of_galaxies(subtracted_image=True)

    assert path.join(plot_path, "subtracted_image_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "subtracted_image_of_galaxy_1.png") in plot_patch.paths

    fit_imaging_plotter.figures_of_galaxies(model_image=True)

    assert path.join(plot_path, "model_image_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "model_image_of_galaxy_1.png") in plot_patch.paths

    plot_patch.paths = []

    fit_imaging_plotter.figures_of_galaxies(subtracted_image=True, galaxy_index=0)

    assert path.join(plot_path, "subtracted_image_of_galaxy_0.png") in plot_patch.paths
    assert (
        path.join(plot_path, "subtracted_image_of_galaxy_1.png") not in plot_patch.paths
    )

    fit_imaging_plotter.figures_of_galaxies(model_image=True, galaxy_index=1)

    assert path.join(plot_path, "model_image_of_galaxy_0.png") not in plot_patch.paths
    assert path.join(plot_path, "model_image_of_galaxy_1.png") in plot_patch.paths


def test__subplot_of_galaxy(
    masked_imaging_fit_x2_galaxy_7x7, include_2d_all, plot_path, plot_patch
):

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=masked_imaging_fit_x2_galaxy_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )
    fit_imaging_plotter.subplots_of_galaxies()
    assert path.join(plot_path, "subplot_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_of_galaxy_1.png") in plot_patch.paths

    plot_patch.paths = []

    fit_imaging_plotter.subplots_of_galaxies(galaxy_index=0)

    assert path.join(plot_path, "subplot_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_of_galaxy_1.png") not in plot_patch.paths

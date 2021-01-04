import os
from os import path

import pytest

import autogalaxy.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_imaging_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__subtracted_image_of_galaxy_is_output(
    masked_imaging_fit_x2_galaxy_7x7, include_2d_all, plot_path, plot_patch
):

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=masked_imaging_fit_x2_galaxy_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_imaging_plotter.figure_subtracted_image_of_galaxy(galaxy_index=0)
    assert path.join(plot_path, "subtracted_image_of_galaxy_0.png") in plot_patch.paths

    fit_imaging_plotter.figure_subtracted_image_of_galaxy(galaxy_index=1)
    assert path.join(plot_path, "subtracted_image_of_galaxy_1.png") in plot_patch.paths


def test__model_image_of_galaxy_is_output(
    masked_imaging_fit_x2_galaxy_7x7, include_2d_all, plot_path, plot_patch
):

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=masked_imaging_fit_x2_galaxy_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_imaging_plotter.figure_model_image_of_galaxy(galaxy_index=0)
    assert path.join(plot_path, "model_image_of_galaxy_0.png") in plot_patch.paths

    fit_imaging_plotter.figure_model_image_of_galaxy(galaxy_index=1)
    assert path.join(plot_path, "model_image_of_galaxy_1.png") in plot_patch.paths


def test__subplot_of_galaxy(
    masked_imaging_fit_x2_galaxy_7x7, include_2d_all, plot_path, plot_patch
):

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=masked_imaging_fit_x2_galaxy_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_imaging_plotter.subplot_of_galaxy(galaxy_index=0)
    assert path.join(plot_path, "subplot_of_galaxy_0.png") in plot_patch.paths

    fit_imaging_plotter.subplot_of_galaxy(galaxy_index=1)
    assert path.join(plot_path, "subplot_of_galaxy_1.png") in plot_patch.paths

    fit_imaging_plotter.subplots_of_all_galaxies()
    assert path.join(plot_path, "subplot_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_of_galaxy_1.png") in plot_patch.paths


def test__fit_individuals__source_and_galaxy__dependent_on_input(
    masked_imaging_fit_x2_galaxy_7x7, include_2d_all, plot_path, plot_patch
):

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=masked_imaging_fit_x2_galaxy_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_imaging_plotter.figure_individuals(
        plot_image=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=True,
        plot_chi_squared_map=True,
        plot_subtracted_images_of_galaxies=True,
        plot_model_images_of_galaxies=True,
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "model_image.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths
    assert path.join(plot_path, "subtracted_image_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "subtracted_image_of_galaxy_1.png") in plot_patch.paths
    assert path.join(plot_path, "model_image_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "model_image_of_galaxy_1.png") in plot_patch.paths

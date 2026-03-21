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
    fit_imaging_x2_galaxy_7x7, plot_path, plot_patch
):
    aplt.plot_fit_imaging_data(
        fit=fit_imaging_x2_galaxy_7x7,
        output_path=plot_path,
        output_format="png",
    )
    aplt.plot_fit_imaging_model_image(
        fit=fit_imaging_x2_galaxy_7x7,
        output_path=plot_path,
        output_format="png",
    )
    aplt.plot_fit_imaging_chi_squared_map(
        fit=fit_imaging_x2_galaxy_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "model_image.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths


def test__figures_of_galaxies(fit_imaging_x2_galaxy_7x7, plot_path, plot_patch):
    aplt.plot_subtracted_image_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )
    aplt.plot_subtracted_image_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=1,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "subtracted_image_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "subtracted_image_of_galaxy_1.png") in plot_patch.paths

    aplt.plot_model_image_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )
    aplt.plot_model_image_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=1,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "model_image_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "model_image_of_galaxy_1.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.plot_subtracted_image_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "subtracted_image_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "subtracted_image_of_galaxy_1.png") not in plot_patch.paths

    aplt.plot_model_image_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=1,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "model_image_of_galaxy_0.png") not in plot_patch.paths
    assert path.join(plot_path, "model_image_of_galaxy_1.png") in plot_patch.paths


def test__subplot_of_galaxy(fit_imaging_x2_galaxy_7x7, plot_path, plot_patch):
    aplt.subplot_fit_imaging_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )
    aplt.subplot_fit_imaging_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=1,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_of_galaxy_1.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.subplot_fit_imaging_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "subplot_of_galaxy_0.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_of_galaxy_1.png") not in plot_patch.paths

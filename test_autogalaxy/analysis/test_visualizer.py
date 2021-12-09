import shutil
from os import path
import pytest

from autoconf import conf
from autogalaxy.analysis import visualizer as vis

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


@pytest.fixture(autouse=True)
def push_config(plot_path):
    conf.instance.push(path.join(directory, "config"), output_path=plot_path)


def test__visualizes_imaging__uses_configs(
    imaging_7x7, include_2d_all, plot_path, plot_patch
):

    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_imaging(imaging=imaging_7x7)

    plot_path = path.join(plot_path, "imaging")

    assert path.join(plot_path, "subplot_imaging.png") in plot_patch.paths
    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "psf.png") in plot_patch.paths
    assert path.join(plot_path, "inverse_noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert (
        path.join(plot_path, "absolute_signal_to_noise_map.png") not in plot_patch.paths
    )
    assert path.join(plot_path, "potential_chi_squared_map.png") in plot_patch.paths


def test__visualizes_interferometer__uses_configs(
    interferometer_7, include_2d_all, plot_path, plot_patch
):

    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_interferometer(interferometer=interferometer_7)

    plot_path = path.join(plot_path, "interferometer")

    assert path.join(plot_path, "subplot_interferometer.png") in plot_patch.paths
    assert path.join(plot_path, "visibilities.png") in plot_patch.paths
    assert path.join(plot_path, "u_wavelengths.png") not in plot_patch.paths
    assert path.join(plot_path, "v_wavelengths.png") not in plot_patch.paths


def test__visualize_inversion__uses_configs(
    masked_imaging_7x7, voronoi_inversion_9_3x3, include_2d_all, plot_path, plot_patch
):

    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_inversion(
        inversion=voronoi_inversion_9_3x3, during_analysis=True
    )

    plot_path = path.join(plot_path, "inversion")

    assert path.join(plot_path, "subplot_inversion_0.png") in plot_patch.paths
    assert path.join(plot_path, "reconstructed_image.png") in plot_patch.paths
    assert path.join(plot_path, "reconstruction.png") in plot_patch.paths
    assert path.join(plot_path, "inversion", "errors.png") not in plot_patch.paths

    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths
    assert path.join(plot_path, "regularization_weights.png") not in plot_patch.paths


def test__visualize_hyper_images__uses_config(
    masked_imaging_7x7,
    hyper_model_image_7x7,
    include_2d_all,
    hyper_galaxy_image_path_dict_7x7,
    fit_imaging_x2_galaxy_inversion_7x7,
    plot_path,
    plot_patch,
):

    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_hyper_images(
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict_7x7,
        hyper_model_image=hyper_model_image_7x7,
    )

    plot_path = path.join(plot_path, "hyper")

    assert path.join(plot_path, "hyper_model_image.png") in plot_patch.paths
    assert (
        path.join(plot_path, "subplot_hyper_images_of_galaxies.png") in plot_patch.paths
    )

    visualizer.visualize_contribution_maps(
        plane=fit_imaging_x2_galaxy_inversion_7x7.plane
    )

    assert (
        path.join(plot_path, "subplot_contribution_map_list.png")
        not in plot_patch.paths
    )

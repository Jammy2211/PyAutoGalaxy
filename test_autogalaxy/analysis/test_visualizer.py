import shutil
from os import path
import pytest

import autogalaxy as ag

from autogalaxy.analysis import visualizer as vis

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__visualizes_plane__uses_configs(
    masked_imaging_7x7, plane_7x7, include_2d_all, plot_path, plot_patch
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_plane(
        plane=plane_7x7, grid=masked_imaging_7x7.grid, during_analysis=False
    )

    plot_path = path.join(plot_path, "plane")

    assert path.join(plot_path, "subplot_plane.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_galaxy_images.png") in plot_patch.paths
    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths
    assert path.join(plot_path, "potential_2d.png") not in plot_patch.paths
    assert path.join(plot_path, "deflections_y_2d.png") not in plot_patch.paths
    assert path.join(plot_path, "deflections_x_2d.png") not in plot_patch.paths
    assert path.join(plot_path, "magnification_2d.png") in plot_patch.paths

    convergence = ag.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=path.join(plot_path, "fits", "convergence_2d.fits"), hdu=0
    )

    assert convergence.shape == (5, 5)


def test__visualizes_galaxies__uses_configs(
    masked_imaging_7x7, plane_7x7, include_2d_all, plot_path, plot_patch
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_galaxies(
        galaxies=plane_7x7.galaxies, grid=masked_imaging_7x7.grid, during_analysis=False
    )

    plot_path = path.join(plot_path, "galaxies")

    assert path.join(plot_path, "image_1d_decomposed.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_1d_decomposed.png") not in plot_patch.paths
    assert path.join(plot_path, "potential_1d_decomposed.png") in plot_patch.paths


def test__visualizes_imaging__uses_configs(
    imaging_7x7, include_2d_all, plot_path, plot_patch
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_imaging(imaging=imaging_7x7)

    plot_path = path.join(plot_path, "dataset")

    assert path.join(plot_path, "subplot_dataset.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "psf.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths


def test__visualizes_interferometer__uses_configs(
    interferometer_7, include_2d_all, plot_path, plot_patch
):
    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_interferometer(interferometer=interferometer_7)

    plot_path = path.join(plot_path, "dataset")

    assert path.join(plot_path, "subplot_dataset.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
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
    assert path.join(plot_path, "regularization_weights.png") not in plot_patch.paths


def test__visualize_adapt_images__uses_config(
    masked_imaging_7x7,
    adapt_model_image_7x7,
    include_2d_all,
    adapt_galaxy_image_path_dict_7x7,
    fit_imaging_x2_galaxy_inversion_7x7,
    plot_path,
    plot_patch,
):
    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_adapt_images(
        adapt_galaxy_image_path_dict=adapt_galaxy_image_path_dict_7x7,
        adapt_model_image=adapt_model_image_7x7,
    )

    plot_path = path.join(plot_path, "adapt")

    assert path.join(plot_path, "adapt_model_image.png") in plot_patch.paths
    assert (
        path.join(plot_path, "subplot_adapt_images_of_galaxies.png") in plot_patch.paths
    )

    visualizer.visualize_contribution_maps(
        plane=fit_imaging_x2_galaxy_inversion_7x7.plane
    )

    assert (
        path.join(plot_path, "subplot_contribution_map_list.png")
        not in plot_patch.paths
    )

import shutil
from os import path

import pytest

import autogalaxy as ag
from autoconf import conf
from autogalaxy.pipeline import visualizer as vis

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files", "plot", "visualizer")


@pytest.fixture()
def set_config_path(plot_path):

    conf.instance.push(
        new_path=path.join(directory, "unit", "pipeline", "config"),
        output_path=path.join(plot_path),
    )


class TestVisualizer:
    def test__visualizes_imaging__uses_configs(
        self, imaging_7x7, include_2d_all, plot_path, plot_patch
    ):

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_imaging(imaging=imaging_7x7)

        plot_path = path.join(plot_path, "imaging")

        assert path.join(plot_path, "subplot_imaging.png") in plot_patch.paths
        assert path.join(plot_path, "image.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "psf.png") in plot_patch.paths
        assert path.join(plot_path, "inverse_noise_map.png") in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert (
            path.join(plot_path, "absolute_signal_to_noise_map.png")
            not in plot_patch.paths
        )
        assert path.join(plot_path, "potential_chi_squared_map.png") in plot_patch.paths

    def test__visualizes_fit_imaging__uses_configs(
        self,
        masked_imaging_7x7,
        masked_imaging_fit_x2_galaxy_inversion_7x7,
        include_2d_all,
        plot_path,
        plot_patch,
    ):

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_fit_imaging(
            fit=masked_imaging_fit_x2_galaxy_inversion_7x7, during_analysis=False
        )

        plot_path = path.join(plot_path, "fit_imaging")

        assert path.join(plot_path, "subplot_fit_imaging.png") in plot_patch.paths
        assert path.join(plot_path, "image.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "model_image.png") in plot_patch.paths
        assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
        assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths
        assert (
            path.join(plot_path, "subtracted_image_of_galaxy_0.png") in plot_patch.paths
        )
        assert (
            path.join(plot_path, "subtracted_image_of_galaxy_1.png") in plot_patch.paths
        )

        print(plot_patch.paths)

        assert (
            path.join(plot_path, "model_image_of_galaxy_0.png") not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "model_image_of_galaxy_1.png") not in plot_patch.paths
        )

        image = ag.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "fits", "image.fits"), hdu=0
        )

        assert image.shape == (5, 5)

    def test__visualizes_interferometer__uses_configs(
        self, interferometer_7, include_2d_all, plot_path, plot_patch
    ):
        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_interferometer(interferometer=interferometer_7)

        plot_path = path.join(plot_path, "interferometer")

        assert path.join(plot_path, "subplot_interferometer.png") in plot_patch.paths
        assert path.join(plot_path, "visibilities.png") in plot_patch.paths
        assert path.join(plot_path, "u_wavelengths.png") not in plot_patch.paths
        assert path.join(plot_path, "v_wavelengths.png") not in plot_patch.paths

    def test__visualizes_fit_interferometer__uses_configs(
        self,
        masked_interferometer_7,
        masked_interferometer_fit_x2_galaxy_inversion_7x7,
        include_2d_all,
        plot_path,
        plot_patch,
    ):
        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_fit_interferometer(
            fit=masked_interferometer_fit_x2_galaxy_inversion_7x7, during_analysis=True
        )

        plot_path = path.join(plot_path, "fit_interferometer")

        assert (
            path.join(plot_path, "subplot_fit_interferometer.png") in plot_patch.paths
        )
        assert path.join(plot_path, "visibilities.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "model_visibilities.png") in plot_patch.paths
        assert (
            path.join(plot_path, "real_residual_map_vs_uv_distances.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "real_normalized_residual_map_vs_uv_distances.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "real_chi_squared_map_vs_uv_distances.png")
            in plot_patch.paths
        )

    def test__visualize_inversion__uses_configs(
        self,
        masked_imaging_7x7,
        voronoi_inversion_9_3x3,
        include_2d_all,
        plot_path,
        plot_patch,
    ):
        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_inversion(
            inversion=voronoi_inversion_9_3x3, during_analysis=True
        )

        plot_path = path.join(plot_path, "inversion")

        assert path.join(plot_path, "subplot_inversion.png") in plot_patch.paths
        assert path.join(plot_path, "reconstructed_image.png") in plot_patch.paths
        assert path.join(plot_path, "reconstruction.png") in plot_patch.paths
        # assert path.join(plot_path,"inversion","errors.png") not in plot_patch.paths

        assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
        assert (
            path.join(plot_path, "normalized_residual_map.png") not in plot_patch.paths
        )
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths
        assert (
            path.join(plot_path, "regularization_weights.png") not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "interpolated_reconstruction.png") in plot_patch.paths
        )
        assert path.join(plot_path, "interpolated_errors.png") in plot_patch.paths

    def test__visualize_hyper_images__uses_config(
        self,
        masked_imaging_7x7,
        hyper_model_image_7x7,
        include_2d_all,
        hyper_galaxy_image_path_dict_7x7,
        masked_imaging_fit_x2_galaxy_inversion_7x7,
        plot_path,
        plot_patch,
    ):

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_hyper_images(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict_7x7,
            hyper_model_image=hyper_model_image_7x7,
            plane=masked_imaging_fit_x2_galaxy_inversion_7x7.plane,
        )

        plot_path = path.join(plot_path, "hyper")

        assert path.join(plot_path, "hyper_model_image.png") in plot_patch.paths
        assert (
            path.join(plot_path, "subplot_hyper_images_of_galaxies.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "subplot_contribution_maps_of_galaxies.png")
            not in plot_patch.paths
        )

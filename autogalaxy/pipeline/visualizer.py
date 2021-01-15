from os import path

from autoconf import conf
from autoarray.plot.mat_wrap.wrap import wrap_base
from autoarray.plot.plotters import (
    imaging_plotters,
    interferometer_plotters,
    inversion_plotters,
)
from autogalaxy.plot.plotters import (
    fit_imaging_plotters,
    fit_interferometer_plotters,
    fit_galaxy_plotters,
    hyper_plotters,
)
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals


def setting(section, name):
    return conf.instance["visualize"]["plots"][section][name]


def plot_setting(section, name):
    return setting(section, name)


class Visualizer:
    def __init__(self, visualize_path):

        self.visualize_path = visualize_path

        self.plot_fit_no_hyper = plot_setting("hyper", "fit_no_hyper")

        self.include_2d = lensing_include.Include2D()

    def mat_plot_1d_from(self, subfolders, format="png"):
        return lensing_mat_plot.MatPlot1D(
            output=wrap_base.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def mat_plot_2d_from(self, subfolders, format="png"):
        return lensing_mat_plot.MatPlot2D(
            output=wrap_base.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def visualize_imaging(self, imaging):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="imaging")

        imaging_plotter = imaging_plotters.ImagingPlotter(
            imaging=imaging, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        imaging_plotter.figures(
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            psf=should_plot("psf"),
            inverse_noise_map=should_plot("inverse_noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            absolute_signal_to_noise_map=should_plot("absolute_signal_to_noise_map"),
            potential_chi_squared_map=should_plot("potential_chi_squared_map"),
        )

        if should_plot("subplot_dataset"):

            imaging_plotter.subplot_imaging()

    def visualize_fit_imaging(self, fit, during_analysis, subfolders="fit_imaging"):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        fit_imaging_plotter = fit_imaging_plotters.FitImagingPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        fit_imaging_plotter.figures(
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            model_image=should_plot("model_data"),
            residual_map=should_plot("residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
        )

        fit_imaging_plotter.figures_of_galaxies(
            subtracted_image=should_plot("subtracted_images_of_galaxies"),
            model_image=should_plot("model_images_of_galaxies"),
        )

        if should_plot("subplot_fit"):
            fit_imaging_plotter.subplot_fit_imaging()

        if should_plot("subplots_of_galaxies_fits"):
            fit_imaging_plotter.subplots_of_galaxies()

        if not during_analysis:

            if should_plot("all_at_end_png"):

                fit_imaging_plotter.figures(
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_image=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

                fit_imaging_plotter.figures_of_galaxies(
                    subtracted_image=True, model_image=True
                )

            if should_plot("all_at_end_fits"):

                mat_plot_2d = self.mat_plot_2d_from(
                    subfolders="fit_imaging/fits", format="fits"
                )

                fit_imaging_plotter = fit_imaging_plotters.FitImagingPlotter(
                    fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
                )

                fit_imaging_plotter.figures(
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_image=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

                fit_imaging_plotter.figures_of_galaxies(
                    subtracted_image=True, model_image=True
                )

    def visualize_interferometer(self, interferometer):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="interferometer")

        interferometer_plotter = interferometer_plotters.InterferometerPlotter(
            interferometer=interferometer,
            include_2d=self.include_2d,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_dataset"):
            interferometer_plotter.subplot_interferometer()

        interferometer_plotter.figures(
            visibilities=should_plot("data"),
            u_wavelengths=should_plot("uv_wavelengths"),
            v_wavelengths=should_plot("uv_wavelengths"),
        )

    def visualize_fit_interferometer(
        self, fit, during_analysis, subfolders="fit_interferometer"
    ):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=subfolders)
        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        fit_interferometer_plotter = fit_interferometer_plotters.FitInterferometerPlotter(
            fit=fit,
            include_2d=self.include_2d,
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_fit"):
            fit_interferometer_plotter.subplot_fit_interferometer()
            fit_interferometer_plotter.subplot_fit_real_space()

        fit_interferometer_plotter.figures(
            visibilities=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            model_visibilities=should_plot("model_data"),
            residual_map_real=should_plot("residual_map"),
            residual_map_imag=should_plot("residual_map"),
            chi_squared_map_real=should_plot("chi_squared_map"),
            chi_squared_map_imag=should_plot("chi_squared_map"),
            normalized_residual_map_real=should_plot("normalized_residual_map"),
            normalized_residual_map_imag=should_plot("normalized_residual_map"),
        )

        if not during_analysis:

            if should_plot("all_at_end_png"):

                fit_interferometer_plotter.figures(
                    visibilities=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_visibilities=True,
                    residual_map_real=True,
                    residual_map_imag=True,
                    chi_squared_map_real=True,
                    chi_squared_map_imag=True,
                    normalized_residual_map_real=True,
                    normalized_residual_map_imag=True,
                )

            if should_plot("all_at_end_fits"):

                mat_plot_2d = self.mat_plot_2d_from(
                    subfolders="fit_interferometer/fits", format="fits"
                )

                fit_interferometer_plotter = fit_interferometer_plotters.FitInterferometerPlotter(
                    fit=fit, include_2d=self.include_2d, mat_plot_2d=mat_plot_2d
                )

                fit_interferometer_plotter.figures(
                    visibilities=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_visibilities=True,
                    residual_map_real=True,
                    residual_map_imag=True,
                    chi_squared_map_real=True,
                    chi_squared_map_imag=True,
                    normalized_residual_map_real=True,
                    normalized_residual_map_imag=True,
                )

    def visualize_inversion(self, inversion, during_analysis):
        def should_plot(name):
            return plot_setting(section="inversion", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="inversion")

        inversion_plotter = inversion_plotters.InversionPlotter(
            inversion=inversion, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("subplot_inversion"):
            inversion_plotter.subplot_inversion()

        inversion_plotter.figures(
            reconstructed_image=should_plot("reconstructed_image"),
            reconstruction=should_plot("reconstruction"),
            errors=should_plot("errors"),
            residual_map=should_plot("residual_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
            regularization_weights=should_plot("regularization_weights"),
            interpolated_reconstruction=should_plot("interpolated_reconstruction"),
            interpolated_errors=should_plot("interpolated_errors"),
        )

        if not during_analysis:

            if should_plot("all_at_end_png"):

                inversion_plotter.figures(
                    reconstructed_image=True,
                    reconstruction=True,
                    errors=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                    regularization_weights=True,
                    interpolated_reconstruction=True,
                    interpolated_errors=True,
                )

    def visualize_hyper_images(
        self, hyper_galaxy_image_path_dict, hyper_model_image, plane
    ):
        def should_plot(name):
            return plot_setting(section="hyper", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="hyper")

        hyper_plotter = hyper_plotters.HyperPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("model_image"):
            hyper_plotter.figure_hyper_model_image(hyper_model_image=hyper_model_image)

        if should_plot("images_of_galaxies"):

            hyper_plotter.subplot_hyper_images_of_galaxies(
                hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict
            )

        if hasattr(plane, "contribution_maps_of_galaxies"):
            if should_plot("contribution_maps_of_galaxies"):
                hyper_plotter.subplot_contribution_maps_of_galaxies(
                    contribution_maps_of_galaxies=plane.contribution_maps_of_galaxies
                )

    def visualize_galaxy_fit(self, fit, visuals_2d=None):
        def should_plot(name):
            return plot_setting(section="galaxy_fit", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="galaxy_fit")

        fit_galaxy_plotter = fit_galaxy_plotters.FitGalaxyPlotter(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=self.include_2d,
        )

        if should_plot("subplot_galaxy_fit"):
            fit_galaxy_plotter.subplot_fit_galaxy()

        mat_plot_2d = self.mat_plot_2d_from(subfolders="galaxy_fit")

        fit_galaxy_plotter = fit_galaxy_plotters.FitGalaxyPlotter(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=self.include_2d,
        )

        fit_galaxy_plotter.figures(
            image=should_plot("image"),
            noise_map=should_plot("noise_map"),
            model_image=should_plot("model_image"),
            residual_map=should_plot("residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
        )

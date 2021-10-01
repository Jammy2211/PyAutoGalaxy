import os
from os import path

from autoconf import conf
import autoarray.plot as aplt

from autogalaxy.galaxy.plot.fit_galaxy_plotters import FitGalaxyPlotter
from autogalaxy.galaxy.plot.hyper_galaxy_plotters import HyperPlotter

from autogalaxy.plot.mat_wrap.lensing_include import Include2D
from autogalaxy.plot.mat_wrap.lensing_mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.lensing_mat_plot import MatPlot2D


def setting(section, name):
    return conf.instance["visualize"]["plots"][section][name]


def plot_setting(section, name):
    return setting(section, name)


class Visualizer:
    def __init__(self, visualize_path):

        self.visualize_path = visualize_path

        self.plot_fit_no_hyper = plot_setting("hyper", "fit_no_hyper")

        self.include_2d = Include2D()

        try:
            os.makedirs(visualize_path)
        except FileExistsError:
            pass

    def mat_plot_1d_from(self, subfolders, format="png"):
        return MatPlot1D(
            output=aplt.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def mat_plot_2d_from(self, subfolders, format="png"):
        return MatPlot2D(
            output=aplt.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def visualize_imaging(self, imaging):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="imaging")

        imaging_plotter = aplt.ImagingPlotter(
            imaging=imaging, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        imaging_plotter.figures_2d(
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

    def visualize_interferometer(self, interferometer):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="interferometer")

        interferometer_plotter = aplt.InterferometerPlotter(
            interferometer=interferometer,
            include_2d=self.include_2d,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_dataset"):
            interferometer_plotter.subplot_interferometer()

        interferometer_plotter.figures_2d(
            visibilities=should_plot("data"),
            u_wavelengths=should_plot("uv_wavelengths"),
            v_wavelengths=should_plot("uv_wavelengths"),
        )

    def visualize_inversion(self, inversion, during_analysis):
        def should_plot(name):
            return plot_setting(section="inversion", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="inversion")

        inversion_plotter = aplt.InversionPlotter(
            inversion=inversion, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("subplot_inversion"):
            for mapper_index in range(len(inversion.mapper_list)):
                inversion_plotter.subplot_of_mapper(mapper_index=mapper_index)

        inversion_plotter.figures_2d(
            reconstructed_image=should_plot("reconstructed_image")
        )

        inversion_plotter.figures_2d_of_mapper(
            mapper_index=0,
            reconstructed_image=should_plot("reconstructed_image"),
            reconstruction=should_plot("reconstruction"),
            errors=should_plot("errors"),
            residual_map=should_plot("residual_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
            regularization_weights=should_plot("regularization_weights"),
        )

        if not during_analysis:

            if should_plot("all_at_end_png"):

                inversion_plotter.figures_2d(reconstructed_image=True)

                inversion_plotter.figures_2d_of_mapper(
                    mapper_index=0,
                    reconstructed_image=True,
                    reconstruction=True,
                    errors=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                    regularization_weights=True,
                )

    def visualize_hyper_images(self, hyper_galaxy_image_path_dict, hyper_model_image):
        def should_plot(name):
            return plot_setting(section="hyper", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="hyper")

        hyper_plotter = HyperPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("model_image"):
            hyper_plotter.figure_hyper_model_image(hyper_model_image=hyper_model_image)

        if should_plot("images_of_galaxies"):

            hyper_plotter.subplot_hyper_images_of_galaxies(
                hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict
            )

    def visualize_contribution_maps(self, plane):
        def should_plot(name):
            return plot_setting(section="hyper", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="hyper")

        hyper_plotter = HyperPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
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

        fit_galaxy_plotter = FitGalaxyPlotter(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=self.include_2d,
        )

        if should_plot("subplot_galaxy_fit"):
            fit_galaxy_plotter.subplot_fit_galaxy()

        mat_plot_2d = self.mat_plot_2d_from(subfolders="galaxy_fit")

        fit_galaxy_plotter = FitGalaxyPlotter(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=self.include_2d,
        )

        fit_galaxy_plotter.figures_2d(
            image=should_plot("image"),
            noise_map=should_plot("noise_map"),
            model_image=should_plot("model_image"),
            residual_map=should_plot("residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
        )

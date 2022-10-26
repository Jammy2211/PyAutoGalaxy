from os import path

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.imaging.plot.fit_imaging_plotters import FitImagingPlotter
from autogalaxy.analysis.visualizer import Visualizer

from autogalaxy.analysis.visualizer import plot_setting


class VisualizerImaging(Visualizer):
    def visualize_fit_imaging(
        self, fit: FitImaging, during_analysis: bool, subfolders: str = "fit_imaging"
    ):
        """
        Visualizes a `FitImaging` object, which fits an imaging dataset.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `fit`. When
        used with a non-linear search the `visualize_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `FitImaging` inferred by the search so far.

        Visualization includes individual images of attributes of the `FitImaging` (e.g. the model data, residual map)
        and a subplot of all `FitImaging`'s images on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [fit] header.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitImaging` of the non-linear search which is used to plot the fit.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        visuals_2d
            An object containing attributes which may be plotted over the figure (e.g. the centres of mass and light
            profiles).
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        fit_imaging_plotter = FitImagingPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        fit_imaging_plotter.figures_2d(
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            model_image=should_plot("model_data"),
            residual_map=should_plot("residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
        )

        fit_imaging_plotter.figures_2d_of_galaxies(
            subtracted_image=should_plot("subtracted_images_of_galaxies"),
            model_image=should_plot("model_images_of_galaxies"),
        )

        if should_plot("subplot_fit"):
            fit_imaging_plotter.subplot_fit_imaging()

        if should_plot("subplot_of_galaxies"):
            fit_imaging_plotter.subplot_of_galaxies()

        if not during_analysis:

            if should_plot("all_at_end_png"):

                fit_imaging_plotter.figures_2d(
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_image=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

                fit_imaging_plotter.figures_2d_of_galaxies(
                    subtracted_image=True, model_image=True
                )

            if should_plot("all_at_end_fits"):

                mat_plot_2d = self.mat_plot_2d_from(
                    subfolders=path.join("fit_imaging", "fits"), format="fits"
                )

                fit_imaging_plotter = FitImagingPlotter(
                    fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
                )

                fit_imaging_plotter.figures_2d(
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_image=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

                fit_imaging_plotter.figures_2d_of_galaxies(
                    subtracted_image=True, model_image=True
                )

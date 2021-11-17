from autogalaxy.quantity.fit_quantity import FitQuantity
from autogalaxy.quantity.plot.fit_quantity_plotters import FitQuantityPlotter
from autogalaxy.analysis.visualizer import Visualizer
from autogalaxy.analysis.visualizer import plot_setting
from autogalaxy.plot.mat_wrap.visuals import Visuals2D


class VisualizerQuantity(Visualizer):
    def visualize_fit_quantity(
        self,
        fit: FitQuantity,
        visuals_2d: Visuals2D = Visuals2D(),
        fit_quanaity_plotter_cls=FitQuantityPlotter,
    ):
        """
        Visualizes a `FitQuantity` object, which fits a quantity of a light or mass profile (e.g. an image, potential)
        to the same quantity of another light or mass profile.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `fit_quantity`. When
        used with a non-linear search the `visualize_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `FitQuantity` inferred by the search so far.

        Visualization includes individual images of attributes of the `FitQuantity` (e.g. the model data, residual map)
        and a subplot of all `FitQuantity`'s images on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [fit_quantity] header.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitQuantity` of the non-linear search which is used to plot the fit.
        visuals_2d
            An object containing attributes which may be plotted over the figure (e.g. the centres of mass and light
            profiles).
        """

        def should_plot(name):
            return plot_setting(section="fit_quantity", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="fit_quantity")

        fit_quantity_plotter = fit_quanaity_plotter_cls(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=self.include_2d,
        )

        if should_plot("subplot_fit"):
            fit_quantity_plotter.subplot_fit_quantity()

        mat_plot_2d = self.mat_plot_2d_from(subfolders="fit_quantity")

        fit_quantity_plotter = FitQuantityPlotter(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=self.include_2d,
        )

        fit_quantity_plotter.figures_2d(
            image=should_plot("image"),
            noise_map=should_plot("noise_map"),
            model_image=should_plot("model_image"),
            residual_map=should_plot("residual_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
        )

        if should_plot("all_at_end_fits"):

            mat_plot_2d = self.mat_plot_2d_from(
                subfolders="fit_quantity/fits", format="fits"
            )

            fit_imaging_plotter = FitQuantityPlotter(
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

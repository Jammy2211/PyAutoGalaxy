import autoarray as aa

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autogalaxy.quantity.fit_quantity import FitQuantity

from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.two_d import Visuals2D


# TODO : Ew, this is a mass, but it works. Clean up one day!


class FitQuantityPlotter(Plotter):
    def __init__(
        self,
        fit: FitQuantity,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        """
        Plots the attributes of `FitQuantity` objects using the matplotlib method `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2d` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `FitQuantity` and plotted via the visuals object.

        Parameters
        ----------
        fit
            The fit to an interferometer dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        """
        super().__init__(mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d)

        self.fit = fit

    def figures_2d(
        self,
        image: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `FitImaging` object in 2D.

        The API is such that every plottable attribute of the `FitImaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether to make a 2D plot (via `imshow`) of the noise map.
        signal_to_noise_map
            Whether to make a 2D plot (via `imshow`) of the signal-to-noise map.
        model_image
            Whether to make a 2D plot (via `imshow`) of the model image.
        residual_map
            Whether to make a 2D plot (via `imshow`) of the residual map.
        normalized_residual_map
            Whether to make a 2D plot (via `imshow`) of the normalized residual map.
        chi_squared_map
            Whether to make a 2D plot (via `imshow`) of the chi-squared map.
        """

        if isinstance(self.fit.dataset.data, aa.Array2D):
            fit_plotter = FitImagingPlotterMeta(
                fit=self.fit,
                mat_plot_2d=self.mat_plot_2d,
                visuals_2d=self.visuals_2d,
            )

            fit_plotter.figures_2d(
                data=image,
                noise_map=noise_map,
                signal_to_noise_map=signal_to_noise_map,
                model_image=model_image,
                residual_map=residual_map,
                normalized_residual_map=normalized_residual_map,
                chi_squared_map=chi_squared_map,
            )

        else:
            fit_plotter_y = FitImagingPlotterMeta(
                fit=self.fit.y,
                mat_plot_2d=self.mat_plot_2d,
                visuals_2d=self.visuals_2d,
            )

            fit_plotter_y.figures_2d(
                data=image,
                noise_map=noise_map,
                signal_to_noise_map=signal_to_noise_map,
                model_image=model_image,
                residual_map=residual_map,
                normalized_residual_map=normalized_residual_map,
                chi_squared_map=chi_squared_map,
                suffix="_y",
            )

            fit_plotter_x = FitImagingPlotterMeta(
                fit=self.fit.y,
                mat_plot_2d=self.mat_plot_2d,
                visuals_2d=self.visuals_2d,
            )

            fit_plotter_x.figures_2d(
                data=image,
                noise_map=noise_map,
                signal_to_noise_map=signal_to_noise_map,
                model_image=model_image,
                residual_map=residual_map,
                normalized_residual_map=normalized_residual_map,
                chi_squared_map=chi_squared_map,
                suffix="_x",
            )

    def subplot_fit(self):
        """
        Standard subplot of the attributes of the plotter's `FitQuantity` object.
        """

        if isinstance(self.fit.dataset.data, aa.Array2D):
            fit_plotter = FitImagingPlotterMeta(
                fit=self.fit,
                mat_plot_2d=self.mat_plot_2d,
                visuals_2d=self.visuals_2d,
            )

            fit_plotter.subplot(
                data=True,
                signal_to_noise_map=True,
                model_image=True,
                residual_map=True,
                normalized_residual_map=True,
                chi_squared_map=True,
                auto_filename="subplot_fit",
            )

        else:
            fit_plotter_y = FitImagingPlotterMeta(
                fit=self.fit.y,
                mat_plot_2d=self.mat_plot_2d,
                visuals_2d=self.visuals_2d,
            )

            fit_plotter_y.subplot(
                data=True,
                signal_to_noise_map=True,
                model_image=True,
                residual_map=True,
                normalized_residual_map=True,
                chi_squared_map=True,
                auto_filename="subplot_fit_y",
            )

            fit_plotter_x = FitImagingPlotterMeta(
                fit=self.fit.x,
                mat_plot_2d=self.mat_plot_2d,
                visuals_2d=self.visuals_2d,
            )

            fit_plotter_x.subplot(
                data=True,
                signal_to_noise_map=True,
                model_image=True,
                residual_map=True,
                normalized_residual_map=True,
                chi_squared_map=True,
                auto_filename="subplot_fit_x",
            )

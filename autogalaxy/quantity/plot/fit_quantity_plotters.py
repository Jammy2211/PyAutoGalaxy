import autoarray as aa

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autogalaxy.quantity.fit_quantity import FitQuantity

from autogalaxy.plot.abstract_plotters import Plotter, _to_positions
from autogalaxy.plot.mat_plot.two_d import MatPlot2D


# TODO : Ew, this is a mass, but it works. Clean up one day!


class FitQuantityPlotter(Plotter):
    def __init__(
        self,
        fit: FitQuantity,
        mat_plot_2d: MatPlot2D = None,
        positions=None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d)

        self.fit = fit
        self.positions = positions

    def _make_positions(self):
        return _to_positions(self.positions)

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
        if isinstance(self.fit.dataset.data, aa.Array2D):
            fit_plotter = FitImagingPlotterMeta(
                fit=self.fit,
                mat_plot_2d=self.mat_plot_2d,
                positions=self._make_positions(),
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
                positions=self._make_positions(),
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
                fit=self.fit.x,
                mat_plot_2d=self.mat_plot_2d,
                positions=self._make_positions(),
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
        if isinstance(self.fit.dataset.data, aa.Array2D):
            fit_plotter = FitImagingPlotterMeta(
                fit=self.fit,
                mat_plot_2d=self.mat_plot_2d,
                positions=self._make_positions(),
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
                positions=self._make_positions(),
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
                positions=self._make_positions(),
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

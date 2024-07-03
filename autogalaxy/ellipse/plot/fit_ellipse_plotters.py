from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.include.one_d import Include1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.two_d import Visuals2D
from autogalaxy.plot.include.two_d import Include2D


class FitEllipsePlotter(Plotter):
    def __init__(
        self,
        fit: FitEllipse,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

        self.fit = fit

    def get_visuals_1d(self) -> Visuals1D:
        return self.visuals_1d

    def figures_1d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_data: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        suffix: str = "",
    ):
        """
        Plots the individual attributes of the plotter's `FitEllipse` object in 1D.

        The API is such that every plottable attribute of the `FitEllipse` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        data
            Whether to make a 1D plot (via `imshow`) of the image data.
        noise_map
            Whether to make a 1D plot (via `imshow`) of the noise map.
        signal_to_noise_map
            Whether to make a 1D plot (via `imshow`) of the signal-to-noise map.
        model_image
            Whether to make a 1D plot (via `imshow`) of the model image.
        residual_map
            Whether to make a 1D plot (via `imshow`) of the residual map.
        normalized_residual_map
            Whether to make a 1D plot (via `imshow`) of the normalized residual map.
        chi_squared_map
            Whether to make a 1D plot (via `imshow`) of the chi-squared map.
        """

        if data:
            self.mat_plot_1d.plot_yx(
                array=self.fit.data,
                visuals_2d=self.get_visuals_1d(),
                auto_labels=AutoLabels(title="Data", filename=f"data{suffix}"),
            )

        if noise_map:
            self.mat_plot_1d.plot_yx(
                array=self.fit.noise_map,
                visuals_2d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Noise-Map", filename=f"noise_map{suffix}"
                ),
            )

        if signal_to_noise_map:
            self.mat_plot_1d.plot_yx(
                array=self.fit.signal_to_noise_map,
                visuals_2d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", filename=f"signal_to_noise_map{suffix}"
                ),
            )

        if model_data:
            self.mat_plot_1d.plot_yx(
                array=self.fit.model_data,
                visuals_2d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Model Image", filename=f"model_image{suffix}"
                ),
            )

        cmap_original = self.mat_plot_1d.cmap

        if self.residuals_symmetric_cmap:
            self.mat_plot_1d.cmap = self.mat_plot_1d.cmap.symmetric_cmap_from()

        if residual_map:
            self.mat_plot_1d.plot_yx(
                array=self.fit.residual_map,
                visuals_2d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Residual Map", filename=f"residual_map{suffix}"
                ),
            )

        if normalized_residual_map:
            self.mat_plot_1d.plot_yx(
                array=self.fit.normalized_residual_map,
                visuals_2d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Normalized Residual Map",
                    filename=f"normalized_residual_map{suffix}",
                ),
            )

        self.mat_plot_1d.cmap = cmap_original

        if chi_squared_map:
            self.mat_plot_1d.plot_yx(
                array=self.fit.chi_squared_map,
                visuals_2d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Chi-Squared Map", filename=f"chi_squared_map{suffix}"
                ),
            )

        if residual_flux_fraction_map:
            self.mat_plot_1d.plot_yx(
                array=self.fit.residual_map,
                visuals_2d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Residual Flux Fraction Map",
                    filename=f"residual_flux_fraction_map{suffix}",
                ),
            )

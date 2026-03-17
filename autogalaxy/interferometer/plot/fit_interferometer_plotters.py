from typing import List

import autoarray as aa
import autoarray.plot as aplt

from autoarray.fit.plot.fit_interferometer_plotters import FitInterferometerPlotterMeta

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D

from autogalaxy.galaxy.plot.galaxies_plotters import GalaxiesPlotter


class FitInterferometerPlotter(Plotter):
    def __init__(
        self,
        fit: FitInterferometer,
        mat_plot_1d: MatPlot1D = None,
        mat_plot_2d: MatPlot2D = None,
        positions=None,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        self.fit = fit
        self.positions = positions

        from autogalaxy.plot.visuals.one_d import Visuals1D
        from autogalaxy.plot.visuals.two_d import Visuals2D

        self._fit_interferometer_meta_plotter = FitInterferometerPlotterMeta(
            fit=self.fit,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=Visuals1D(),
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=Visuals2D(positions=positions),
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.figures_2d = self._fit_interferometer_meta_plotter.figures_2d
        self.subplot = self._fit_interferometer_meta_plotter.subplot
        self.subplot_fit_dirty_images = (
            self._fit_interferometer_meta_plotter.subplot_fit_dirty_images
        )

    @property
    def galaxies(self) -> List[Galaxy]:
        return self.fit.galaxies_linear_light_profiles_to_light_profiles

    def galaxies_plotter_from(self, galaxies: List[Galaxy]) -> GalaxiesPlotter:
        return GalaxiesPlotter(
            galaxies=galaxies,
            grid=self.fit.grids.lp,
            mat_plot_2d=self.mat_plot_2d,
        )

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
        )

    def subplot_fit(self):
        self.open_subplot_figure(number_subplots=9)

        self.figures_2d(amplitudes_vs_uv_distances=True)

        self.mat_plot_1d.subplot_index = 2
        self.mat_plot_2d.subplot_index = 2

        self.figures_2d(dirty_image=True)
        self.figures_2d(dirty_signal_to_noise_map=True)

        self.mat_plot_1d.subplot_index = 4
        self.mat_plot_2d.subplot_index = 4

        self.figures_2d(dirty_model_image=True)

        self.mat_plot_1d.subplot_index = 5
        self.mat_plot_2d.subplot_index = 5

        self.figures_2d(normalized_residual_map_real=True)
        self.figures_2d(normalized_residual_map_imag=True)

        self.mat_plot_1d.subplot_index = 7
        self.mat_plot_2d.subplot_index = 7

        self.figures_2d(dirty_normalized_residual_map=True)

        self.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
        self.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

        self.set_title(label=r"Normalized Residual Map $1\sigma$")
        self.figures_2d(dirty_normalized_residual_map=True)
        self.set_title(label=None)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.figures_2d(dirty_chi_squared_map=True)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_fit")
        self.close_subplot_figure()

    def subplot_fit_real_space(self):
        if not self.galaxies.has(cls=aa.Pixelization):
            galaxies_plotter = self.galaxies_plotter_from(galaxies=self.galaxies)

            galaxies_plotter.subplot(image=True, auto_filename="subplot_fit_real_space")

        elif self.galaxies.has(cls=aa.Pixelization):
            self.open_subplot_figure(number_subplots=6)

            mapper_index = 0

            self.inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=mapper_index, reconstructed_operated_data=True
            )
            self.inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=mapper_index, reconstruction=True
            )

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_fit_real_space"
            )

            self.close_subplot_figure()

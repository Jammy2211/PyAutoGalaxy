from typing import List

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap

import autoarray as aa
import autoarray.plot as aplt

from autoarray.fit.plot.fit_interferometer_plotters import FitInterferometerPlotterMeta

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.galaxy.plot.galaxies_plotters import GalaxiesPlotter


class FitInterferometerPlotter(Plotter):
    def __init__(
        self,
        fit: FitInterferometer,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        positions=None,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.fit = fit
        self.positions = positions

        self._fit_interferometer_meta_plotter = FitInterferometerPlotterMeta(
            fit=self.fit,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.figures_2d = self._fit_interferometer_meta_plotter.figures_2d
        self.subplot_fit = self._fit_interferometer_meta_plotter.subplot_fit
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
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
        )

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
        )

    def subplot_fit_real_space(self):
        if not self.galaxies.has(cls=aa.Pixelization):
            galaxies_plotter = self.galaxies_plotter_from(galaxies=self.galaxies)
            galaxies_plotter.subplot(image=True, auto_filename="subplot_fit_real_space")
        else:
            self.inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=0, reconstructed_operated_data=True
            )
            self.inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=0, reconstruction=True
            )

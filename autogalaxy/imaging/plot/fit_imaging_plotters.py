import matplotlib.pyplot as plt
from typing import List, Optional

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap

import autoarray as aa
import autoarray.plot as aplt

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.plot.abstract_plotters import Plotter, _to_positions, _save_subplot


class FitImagingPlotter(Plotter):
    def __init__(
        self,
        fit: FitImaging,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        positions=None,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.fit = fit
        self.positions = positions

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            positions=_to_positions(positions),
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot_fit = self._fit_imaging_meta_plotter.subplot_fit

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
        )

    @property
    def galaxies(self) -> List[Galaxy]:
        return self.fit.galaxies_linear_light_profiles_to_light_profiles

    @property
    def galaxy_indices(self) -> List[int]:
        return list(range(len(self.fit.galaxies)))

    def figures_2d_of_galaxies(
        self,
        galaxy_index: Optional[int] = None,
        subtracted_image: bool = False,
        model_image: bool = False,
    ):
        if galaxy_index is None:
            galaxy_indices = self.galaxy_indices
        else:
            galaxy_indices = [galaxy_index]

        positions = _to_positions(self.positions)

        for galaxy_index in galaxy_indices:
            if subtracted_image:
                self._plot_array(
                    array=self.fit.subtracted_images_of_galaxies_list[galaxy_index],
                    auto_filename=f"subtracted_image_of_galaxy_{galaxy_index}",
                    title=f"Subtracted Image of Galaxy {galaxy_index}",
                    positions=positions,
                )

            if model_image:
                self._plot_array(
                    array=self.fit.model_images_of_galaxies_list[galaxy_index],
                    auto_filename=f"model_image_of_galaxy_{galaxy_index}",
                    title=f"Model Image of Galaxy {galaxy_index}",
                    positions=positions,
                )

    def subplot_of_galaxies(self, galaxy_index: Optional[int] = None):
        if galaxy_index is None:
            galaxy_indices = self.galaxy_indices
        else:
            galaxy_indices = [galaxy_index]

        for galaxy_index in galaxy_indices:
            has_pix = self.galaxies.has(cls=aa.Pixelization)
            n = 4 if has_pix else 3
            fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
            axes_flat = list(axes.flatten())

            self._fit_imaging_meta_plotter._plot_array(
                self.fit.data, "data", "Data", ax=axes_flat[0]
            )
            self._fit_imaging_meta_plotter._plot_array(
                self.fit.subtracted_images_of_galaxies_list[galaxy_index],
                f"subtracted_image_of_galaxy_{galaxy_index}",
                f"Subtracted Image of Galaxy {galaxy_index}",
                ax=axes_flat[1],
            )
            self._fit_imaging_meta_plotter._plot_array(
                self.fit.model_images_of_galaxies_list[galaxy_index],
                f"model_image_of_galaxy_{galaxy_index}",
                f"Model Image of Galaxy {galaxy_index}",
                ax=axes_flat[2],
            )

            if has_pix:
                self.inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=0, reconstruction=True
                )

            plt.tight_layout()
            _save_subplot(fig, self.output, f"subplot_of_galaxy_{galaxy_index}")

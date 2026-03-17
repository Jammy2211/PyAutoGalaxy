from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.two_d import MatPlot2D


class FitImagingPlotter(Plotter):
    def __init__(
        self,
        fit: FitImaging,
        mat_plot_2d: MatPlot2D = None,
        positions=None,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d)

        self.fit = fit
        self.positions = positions

        from autogalaxy.plot.visuals.two_d import Visuals2D

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=Visuals2D(positions=positions),
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot = self._fit_imaging_meta_plotter.subplot

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
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

        for galaxy_index in galaxy_indices:
            from autogalaxy.plot.visuals.two_d import Visuals2D

            if subtracted_image:
                self._plot_array(
                    array=self.fit.subtracted_images_of_galaxies_list[galaxy_index],
                    visuals_2d=Visuals2D(positions=self.positions),
                    auto_labels=aplt.AutoLabels(
                        title=f"Subtracted Image of Galaxy {galaxy_index}",
                        filename=f"subtracted_image_of_galaxy_{galaxy_index}",
                    ),
                )

            if model_image:
                self._plot_array(
                    array=self.fit.model_images_of_galaxies_list[galaxy_index],
                    visuals_2d=Visuals2D(positions=self.positions),
                    auto_labels=aplt.AutoLabels(
                        title=f"Model Image of Galaxy {galaxy_index}",
                        filename=f"model_image_of_galaxy_{galaxy_index}",
                    ),
                )

    def subplot_fit(self):
        self.open_subplot_figure(number_subplots=6)

        self.figures_2d(data=True)

        self.figures_2d(signal_to_noise_map=True)
        self.figures_2d(model_image=True)
        self.figures_2d(normalized_residual_map=True)

        self.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
        self.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

        self.set_title(label=r"Normalized Residual Map $1\sigma$")
        self.figures_2d(normalized_residual_map=True)
        self.set_title(label=None)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.figures_2d(chi_squared_map=True)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_fit")
        self.close_subplot_figure()

    def subplot_of_galaxies(self, galaxy_index: Optional[int] = None):
        if galaxy_index is None:
            galaxy_indices = self.galaxy_indices
        else:
            galaxy_indices = [galaxy_index]

        for galaxy_index in galaxy_indices:
            self.open_subplot_figure(number_subplots=4)

            self.figures_2d(data=True)
            self.figures_2d_of_galaxies(
                galaxy_index=galaxy_index, subtracted_image=True
            )
            self.figures_2d_of_galaxies(galaxy_index=galaxy_index, model_image=True)

            if self.galaxies.has(cls=aa.Pixelization):
                self.inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=0, reconstruction=True
                )

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_of_galaxy_{galaxy_index}"
            )
            self.close_subplot_figure()

import numpy as np
from typing import List, Optional

import autoarray.plot as aplt

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autogalaxy.plane.plane import Plane
from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include2D


class FitImagingPlotter(Plotter):
    def __init__(
        self,
        fit: FitImaging,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):

        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.fit = fit

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot = self._fit_imaging_meta_plotter.subplot
        self.subplot_fit_imaging = self._fit_imaging_meta_plotter.subplot_fit_imaging

    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_fit_from(fit=self.fit)

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        return aplt.InversionPlotter(
            inversion=self.fit.inversion.linear_eqn,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d(),
            include_2d=self.include_2d,
        )

    @property
    def plane(self) -> Plane:
        return self.fit.plane

    def galaxy_indexes_from(self, galaxy_index: Optional[int]) -> List[int]:

        if galaxy_index is None:
            return list(range(len(self.fit.galaxies)))
        return [galaxy_index]

    def figures_2d_of_galaxies(
        self,
        subtracted_image: bool = False,
        model_image: bool = False,
        galaxy_index: Optional[int] = None,
    ):

        galaxy_indexes = self.galaxy_indexes_from(galaxy_index=galaxy_index)

        for galaxy_index in galaxy_indexes:

            if subtracted_image:

                self.mat_plot_2d.cmap.kwargs["vmin"] = np.max(
                    self.fit.model_images_of_galaxies[galaxy_index]
                )
                self.mat_plot_2d.cmap.kwargs["vmin"] = np.min(
                    self.fit.model_images_of_galaxies[galaxy_index]
                )

                self.mat_plot_2d.plot_array(
                    array=self.fit.subtracted_images_of_galaxies[galaxy_index],
                    visuals_2d=self.get_visuals_2d(),
                    auto_labels=aplt.AutoLabels(
                        title=f"Subtracted Image of Galaxy {galaxy_index}",
                        filename=f"subtracted_image_of_galaxy_{galaxy_index}",
                    ),
                )

            if model_image:

                self.mat_plot_2d.plot_array(
                    array=self.fit.model_images_of_galaxies[galaxy_index],
                    visuals_2d=self.get_visuals_2d(),
                    auto_labels=aplt.AutoLabels(
                        title=f"Model Image of Galaxy {galaxy_index}",
                        filename=f"model_image_of_galaxy_{galaxy_index}",
                    ),
                )

    def subplots_of_galaxies(self, galaxy_index: Optional[int] = None):
        """Plot the model data of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autogalaxy.lens.fitting.Fitter
            Class containing fit between the model data and observed lens data (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the data is output if the output_type is a file format (e.g. png, fits)
        output_filename : str
            The name of the file that is output, if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the data is output. File formats (e.g. png, fits) output the data to harddisk. 'show' displays the data \
            in the python interpreter window.
        """

        galaxy_indexes = self.galaxy_indexes_from(galaxy_index=galaxy_index)

        for galaxy_index in galaxy_indexes:

            self.open_subplot_figure(number_subplots=4)

            self.figures_2d(image=True)
            self.figures_2d_of_galaxies(
                galaxy_index=galaxy_index, subtracted_image=True
            )
            self.figures_2d_of_galaxies(galaxy_index=galaxy_index)

            if self.plane.has_pixelization:
                self.inversion_plotter.figures_2d(reconstruction=True)

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_of_galaxy_{galaxy_index}"
            )
            self.close_subplot_figure()

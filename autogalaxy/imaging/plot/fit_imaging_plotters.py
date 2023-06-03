import numpy as np
from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autogalaxy.plane.plane import Plane
from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.two_d import Visuals2D
from autogalaxy.plot.include.two_d import Include2D


class FitImagingPlotter(Plotter):
    def __init__(
        self,
        fit: FitImaging,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
        residuals_symmetric_cmap: bool = True,
    ):
        """
        Plots the attributes of `FitImaging` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `FitImaging` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an imaging dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        include_2d
            Specifies which attributes of the `FitImaging` are extracted and plotted as visuals.
        residuals_symmetric_cmap
            If true, the `residual_map` and `normalized_residual_map` are plotted with a symmetric color map such
            that `abs(vmin) = abs(vmax)`.
        """
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
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot = self._fit_imaging_meta_plotter.subplot

    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_fit_imaging_from(fit=self.fit)

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        """
        Returns an `InversionPlotter` corresponding to the `Inversion` of the fit.

        Returns
        -------
        InversionPlotter
            An object that plots inversions which is used for plotting attributes of the inversion.
        """
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d(),
            include_2d=self.include_2d,
        )

    @property
    def plane(self) -> Plane:
        return self.fit.plane_linear_light_profiles_to_light_profiles

    @property
    def galaxy_indices(self) -> List[int]:
        """
        Returns a list of all indexes of the galaxies in the fit, which is iterated over in figures that plot
        individual figures of each galaxy in a plane.


        Parameters
        ----------
        galaxy_index
            A specific galaxy index such that only a single galaxy index is returned.

        Returns
        -------
        list
            A list of galaxy indexes corresponding to galaxies in the plane.
        """
        return list(range(len(self.fit.galaxies)))

    def figures_2d_of_galaxies(
        self,
        galaxy_index: Optional[int] = None,
        subtracted_image: bool = False,
        model_image: bool = False,
    ):
        """
        Plots images representing each individual `Galaxy` in the plotter's `Plane` in 2D, which are computed via the
        plotter's 2D grid object.

        These images subtract or omit the contribution of other galaxies in the plane, such that plots showing
        each individual galaxy are made.

        The API is such that every plottable attribute of the `Galaxy` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        subtracted_image
            Whether to make a 2D plot (via `imshow`) of the subtracted image of a galaxy, where this image is
            the fit's `data` minus the model images of all other galaxies, thereby showing an individual galaxy in the
            data.
        model_image
            Whether to make a 2D plot (via `imshow`) of the model image of a galaxy, where this image is the
            model image of one galaxy, thereby showing how much it contributes to the overall model image.
        galaxy_index
            If input, plots for only a single galaxy based on its index in the plane are created.
        """
        if galaxy_index is None:
            galaxy_indices = self.galaxy_indices
        else:
            galaxy_indices = [galaxy_index]

        for galaxy_index in galaxy_indices:
            if subtracted_image:
                self.mat_plot_2d.cmap.kwargs["vmin"] = np.max(
                    self.fit.model_images_of_galaxies_list[galaxy_index]
                )
                self.mat_plot_2d.cmap.kwargs["vmin"] = np.min(
                    self.fit.model_images_of_galaxies_list[galaxy_index]
                )

                self.mat_plot_2d.plot_array(
                    array=self.fit.subtracted_images_of_galaxies_list[galaxy_index],
                    visuals_2d=self.get_visuals_2d(),
                    auto_labels=aplt.AutoLabels(
                        title=f"Subtracted Image of Galaxy {galaxy_index}",
                        filename=f"subtracted_image_of_galaxy_{galaxy_index}",
                    ),
                )

            if model_image:
                self.mat_plot_2d.plot_array(
                    array=self.fit.model_images_of_galaxies_list[galaxy_index],
                    visuals_2d=self.get_visuals_2d(),
                    auto_labels=aplt.AutoLabels(
                        title=f"Model Image of Galaxy {galaxy_index}",
                        filename=f"model_image_of_galaxy_{galaxy_index}",
                    ),
                )

    def subplot_fit(self):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """

        self.open_subplot_figure(number_subplots=6)

        self.figures_2d(data=True)

        self.figures_2d(signal_to_noise_map=True)
        self.figures_2d(model_image=True)
        self.figures_2d(normalized_residual_map=True)

        self.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
        self.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

        self.set_title(label="Normalized Residual Map (1 sigma)")
        self.figures_2d(normalized_residual_map=True)
        self.set_title(label=None)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.figures_2d(chi_squared_map=True)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_fit")
        self.close_subplot_figure()

    def subplot_of_galaxies(self, galaxy_index: Optional[int] = None):
        """
        Plots images representing each individual `Galaxy` in the plotter's `Plane` in 2D on a subplot, which are
        computed via the plotter's 2D grid object.

        These images subtract or omit the contribution of other galaxies in the plane, such that plots showing
        each individual galaxy are made.

        The subplot plots the subtracted image and model image of each galaxy, where are described in the
        `figures_2d_of_galaxies` function.

        Parameters
        ----------
        galaxy_index
            If input, plots for only a single galaxy based on its index in the plane are created.
        """

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

            if self.plane.has(cls=aa.Pixelization):
                self.inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=0, reconstruction=True
                )

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_of_galaxy_{galaxy_index}"
            )
            self.close_subplot_figure()

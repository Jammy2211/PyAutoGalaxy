from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.visuals.two_d import Visuals2D
from autogalaxy.plot.mass_plotter import MassPlotter
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.galaxy.plot.galaxy_plotters import GalaxyPlotter

from autogalaxy import exc


class GalaxiesPlotter(Plotter):
    def __init__(
        self,
        galaxies: List[Galaxy],
        grid: aa.type.Grid1D2DLike,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        """
        Plots the attributes of a list of galaxies using the matplotlib methods `plot()` and `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `MassProfile` and plotted via the visuals object.

        Parameters
        ----------
        galaxies
            The galaxies the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the galaxies light and mass quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        """

        self.galaxies = Galaxies(galaxies=galaxies)

        from autogalaxy.profiles.light.linear import (
            LightProfileLinear,
        )

        if self.galaxies.has(cls=LightProfileLinear):
            raise exc.raise_linear_light_profile_in_plot(
                plotter_type=self.__class__.__name__,
            )

        super().__init__(
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
        )

        self.grid = grid

        self._mass_plotter = MassPlotter(
            mass_obj=self.galaxies,
            grid=self.grid,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
        )

    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_light_mass_obj_from(
            light_mass_obj=self.galaxies, grid=self.grid
        )

    def get_visuals_2d_of_galaxy(self, galaxy_index: int) -> aplt.Visuals2D:
        return self.get_2d.via_galaxies_from(
            galaxies=self.galaxies, grid=self.grid, galaxy_index=galaxy_index
        )

    def galaxy_plotter_from(self, galaxy_index: int) -> GalaxyPlotter:
        """
        Returns an `GalaxyPlotter` corresponding to a `Galaxy` in the `Tracer`.

        Returns
        -------
        galaxy_index
            The index of the galaxy in the `Tracer` used to make the `GalaxyPlotter`.
        """

        return GalaxyPlotter(
            galaxy=self.galaxies[galaxy_index],
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d_of_galaxy(galaxy_index=galaxy_index),
        )

    def figures_2d(
        self,
        image: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        plane_image: bool = False,
        plane_grid: bool = False,
        zoom_to_brightest: bool = True,
        title_suffix: str = "",
        filename_suffix: str = "",
        source_plane_title: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `Galaxies` object in 2D, which are computed via the plotter's 2D
        grid object.

        The API is such that every plottable attribute of the `Galaxies` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 2D plot (via `imshow`) of the image of the galaxies.
        convergence
            Whether to make a 2D plot (via `imshow`) of the convergence.
        potential
            Whether to make a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether to make a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether to make a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether to make a 2D plot (via `imshow`) of the magnification.
        plane_image
            Whether to make a 2D plot (via `imshow`) of the image of the plane in the soure-plane (e.g. its
            unlensed light).
        zoom_to_brightest
            Whether to automatically zoom the plot to the brightest regions of the galaxies being plotted as
            opposed to the full extent of the grid.
        title_suffix
            Add a suffix to the end of the matplotlib title label.
        filename_suffix
            Add a suffix to the end of the filename the plot is saved to hard-disk using.
        """
        if image:
            self.mat_plot_2d.plot_array(
                array=self.galaxies.image_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Image{title_suffix}", filename=f"image_2d{filename_suffix}"
                ),
            )

        if plane_image:
            if source_plane_title:
                title = "Source Plane Image"
            else:
                title = f"Plane Image{title_suffix}"

            self.mat_plot_2d.plot_array(
                array=self.galaxies.plane_image_2d_from(
                    grid=self.grid, zoom_to_brightest=zoom_to_brightest
                ),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=title,
                    filename=f"plane_image{filename_suffix}",
                ),
            )

        if plane_grid:
            if source_plane_title:
                title = "Source Plane Grid"
            else:
                title = f"Plane Grid{title_suffix}"

            self.mat_plot_2d.plot_grid(
                grid=self.grid,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=title,
                    filename=f"plane_grid{filename_suffix}",
                ),
            )

        self._mass_plotter.figures_2d(
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
        )

    def galaxy_indexes_from(self, galaxy_index: Optional[int]) -> List[int]:
        """
        Returns a list of all indexes of the galaxys in the fit, which is iterated over in figures that plot
        individual figures of each galaxy.

        Parameters
        ----------
        galaxy_index
            A specific galaxy index which when input means that only a single galaxy index is returned.

        Returns
        -------
        list
            A list of galaxy indexes corresponding to galaxys in the galaxy.
        """
        if galaxy_index is None:
            return list(range(len(self.galaxies)))
        return [galaxy_index]

    def figures_2d_of_galaxies(
        self, image: bool = False, galaxy_index: Optional[int] = None
    ):
        """
        Plots galaxy images for each individual `Galaxy` in the plotter's `Galaxies` in 2D,  which are computed via the
        plotter's 2D grid object.

        The API is such that every plottable attribute of the `galaxy` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 2D plot (via `imshow`) of the image of the galaxy in the soure-galaxy (e.g. its
            unlensed light).
        galaxy_index
            If input, plots for only a single galaxy based on its index are created.
        """
        galaxy_indexes = self.galaxy_indexes_from(galaxy_index=galaxy_index)

        for galaxy_index in galaxy_indexes:
            galaxy_plotter = self.galaxy_plotter_from(galaxy_index=galaxy_index)

            if image:
                galaxy_plotter.figures_2d(
                    image=True,
                    title_suffix=f" Of Galaxy {galaxy_index}",
                    filename_suffix=f"_of_galaxy_{galaxy_index}",
                )

    def subplot(
        self,
        image: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        auto_filename: str = "subplot_galaxies",
    ):
        """
        Plots the individual attributes of the plotter's `Galaxies` object in 2D on a subplot, which are computed via the
        plotter's 2D grid object.

        The API is such that every plottable attribute of the `Galaxies` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        image
            Whether or not to include a 2D plot (via `imshow`) of the image.
        convergence
            Whether or not to include a 2D plot (via `imshow`) of the convergence.
        potential
            Whether or not to include a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether or not to include a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether or not to include a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether or not to include a 2D plot (via `imshow`) of the magnification.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """
        self._subplot_custom_plot(
            image=image,
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
            auto_labels=aplt.AutoLabels(filename=auto_filename),
        )

    def subplot_galaxies(self):
        """
        Standard subplot of the attributes of the plotter's `Galaxies` object.
        """
        return self.subplot(
            image=True,
            convergence=True,
            potential=True,
            deflections_y=True,
            deflections_x=True,
        )

    def subplot_galaxy_images(self):
        """
        Subplot of the image of every galaxy.

        For example, for a 2 galaxy `Galaxies`, this creates a subplot with 2 panels, one for each galaxy.
        """
        number_subplots = len(self.galaxies)

        self.open_subplot_figure(number_subplots=number_subplots)

        for galaxy_index in range(0, len(self.galaxies)):
            galaxy_plotter = self.galaxy_plotter_from(galaxy_index=galaxy_index)
            galaxy_plotter.figures_2d(
                image=True, title_suffix=f" Of Galaxies {galaxy_index}"
            )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"subplot_galaxy_images"
        )
        self.close_subplot_figure()

    def subplot_galaxies_1d(self):
        """
        Output a subplot of attributes of every individual 1D attribute of the `Galaxy` object.

        For example, a 1D plot showing how the image, convergence of each component varies radially outwards.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from the light profile. This is performed by aligning a 1D grid with the  major-axis of the light
        profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.
        """
        number_subplots = len(self.galaxies) * 3

        self.open_subplot_figure(number_subplots=number_subplots)

        for galaxy_index in range(0, len(self.galaxies)):
            galaxy_plotter = self.galaxy_plotter_from(galaxy_index=galaxy_index)

            galaxy_plotter.figures_1d(image=True)
            galaxy_plotter.figures_1d(convergence=True)
            galaxy_plotter.figures_1d(potential=True)

        self.mat_plot_1d.output.subplot_to_figure(auto_filename="subplot_galaxies_1d")
        self.close_subplot_figure()

    def subplot_galaxies_1d_decomposed(self):
        """
        Output a subplot of attributes of every individual 1D attribute of the `Galaxy` object decompoed into
        their different light and mass profiles.

        For example, a 1D plot showing how the image, convergence of each component varies radially outwards.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from the light profile. This is performed by aligning a 1D grid with the  major-axis of the light
        profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.
        """
        number_subplots = len(self.galaxies) * 3

        self.open_subplot_figure(number_subplots=number_subplots)

        for galaxy_index in range(0, len(self.galaxies)):
            galaxy_plotter = self.galaxy_plotter_from(galaxy_index=galaxy_index)

            galaxy_plotter.figures_1d_decomposed(image=True)
            galaxy_plotter.figures_1d_decomposed(convergence=True)
            galaxy_plotter.figures_1d_decomposed(potential=True)

        self.mat_plot_1d.output.subplot_to_figure(
            auto_filename="subplot_galaxies_1d_decomposed"
        )
        self.close_subplot_figure()

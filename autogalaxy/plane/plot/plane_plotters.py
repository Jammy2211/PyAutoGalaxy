from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals1D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include1D
from autogalaxy.plot.mat_wrap.include import Include2D
from autogalaxy.plot.mass_plotter import MassPlotter
from autogalaxy.galaxy.plot.galaxy_plotters import GalaxyPlotter

from autogalaxy.plane.plane import Plane


class PlanePlotter(Plotter):
    def __init__(
        self,
        plane: Plane,
        grid: aa.type.Grid1D2DLike,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        """
        Plots the attributes of `Plane` objects using the matplotlib methods `plot()` and `imshow()` and many 
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default, 
        the settings passed to every matplotlib function called are those specified in 
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to 
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be 
        extracted from the `MassProfile` and plotted via the visuals object, if the corresponding entry is `True` in 
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        plane
            The plane the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the plane's light and mass quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 2D plots.
        """
        super().__init__(
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
        )

        self.plane = plane
        self.grid = grid

        self._mass_plotter = MassPlotter(
            mass_obj=self.plane,
            grid=self.grid,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_light_mass_obj_from(
            light_mass_obj=self.plane, grid=self.grid
        )

    def get_visuals_2d_of_galaxy(self, galaxy_index: int) -> aplt.Visuals2D:
        return self.get_2d.via_plane_from(
            plane=self.plane, grid=self.grid, galaxy_index=galaxy_index
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
            galaxy=self.plane.galaxies[galaxy_index],
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d_of_galaxy(galaxy_index=galaxy_index),
            include_2d=self.include_2d,
        )

    def figures_2d(
        self,
        image: bool = False,
        plane_image: bool = False,
        plane_grid: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        contribution_map: bool = False,
        title_suffix: str = "",
        filename_suffix: str = "",
    ):
        """
        Plots the individual attributes of the plotter's `Plane` object in 2D, which are computed via the plotter's 2D
        grid object.

        The API is such that every plottable attribute of the `Plane` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether or not to make a 2D plot (via `imshow`) of the image of plane in its image-plane (e.g. after
            lensing).
        plane_image
            Whether or not to make a 2D plot (via `imshow`) of the image of the plane in the soure-plane (e.g. its
            unlensed light).
        convergence
            Whether or not to make a 2D plot (via `imshow`) of the convergence.
        potential
            Whether or not to make a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether or not to make a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether or not to make a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether or not to make a 2D plot (via `imshow`) of the magnification.
        contribution_map
            Whether or not to make a 2D plot (via `imshow`) of the contribution map.
        title_suffix
            Add a suffix to the end of the matplotlib title label.
        filename_suffix
            Add a suffix to the end of the filename the plot is saved to hard-disk using.
        """
        if image:

            self.mat_plot_2d.plot_array(
                array=self.plane.image_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Image{title_suffix}", filename=f"image_2d{filename_suffix}"
                ),
            )

        if plane_image:

            self.mat_plot_2d.plot_array(
                array=self.plane.plane_image_2d_from(grid=self.grid).array,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Plane Image{title_suffix}",
                    filename=f"plane_image{filename_suffix}",
                ),
            )

        if plane_grid:

            self.mat_plot_2d.plot_grid(
                grid=self.grid,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Plane Grid2D{title_suffix}",
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

        if contribution_map:

            self.mat_plot_2d.plot_array(
                array=self.plane.contribution_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title="Contribution Map", filename="contribution_map_2d"
                ),
            )

    def galaxy_indexes_from(self, galaxy_index: Optional[int]) -> List[int]:
        """
        Returns a list of all indexes of the galaxys in the fit, which is iterated over in figures that plot
        individual figures of each galaxy in a plane.

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
            return list(range(len(self.plane.galaxies)))
        return [galaxy_index]

    def figures_2d_of_galaxies(
        self, image: bool = False, galaxy_index: Optional[int] = None
    ):
        """
        Plots galaxy images for each individual `Galaxy` in the plotter's `Plane` in 2D,  which are computed via the
        plotter's 2D grid object.

        The API is such that every plottable attribute of the `galaxy` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether or not to make a 2D plot (via `imshow`) of the image of the galaxy in the soure-galaxy (e.g. its
            unlensed light).
        galaxy_index
            If input, plots for only a single galaxy based on its index in the plane are created.
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
        plane_image: bool = False,
        plane_grid: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        contribution_map: bool = False,
        auto_filename: str = "subplot_plane",
    ):
        """
        Plots the individual attributes of the plotter's `Plane` object in 2D on a subplot, which are computed via the 
        plotter's 2D grid object.

        The API is such that every plottable attribute of the `Plane` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        image
            Whether or not to  include a 2D plot (via `imshow`) of the image of plane in its image-plane (e.g. after
            lensing).
        plane_image
            Whether or not to  include a 2D plot (via `imshow`) of the image of the plane in the soure-plane (e.g. its
            unlensed light).
        convergence
            Whether or not to  include a 2D plot (via `imshow`) of the convergence.
        potential
            Whether or not to  include a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether or not to  include a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether or not to  include a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether or not to  include a 2D plot (via `imshow`) of the magnification.
        contribution_map
            Whether or not to  include a 2D plot (via `imshow`) of the contribution map.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """
        self._subplot_custom_plot(
            image=image,
            plane_image=plane_image,
            plane_grid=plane_grid,
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
            contribution_map=contribution_map,
            auto_labels=aplt.AutoLabels(filename=auto_filename),
        )

    def subplot_plane(self):
        """
        Standard subplot of the attributes of the plotter's `Plane` object.
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
        Subplot of the image of every galaxy in the plane.

        For example, for a 2 galaxy `Plane`, this creates a subplot with 2 panels, one for each galaxy.
        """
        number_subplots = len(self.plane.galaxies)

        self.open_subplot_figure(number_subplots=number_subplots)

        for galaxy_index in range(0, len(self.plane.galaxies)):

            galaxy_plotter = self.galaxy_plotter_from(galaxy_index=galaxy_index)
            galaxy_plotter.figures_2d(
                image=True, title_suffix=f" Of Plane {galaxy_index}"
            )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"subplot_galaxy_images"
        )
        self.close_subplot_figure()

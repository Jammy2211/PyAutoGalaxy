from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
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
        mat_plot_2d: MatPlot2D = None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        multiple_images=None,
        tangential_critical_curves=None,
        radial_critical_curves=None,
    ):
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
            mat_plot_1d=mat_plot_1d,
        )

        self.grid = grid
        self.positions = positions
        self.light_profile_centres = light_profile_centres
        self.mass_profile_centres = mass_profile_centres
        self.multiple_images = multiple_images

        self._mass_plotter = MassPlotter(
            mass_obj=self.galaxies,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            positions=positions,
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            multiple_images=multiple_images,
            tangential_critical_curves=tangential_critical_curves,
            radial_critical_curves=radial_critical_curves,
        )

    def galaxy_plotter_from(self, galaxy_index: int) -> GalaxyPlotter:
        visuals_with_cc = self._mass_plotter.visuals_2d_with_critical_curves
        tc = visuals_with_cc.tangential_critical_curves
        rc = visuals_with_cc.radial_critical_curves

        return GalaxyPlotter(
            galaxy=self.galaxies[galaxy_index],
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            tangential_critical_curves=tc,
            radial_critical_curves=rc,
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
        if image:
            from autogalaxy.plot.visuals.two_d import Visuals2D

            self._plot_array(
                array=self.galaxies.image_2d_from(grid=self.grid),
                visuals_2d=Visuals2D(
                    positions=self.positions,
                    light_profile_centres=self.light_profile_centres,
                    mass_profile_centres=self.mass_profile_centres,
                ),
                auto_labels=aplt.AutoLabels(
                    title=f"Image{title_suffix}", filename=f"image_2d{filename_suffix}"
                ),
            )

        if plane_image:
            if source_plane_title:
                title = "Source Plane Image"
            else:
                title = f"Plane Image{title_suffix}"

            from autogalaxy.plot.visuals.two_d import Visuals2D

            self._plot_array(
                array=self.galaxies.plane_image_2d_from(
                    grid=self.grid, zoom_to_brightest=zoom_to_brightest
                ),
                visuals_2d=Visuals2D(positions=self.positions),
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

            from autogalaxy.plot.visuals.two_d import Visuals2D

            self._plot_grid(
                grid=self.grid,
                visuals_2d=Visuals2D(positions=self.positions),
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
        if galaxy_index is None:
            return list(range(len(self.galaxies)))
        return [galaxy_index]

    def figures_2d_of_galaxies(
        self, image: bool = False, galaxy_index: Optional[int] = None
    ):
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
        return self.subplot(
            image=True,
            convergence=True,
            potential=True,
            deflections_y=True,
            deflections_x=True,
        )

    def subplot_galaxy_images(self):
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

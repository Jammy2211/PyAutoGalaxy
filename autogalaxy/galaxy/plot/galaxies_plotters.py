import matplotlib.pyplot as plt
from typing import List, Optional

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap

import autoarray as aa

from autogalaxy.plot.abstract_plotters import Plotter, _to_positions, _save_subplot
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
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        multiple_images=None,
        tangential_critical_curves=None,
        radial_critical_curves=None,
    ):
        self.galaxies = Galaxies(galaxies=galaxies)

        from autogalaxy.profiles.light.linear import LightProfileLinear

        if self.galaxies.has(cls=LightProfileLinear):
            raise exc.raise_linear_light_profile_in_plot(
                plotter_type=self.__class__.__name__,
            )

        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.grid = grid
        self.positions = positions
        self.light_profile_centres = light_profile_centres
        self.mass_profile_centres = mass_profile_centres
        self.multiple_images = multiple_images

        self._mass_plotter = MassPlotter(
            mass_obj=self.galaxies,
            grid=self.grid,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            positions=positions,
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            multiple_images=multiple_images,
            tangential_critical_curves=tangential_critical_curves,
            radial_critical_curves=radial_critical_curves,
        )

    def galaxy_plotter_from(self, galaxy_index: int) -> GalaxyPlotter:
        tc = self._mass_plotter.tangential_critical_curves
        rc = self._mass_plotter.radial_critical_curves

        return GalaxyPlotter(
            galaxy=self.galaxies[galaxy_index],
            grid=self.grid,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
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
        ax=None,
    ):
        if image:
            positions = _to_positions(
                self.positions,
                self.light_profile_centres,
                self.mass_profile_centres,
            )
            self._plot_array(
                array=self.galaxies.image_2d_from(grid=self.grid),
                auto_filename=f"image_2d{filename_suffix}",
                title=f"Image{title_suffix}",
                positions=positions,
                ax=ax,
            )

        if plane_image:
            title = "Source Plane Image" if source_plane_title else f"Plane Image{title_suffix}"
            self._plot_array(
                array=self.galaxies.plane_image_2d_from(
                    grid=self.grid, zoom_to_brightest=zoom_to_brightest
                ),
                auto_filename=f"plane_image{filename_suffix}",
                title=title,
                positions=_to_positions(self.positions),
                ax=ax,
            )

        if plane_grid:
            title = "Source Plane Grid" if source_plane_title else f"Plane Grid{title_suffix}"
            self._plot_grid(
                grid=self.grid,
                auto_filename=f"plane_grid{filename_suffix}",
                title=title,
                ax=ax,
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
        panels = [
            ("image", image),
            ("convergence", convergence),
            ("potential", potential),
            ("deflections_y", deflections_y),
            ("deflections_x", deflections_x),
            ("magnification", magnification),
        ]
        active = [(n, f) for n, f in panels if f]
        if not active:
            return

        n = len(active)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
        import numpy as np
        axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

        for i, (name, _) in enumerate(active):
            self.figures_2d(**{name: True}, ax=axes_flat[i])

        plt.tight_layout()
        _save_subplot(fig, self.output, auto_filename)

    def subplot_galaxies(self):
        return self.subplot(
            image=True,
            convergence=True,
            potential=True,
            deflections_y=True,
            deflections_x=True,
        )

    def subplot_galaxy_images(self):
        n = len(self.galaxies)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
        axes_flat = [axes] if n == 1 else list(axes.flatten())

        for i in range(n):
            galaxy_plotter = self.galaxy_plotter_from(galaxy_index=i)
            galaxy_plotter.figures_2d(
                image=True,
                title_suffix=f" Of Galaxies {i}",
                ax=axes_flat[i],
            )

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_galaxy_images")

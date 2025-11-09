from __future__ import annotations
import math
from typing import TYPE_CHECKING, List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.visuals.two_d import Visuals2D
from autogalaxy.plot.mass_plotter import MassPlotter

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.galaxy.galaxy import Galaxy

if TYPE_CHECKING:
    from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePlotter
from autogalaxy.profiles.plot.mass_profile_plotters import MassProfilePlotter

from autogalaxy import exc


class GalaxyPlotter(Plotter):
    def __init__(
        self,
        galaxy: Galaxy,
        grid: aa.type.Grid1D2DLike,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        """
        Plots the attributes of `Galaxy` objects using the matplotlib methods `plot()` and `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `MassProfile` and plotted via the visuals object.

        Parameters
        ----------
        galaxy
            The galaxy the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the galaxy's light and mass quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        """

        from autogalaxy.profiles.light.linear import (
            LightProfileLinear,
        )

        if galaxy is not None:
            if galaxy.has(cls=LightProfileLinear):
                raise exc.raise_linear_light_profile_in_plot(
                    plotter_type=self.__class__.__name__,
                )

        super().__init__(
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
        )

        self.galaxy = galaxy
        self.grid = grid

        self._mass_plotter = MassPlotter(
            mass_obj=self.galaxy,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
        )

    def light_profile_plotter_from(
        self, light_profile: LightProfile, one_d_only: bool = False
    ) -> LightProfilePlotter:
        """
        Returns a `LightProfilePlotter` given an input light profile, which is typically used for plotting the
        individual light profiles of the plotter's `Galaxy` (e.g. in the function `figures_1d_decomposed`).

        Parameters
        ----------
        light_profile
            The light profile which is used to create the `LightProfilePlotter`.

        Returns
        -------
        LightProfilePlotter
            An object that plots the light profiles, often used for plotting attributes of the galaxy.
        """

        from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePlotter

        if not one_d_only:

            return LightProfilePlotter(
                light_profile=light_profile,
                grid=self.grid,
                mat_plot_2d=self.mat_plot_2d,
                visuals_2d=self.visuals_2d,
                mat_plot_1d=self.mat_plot_1d,
                visuals_1d=self.visuals_1d
                + Visuals1D().add_half_light_radius(light_obj=light_profile),
            )

        return LightProfilePlotter(
            light_profile=light_profile,
            grid=self.grid,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_1d
            + Visuals1D().add_half_light_radius(light_obj=light_profile),
        )

    def mass_profile_plotter_from(
        self, mass_profile: MassProfile, one_d_only: bool = False
    ) -> MassProfilePlotter:
        """
        Returns a `MassProfilePlotter` given an input mass profile, which is typically used for plotting the individual
        mass profiles of the plotter's `Galaxy` (e.g. in the function `figures_1d_decomposed`).

        Parameters
        ----------
        mass_profile
            The mass profile which is used to create the `MassProfilePlotter`.

        Returns
        -------
        MassProfilePlotter
            An object that plots the mass profiles, often used for plotting attributes of the galaxy.
        """

        if not one_d_only:
            return MassProfilePlotter(
                mass_profile=mass_profile,
                grid=self.grid,
                mat_plot_2d=self.mat_plot_2d,
                visuals_2d=self._mass_plotter.visuals_2d_with_critical_curves,
                mat_plot_1d=self.mat_plot_1d,
                visuals_1d=self.visuals_1d
                + Visuals1D().add_einstein_radius(
                    mass_obj=mass_profile, grid=self.grid
                ),
            )

        return MassProfilePlotter(
            mass_profile=mass_profile,
            grid=self.grid,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_1d
            + Visuals1D().add_einstein_radius(mass_obj=mass_profile, grid=self.grid),
        )

    def figures_2d(
        self,
        image: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        title_suffix: str = "",
        filename_suffix: str = "",
    ):
        """
        Plots the individual attributes of the plotter's `Galaxy` object in 2D, which are computed via the plotter's 2D
        grid object.

        The API is such that every plottable attribute of the `Galaxy` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 2D plot (via `imshow`) of the image.
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
        """
        if image:
            self.mat_plot_2d.plot_array(
                array=self.galaxy.image_2d_from(grid=self.grid),
                visuals_2d=self.visuals_2d,
                auto_labels=aplt.AutoLabels(
                    title=f"Image{title_suffix}", filename=f"image_2d{filename_suffix}"
                ),
            )

        self._mass_plotter.figures_2d(
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
            title_suffix=title_suffix,
            filename_suffix=filename_suffix,
        )

    def subplot_of_light_profiles(self, image: bool = False):
        """
        Output a subplot of attributes of every individual light profile in 1D of the `Galaxy` object.

        For example, a 1D plot showing how the image (e.g. luminosity) of each component varies radially outwards.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from the light profile. This is performed by aligning a 1D grid with the  major-axis of the light
        profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        Parameters
        ----------
        image
            Whether to make a 1D subplot (via `plot`) of the image.
        """
        light_profile_plotters = [
            self.light_profile_plotter_from(light_profile)
            for light_profile in self.galaxy.cls_list_from(cls=LightProfile)
        ]

        if image:
            self.subplot_of_plotters_figure(
                plotter_list=light_profile_plotters, name="image"
            )

    def subplot_of_mass_profiles(
        self,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
    ):
        """
        Output a subplot of attributes of every individual mass profile in 1D of the `Galaxy` object.

        For example, a 1D plot showing how the convergence of each component varies radially outwards.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from the light profile. This is performed by aligning a 1D grid with the  major-axis of the light
        profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        Parameters
        ----------
        image
            Whether to make a 1D subplot (via `plot`) of the image.
        convergence
            Whether to make a 1D plot (via `plot`) of the convergence.
        potential
            Whether to make a 1D plot (via `plot`) of the potential.
        """
        mass_profile_plotters = [
            self.mass_profile_plotter_from(mass_profile)
            for mass_profile in self.galaxy.cls_list_from(cls=MassProfile)
        ]

        if convergence:
            self.subplot_of_plotters_figure(
                plotter_list=mass_profile_plotters, name="convergence"
            )

        if potential:
            self.subplot_of_plotters_figure(
                plotter_list=mass_profile_plotters, name="potential"
            )

        if deflections_y:
            self.subplot_of_plotters_figure(
                plotter_list=mass_profile_plotters, name="deflections_y"
            )

        if deflections_x:
            self.subplot_of_plotters_figure(
                plotter_list=mass_profile_plotters, name="deflections_x"
            )

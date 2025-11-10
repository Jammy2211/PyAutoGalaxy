import math
from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.mass_plotter import MassPlotter
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.visuals.two_d import Visuals2D

from autogalaxy.util import error_util


class MassProfilePlotter(Plotter):
    def __init__(
        self,
        mass_profile: MassProfile,
        grid: aa.type.Grid2DLike,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        """
        Plots the attributes of `MassProfile` objects using the matplotlib methods `plot()` and `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `MassProfile` and plotted via the visuals object.

        Parameters
        ----------
        mass_profile
            The mass profile the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the mass profile quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        """
        super().__init__(
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
        )

        self.mass_profile = mass_profile
        self.grid = grid

        self._mass_plotter = MassPlotter(
            mass_obj=self.mass_profile,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
        )

        self.figures_2d = self._mass_plotter.figures_2d

    @property
    def grid_2d_projected(self):
        return self.grid.grid_2d_radial_projected_from(
            centre=self.mass_profile.centre, angle=self.mass_profile.angle()
        )

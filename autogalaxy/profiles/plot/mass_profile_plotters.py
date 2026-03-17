from typing import Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.mass_plotter import MassPlotter
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D

from autogalaxy.util import error_util


class MassProfilePlotter(Plotter):
    def __init__(
        self,
        mass_profile: MassProfile,
        grid: aa.type.Grid2DLike,
        mat_plot_1d: MatPlot1D = None,
        mat_plot_2d: MatPlot2D = None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        tangential_critical_curves=None,
        radial_critical_curves=None,
        einstein_radius: Optional[float] = None,
        einstein_radius_errors=None,
    ):
        super().__init__(
            mat_plot_2d=mat_plot_2d,
            mat_plot_1d=mat_plot_1d,
        )

        self.mass_profile = mass_profile
        self.grid = grid
        self.einstein_radius = einstein_radius
        self.einstein_radius_errors = einstein_radius_errors

        self._mass_plotter = MassPlotter(
            mass_obj=self.mass_profile,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            positions=positions,
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            tangential_critical_curves=tangential_critical_curves,
            radial_critical_curves=radial_critical_curves,
        )

        self.figures_2d = self._mass_plotter.figures_2d

    @property
    def grid_2d_projected(self):
        return self.grid.grid_2d_radial_projected_from(
            centre=self.mass_profile.centre, angle=self.mass_profile.angle()
        )

from typing import Optional

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap

import autoarray as aa

from autogalaxy.plot.mass_plotter import MassPlotter
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy.util import error_util


class MassProfilePlotter(Plotter):
    def __init__(
        self,
        mass_profile: MassProfile,
        grid: aa.type.Grid2DLike,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        tangential_critical_curves=None,
        radial_critical_curves=None,
        einstein_radius: Optional[float] = None,
        einstein_radius_errors=None,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.mass_profile = mass_profile
        self.grid = grid
        self.einstein_radius = einstein_radius
        self.einstein_radius_errors = einstein_radius_errors

        self._mass_plotter = MassPlotter(
            mass_obj=self.mass_profile,
            grid=self.grid,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
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

import autoarray as aa
import autoarray.plot as aplt


from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D

from autogalaxy import exc


class LightProfilePlotter(Plotter):
    def __init__(
        self,
        light_profile: LightProfile,
        grid: aa.type.Grid1D2DLike,
        mat_plot_1d: MatPlot1D = None,
        mat_plot_2d: MatPlot2D = None,
        half_light_radius=None,
        half_light_radius_errors=None,
        positions=None,
        lines=None,
    ):
        from autogalaxy.profiles.light.linear import (
            LightProfileLinear,
        )

        if isinstance(light_profile, LightProfileLinear):
            raise exc.raise_linear_light_profile_in_plot(
                plotter_type=self.__class__.__name__,
            )

        self.light_profile = light_profile
        self.grid = grid
        self.half_light_radius = half_light_radius
        self.half_light_radius_errors = half_light_radius_errors
        self.positions = positions
        self.lines = lines

        super().__init__(
            mat_plot_2d=mat_plot_2d,
            mat_plot_1d=mat_plot_1d,
        )

    @property
    def grid_2d_projected(self):
        return self.grid.grid_2d_radial_projected_from(
            centre=self.light_profile.centre, angle=self.light_profile.angle()
        )

    def _visuals_2d(self):
        from autogalaxy.plot.visuals.two_d import Visuals2D

        return Visuals2D(positions=self.positions, lines=self.lines)

    def _visuals_1d(self):
        from autogalaxy.plot.visuals.one_d import Visuals1D

        return Visuals1D(
            half_light_radius=self.half_light_radius,
            half_light_radius_errors=self.half_light_radius_errors,
        )

    def figures_2d(self, image: bool = False):
        if image:
            self._plot_array(
                array=self.light_profile.image_2d_from(grid=self.grid),
                visuals_2d=self._visuals_2d(),
                auto_labels=aplt.AutoLabels(title="Image", filename="image_2d"),
            )

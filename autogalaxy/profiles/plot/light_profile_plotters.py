from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.plot.abstract_plotters import Plotter, _to_positions
from autogalaxy import exc


class LightProfilePlotter(Plotter):
    def __init__(
        self,
        light_profile: LightProfile,
        grid: aa.type.Grid1D2DLike,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        half_light_radius=None,
        half_light_radius_errors=None,
        positions=None,
        lines=None,
    ):
        from autogalaxy.profiles.light.linear import LightProfileLinear

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

        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

    @property
    def grid_2d_projected(self):
        return self.grid.grid_2d_radial_projected_from(
            centre=self.light_profile.centre, angle=self.light_profile.angle()
        )

    def figures_2d(self, image: bool = False, ax=None):
        if image:
            self._plot_array(
                array=self.light_profile.image_2d_from(grid=self.grid),
                auto_filename="image_2d",
                title="Image",
                positions=_to_positions(self.positions),
                lines=self.lines,
                ax=ax,
            )

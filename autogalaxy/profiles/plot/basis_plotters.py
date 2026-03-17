import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.basis import Basis
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D

from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePlotter

from autogalaxy import exc


class BasisPlotter(Plotter):
    def __init__(
        self,
        basis: Basis,
        grid: aa.type.Grid1D2DLike,
        mat_plot_1d: MatPlot1D = None,
        mat_plot_2d: MatPlot2D = None,
        positions=None,
        lines=None,
    ):
        from autogalaxy.profiles.light.linear import (
            LightProfileLinear,
        )

        for light_profile in basis.light_profile_list:
            if isinstance(light_profile, LightProfileLinear):
                raise exc.raise_linear_light_profile_in_plot(
                    plotter_type=self.__class__.__name__,
                )

        self.basis = basis
        self.grid = grid
        self.positions = positions
        self.lines = lines

        super().__init__(
            mat_plot_2d=mat_plot_2d,
            mat_plot_1d=mat_plot_1d,
        )

    def light_profile_plotter_from(
        self,
        light_profile: LightProfile,
    ) -> LightProfilePlotter:
        return LightProfilePlotter(
            light_profile=light_profile,
            grid=self.grid,
            mat_plot_1d=self.mat_plot_1d,
            half_light_radius=light_profile.half_light_radius,
        )

    def subplot_image(self):
        self.open_subplot_figure(number_subplots=len(self.basis.light_profile_list))

        for light_profile in self.basis.light_profile_list:
            from autogalaxy.plot.visuals.two_d import Visuals2D

            self._plot_array(
                array=light_profile.image_2d_from(grid=self.grid),
                visuals_2d=Visuals2D(positions=self.positions, lines=self.lines),
                auto_labels=aplt.AutoLabels(title=light_profile.coefficient_tag),
            )

        self.mat_plot_2d.output.subplot_to_figure(auto_filename=f"subplot_basis_image")

        self.close_subplot_figure()

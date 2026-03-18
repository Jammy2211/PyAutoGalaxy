from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.abstract_plotters import Plotter, _to_positions
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
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
        mat_plot_2d: MatPlot2D = None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        multiple_images=None,
        tangential_critical_curves=None,
        radial_critical_curves=None,
    ):
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
            mat_plot_1d=mat_plot_1d,
        )

        self.galaxy = galaxy
        self.grid = grid
        self.positions = positions
        self.light_profile_centres = light_profile_centres
        self.mass_profile_centres = mass_profile_centres
        self.multiple_images = multiple_images

        self._mass_plotter = MassPlotter(
            mass_obj=self.galaxy,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            positions=positions,
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            multiple_images=multiple_images,
            tangential_critical_curves=tangential_critical_curves,
            radial_critical_curves=radial_critical_curves,
        )

    def light_profile_plotter_from(
        self, light_profile: LightProfile, one_d_only: bool = False
    ) -> LightProfilePlotter:
        from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePlotter

        if not one_d_only:
            return LightProfilePlotter(
                light_profile=light_profile,
                grid=self.grid,
                mat_plot_2d=self.mat_plot_2d,
                mat_plot_1d=self.mat_plot_1d,
                half_light_radius=light_profile.half_light_radius,
                positions=self.positions,
            )

        return LightProfilePlotter(
            light_profile=light_profile,
            grid=self.grid,
            mat_plot_1d=self.mat_plot_1d,
            half_light_radius=light_profile.half_light_radius,
        )

    def mass_profile_plotter_from(
        self, mass_profile: MassProfile, one_d_only: bool = False
    ) -> MassProfilePlotter:
        from autogalaxy.operate.lens_calc import LensCalc

        tc = self._mass_plotter.tangential_critical_curves
        rc = self._mass_plotter.radial_critical_curves

        einstein_radius = None
        try:
            od = LensCalc.from_mass_obj(mass_profile)
            einstein_radius = od.einstein_radius_from(grid=self.grid)
        except (TypeError, AttributeError):
            pass

        if not one_d_only:
            return MassProfilePlotter(
                mass_profile=mass_profile,
                grid=self.grid,
                mat_plot_2d=self.mat_plot_2d,
                mat_plot_1d=self.mat_plot_1d,
                tangential_critical_curves=tc,
                radial_critical_curves=rc,
                einstein_radius=einstein_radius,
            )

        return MassProfilePlotter(
            mass_profile=mass_profile,
            grid=self.grid,
            mat_plot_1d=self.mat_plot_1d,
            einstein_radius=einstein_radius,
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
        if image:
            positions = _to_positions(
                self.positions,
                self.light_profile_centres,
                self.mass_profile_centres,
            )
            self._plot_array(
                array=self.galaxy.image_2d_from(grid=self.grid),
                auto_labels=aplt.AutoLabels(
                    title=f"Image{title_suffix}", filename=f"image_2d{filename_suffix}"
                ),
                positions=positions,
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

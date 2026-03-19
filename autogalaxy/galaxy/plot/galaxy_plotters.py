from __future__ import annotations
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Optional

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap

import autoarray as aa

from autogalaxy.plot.abstract_plotters import Plotter, _to_positions, _save_subplot
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
        from autogalaxy.profiles.light.linear import LightProfileLinear

        if galaxy is not None:
            if galaxy.has(cls=LightProfileLinear):
                raise exc.raise_linear_light_profile_in_plot(
                    plotter_type=self.__class__.__name__,
                )

        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.galaxy = galaxy
        self.grid = grid
        self.positions = positions
        self.light_profile_centres = light_profile_centres
        self.mass_profile_centres = mass_profile_centres
        self.multiple_images = multiple_images

        self._mass_plotter = MassPlotter(
            mass_obj=self.galaxy,
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

    def light_profile_plotter_from(
        self, light_profile: LightProfile, one_d_only: bool = False
    ) -> LightProfilePlotter:
        from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePlotter

        return LightProfilePlotter(
            light_profile=light_profile,
            grid=self.grid,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            half_light_radius=light_profile.half_light_radius,
            positions=self.positions if not one_d_only else None,
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

        return MassProfilePlotter(
            mass_profile=mass_profile,
            grid=self.grid,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            tangential_critical_curves=tc if not one_d_only else None,
            radial_critical_curves=rc if not one_d_only else None,
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
        ax=None,
    ):
        if image:
            positions = _to_positions(
                self.positions,
                self.light_profile_centres,
                self.mass_profile_centres,
            )
            self._plot_array(
                array=self.galaxy.image_2d_from(grid=self.grid),
                auto_filename=f"image_2d{filename_suffix}",
                title=f"Image{title_suffix}",
                positions=positions,
                ax=ax,
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
        light_profiles = self.galaxy.cls_list_from(cls=LightProfile)
        if not light_profiles or not image:
            return

        n = len(light_profiles)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
        axes_flat = [axes] if n == 1 else list(axes.flatten())

        for i, lp in enumerate(light_profiles):
            plotter = self.light_profile_plotter_from(lp)
            plotter.figures_2d(image=True, ax=axes_flat[i])

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_image")

    def subplot_of_mass_profiles(
        self,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
    ):
        mass_profiles = self.galaxy.cls_list_from(cls=MassProfile)
        if not mass_profiles:
            return

        n = len(mass_profiles)

        for name, flag in [
            ("convergence", convergence),
            ("potential", potential),
            ("deflections_y", deflections_y),
            ("deflections_x", deflections_x),
        ]:
            if not flag:
                continue

            fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
            axes_flat = [axes] if n == 1 else list(axes.flatten())

            for i, mp in enumerate(mass_profiles):
                plotter = self.mass_profile_plotter_from(mp)
                plotter.figures_2d(**{name: True})

            plt.tight_layout()
            _save_subplot(fig, self.output, f"subplot_{name}")

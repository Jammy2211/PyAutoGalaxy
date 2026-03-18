import numpy as np

from autoconf import cached_property

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.mat_plot.two_d import MatPlot2D

from autogalaxy.plot.abstract_plotters import Plotter, _to_lines, _to_positions


class MassPlotter(Plotter):
    def __init__(
        self,
        mass_obj,
        grid: aa.type.Grid2DLike,
        mat_plot_2d: MatPlot2D = None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        multiple_images=None,
        tangential_critical_curves=None,
        radial_critical_curves=None,
        tangential_caustics=None,
        radial_caustics=None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d)

        self.mass_obj = mass_obj
        self.grid = grid
        self.positions = positions
        self.light_profile_centres = light_profile_centres
        self.mass_profile_centres = mass_profile_centres
        self.multiple_images = multiple_images
        self._tc = tangential_critical_curves
        self._rc = radial_critical_curves
        self._tc_caustic = tangential_caustics
        self._rc_caustic = radial_caustics

    @cached_property
    def _critical_curves(self):
        from autogalaxy.operate.lens_calc import LensCalc

        tc = self._tc
        rc = self._rc

        if tc is None:
            od = LensCalc.from_mass_obj(self.mass_obj)
            tc = od.tangential_critical_curve_list_from(grid=self.grid)
            rc_area = od.radial_critical_curve_area_list_from(grid=self.grid)
            if any(area > self.grid.pixel_scale for area in rc_area):
                rc = od.radial_critical_curve_list_from(grid=self.grid)

        return tc, rc

    @property
    def tangential_critical_curves(self):
        tc, rc = self._critical_curves
        return tc

    @property
    def radial_critical_curves(self):
        tc, rc = self._critical_curves
        return rc

    def _lines(self):
        tc, rc = self._critical_curves
        return _to_lines(tc, rc)

    def _positions_list(self):
        return _to_positions(
            self.positions,
            self.light_profile_centres,
            self.mass_profile_centres,
            self.multiple_images,
        )

    def figures_2d(
        self,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        title_suffix: str = "",
        filename_suffix: str = "",
    ):
        lines = self._lines()
        positions = self._positions_list()

        if convergence:
            self._plot_array(
                array=self.mass_obj.convergence_2d_from(grid=self.grid),
                auto_labels=aplt.AutoLabels(
                    title=f"Convergence{title_suffix}",
                    filename=f"convergence_2d{filename_suffix}",
                ),
                lines=lines,
                positions=positions,
            )

        if potential:
            self._plot_array(
                array=self.mass_obj.potential_2d_from(grid=self.grid),
                auto_labels=aplt.AutoLabels(
                    title=f"Potential{title_suffix}",
                    filename=f"potential_2d{filename_suffix}",
                ),
                lines=lines,
                positions=positions,
            )

        if deflections_y:
            deflections = self.mass_obj.deflections_yx_2d_from(grid=self.grid)
            deflections_y = aa.Array2D(
                values=deflections.slim[:, 0], mask=self.grid.mask
            )

            self._plot_array(
                array=deflections_y,
                auto_labels=aplt.AutoLabels(
                    title=f"Deflections Y{title_suffix}",
                    filename=f"deflections_y_2d{filename_suffix}",
                ),
                lines=lines,
                positions=positions,
            )

        if deflections_x:
            deflections = self.mass_obj.deflections_yx_2d_from(grid=self.grid)
            deflections_x = aa.Array2D(
                values=deflections.slim[:, 1], mask=self.grid.mask
            )

            self._plot_array(
                array=deflections_x,
                auto_labels=aplt.AutoLabels(
                    title=f"Deflections X{title_suffix}",
                    filename=f"deflections_x_2d{filename_suffix}",
                ),
                lines=lines,
                positions=positions,
            )

        if magnification:
            from autogalaxy.operate.lens_calc import LensCalc

            self._plot_array(
                array=LensCalc.from_mass_obj(
                    self.mass_obj
                ).magnification_2d_from(grid=self.grid),
                auto_labels=aplt.AutoLabels(
                    title=f"Magnification{title_suffix}",
                    filename=f"magnification_2d{filename_suffix}",
                ),
                lines=lines,
                positions=positions,
            )

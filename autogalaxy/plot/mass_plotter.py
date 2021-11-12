from typing import Callable

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include2D

from autogalaxy.plot.abstract_plotters import Plotter


class MassPlotter(Plotter):
    def __init__(
        self,
        mass_obj,
        grid: aa.type.Grid2DLike,
        get_visuals_2d: Callable,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):

        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.mass_obj = mass_obj
        self.grid = grid
        self.get_visuals_2d = get_visuals_2d

    def figures_2d(
        self,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
    ):

        if convergence:

            self.mat_plot_2d.plot_array(
                array=self.mass_obj.convergence_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title="Convergence", filename="convergence_2d"
                ),
            )

        if potential:

            self.mat_plot_2d.plot_array(
                array=self.mass_obj.potential_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(title="Potential", filename="potential_2d"),
            )

        if deflections_y:

            deflections = self.mass_obj.deflections_2d_from(grid=self.grid)
            deflections_y = aa.Array2D.manual_mask(
                array=deflections.slim[:, 0], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_y,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title="Deflections Y", filename="deflections_y_2d"
                ),
            )

        if deflections_x:

            deflections = self.mass_obj.deflections_2d_from(grid=self.grid)
            deflections_x = aa.Array2D.manual_mask(
                array=deflections.slim[:, 1], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_x,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title="deflections X", filename="deflections_x_2d"
                ),
            )

        if magnification:

            self.mat_plot_2d.plot_array(
                array=self.mass_obj.magnification_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title="Magnification", filename="magnification_2d"
                ),
            )

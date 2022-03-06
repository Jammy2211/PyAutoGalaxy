from typing import Callable

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.operate.deflections import OperateDeflections

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
        self._get_visuals_2d = get_visuals_2d

    def get_visuals_2d(self) -> Visuals2D:
        return self._get_visuals_2d()

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
        """
        Plots the individual attributes of the plotter's mass object in 2D, which are computed via the plotter's 2D
        grid object.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        convergence
            Whether or not to make a 2D plot (via `imshow`) of the convergence.
        potential
            Whether or not to make a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether or not to make a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether or not to make a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether or not to make a 2D plot (via `imshow`) of the magnification.
        """
        if convergence:

            self.mat_plot_2d.plot_array(
                array=self.mass_obj.convergence_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Convergence{title_suffix}",
                    filename=f"convergence_2d{filename_suffix}",
                ),
            )

        if potential:

            self.mat_plot_2d.plot_array(
                array=self.mass_obj.potential_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Potential{title_suffix}",
                    filename=f"potential_2d{filename_suffix}",
                ),
            )

        if deflections_y:

            deflections = self.mass_obj.deflections_yx_2d_from(grid=self.grid)
            deflections_y = aa.Array2D.manual_mask(
                array=deflections.slim[:, 0], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_y,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Deflections Y{title_suffix}",
                    filename=f"deflections_y_2d{filename_suffix}",
                ),
            )

        if deflections_x:

            deflections = self.mass_obj.deflections_yx_2d_from(grid=self.grid)
            deflections_x = aa.Array2D.manual_mask(
                array=deflections.slim[:, 1], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_x,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"deflections X{title_suffix}",
                    filename=f"deflections_x_2d{filename_suffix}",
                ),
            )

        if magnification:

            self.mat_plot_2d.plot_array(
                array=self.mass_obj.magnification_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Magnification{title_suffix}",
                    filename=f"magnification_2d{filename_suffix}",
                ),
            )

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include2D

from autogalaxy.plot.abstract_plotters import Plotter


class LightMassPlotter(Plotter):
    def __init__(
        self,
        light_mass_obj,
        grid: aa.Grid2D,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):

        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.light_and_mass_obj = light_mass_obj
        self.grid = grid

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
                array=self.light_and_mass_obj.convergence_2d_from(grid=self.grid),
                visuals_2d=self.get_2d.via_mass_obj_from(
                    mass_obj=self.light_and_mass_obj, grid=self.grid
                ),
                auto_labels=aplt.AutoLabels(
                    title="Convergence", filename="convergence_2d"
                ),
            )

        if potential:

            self.mat_plot_2d.plot_array(
                array=self.light_and_mass_obj.potential_2d_from(grid=self.grid),
                visuals_2d=self.get_2d.via_mass_obj_from(
                    mass_obj=self.light_and_mass_obj, grid=self.grid
                ),
                auto_labels=aplt.AutoLabels(title="Potential", filename="potential_2d"),
            )

        if deflections_y:

            deflections = self.light_and_mass_obj.deflections_2d_from(grid=self.grid)
            deflections_y = aa.Array2D.manual_mask(
                array=deflections.slim[:, 0], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_y,
                visuals_2d=self.get_2d.via_mass_obj_from(
                    mass_obj=self.light_and_mass_obj, grid=self.grid
                ),
                auto_labels=aplt.AutoLabels(
                    title="Deflections Y", filename="deflections_y_2d"
                ),
            )

        if deflections_x:

            deflections = self.light_and_mass_obj.deflections_2d_from(grid=self.grid)
            deflections_x = aa.Array2D.manual_mask(
                array=deflections.slim[:, 1], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_x,
                visuals_2d=self.get_2d.via_mass_obj_from(
                    mass_obj=self.light_and_mass_obj, grid=self.grid
                ),
                auto_labels=aplt.AutoLabels(
                    title="deflections X", filename="deflections_x_2d"
                ),
            )

        if magnification:

            self.mat_plot_2d.plot_array(
                array=self.light_and_mass_obj.magnification_2d_from(grid=self.grid),
                visuals_2d=self.get_2d.via_mass_obj_from(
                    mass_obj=self.light_and_mass_obj, grid=self.grid
                ),
                auto_labels=aplt.AutoLabels(
                    title="Magnification", filename="magnification_2d"
                ),
            )

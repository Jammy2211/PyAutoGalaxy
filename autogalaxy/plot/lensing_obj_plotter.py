import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.abstract_plotters import Plotter


class LensingObjPlotter(Plotter):

    lensing_obj = None
    grid = None

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
                array=self.lensing_obj.convergence_2d_from(grid=self.grid),
                visuals_2d=self.get_2d.via_lensing_obj_from(
                    lensing_obj=self.lensing_obj, grid=self.grid
                ),
                auto_labels=aplt.AutoLabels(
                    title="Convergence", filename="convergence_2d"
                ),
            )

        if potential:

            self.mat_plot_2d.plot_array(
                array=self.lensing_obj.potential_2d_from(grid=self.grid),
                visuals_2d=self.get_2d.via_lensing_obj_from(
                    lensing_obj=self.lensing_obj, grid=self.grid
                ),
                auto_labels=aplt.AutoLabels(title="Potential", filename="potential_2d"),
            )

        if deflections_y:

            deflections = self.lensing_obj.deflections_2d_from(grid=self.grid)
            deflections_y = aa.Array2D.manual_mask(
                array=deflections.slim[:, 0], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_y,
                visuals_2d=self.get_2d.via_lensing_obj_from(
                    lensing_obj=self.lensing_obj, grid=self.grid
                ),
                auto_labels=aplt.AutoLabels(
                    title="Deflections Y", filename="deflections_y_2d"
                ),
            )

        if deflections_x:

            deflections = self.lensing_obj.deflections_2d_from(grid=self.grid)
            deflections_x = aa.Array2D.manual_mask(
                array=deflections.slim[:, 1], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_x,
                visuals_2d=self.get_2d.via_lensing_obj_from(
                    lensing_obj=self.lensing_obj, grid=self.grid
                ),
                auto_labels=aplt.AutoLabels(
                    title="deflections X", filename="deflections_x_2d"
                ),
            )

        if magnification:

            self.mat_plot_2d.plot_array(
                array=self.lensing_obj.magnification_2d_from(grid=self.grid),
                visuals_2d=self.get_2d.via_lensing_obj_from(
                    lensing_obj=self.lensing_obj, grid=self.grid
                ),
                auto_labels=aplt.AutoLabels(
                    title="Magnification", filename="magnification_2d"
                ),
            )

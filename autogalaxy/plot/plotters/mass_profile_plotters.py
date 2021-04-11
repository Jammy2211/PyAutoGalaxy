from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids.one_d import grid_1d
from autogalaxy.plot.plotters import lensing_obj_plotter
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autoarray.plot.mat_wrap import mat_plot
from autogalaxy.profiles.mass_profiles import mass_profiles as mp


class MassProfilePlotter(lensing_obj_plotter.LensingObjPlotter):
    def __init__(
        self,
        mass_profile: mp.MassProfile,
        grid: grid_2d.Grid2D,
        mat_plot_1d: lensing_mat_plot.MatPlot1D = lensing_mat_plot.MatPlot1D(),
        visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
        include_1d: lensing_include.Include1D = lensing_include.Include1D(),
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):
        super().__init__(
            lensing_obj=mass_profile,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
        )

    @property
    def mass_profile(self):
        return self.lensing_obj

    @property
    def grid_2d_radial_projected(self):
        return self.grid.grid_2d_radial_projected_from(
            centre=self.lensing_obj.centre, angle=self.lensing_obj.angle + 90.0
        )

    @property
    def grid_1d_radial_distances(self):

        radial_distances = self.grid_2d_radial_projected.distances_from_coordinate(
            coordinate=self.lensing_obj.centre
        )

        return grid_1d.Grid1D.manual_native(
            grid=radial_distances,
            pixel_scales=abs(radial_distances[0] - radial_distances[1]),
        )

    def figures_1d(self, convergence=False, potential=False):

        if convergence:

            self.mat_plot_1d.plot_yx(
                y=self.lensing_obj.convergence_from_grid(
                    grid=self.grid_2d_radial_projected
                ),
                x=self.grid_1d_radial_distances,
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=mat_plot.AutoLabels(
                    title="Convergence vs Radius",
                    ylabel="Convergence ",
                    xlabel="Radius",
                    legend=self.lensing_obj.__class__.__name__,
                    filename="convergence_1d",
                ),
            )

        if potential:

            self.mat_plot_1d.plot_yx(
                y=self.lensing_obj.potential_from_grid(
                    grid=self.grid_2d_radial_projected
                ),
                x=self.grid_1d_radial_distances,
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=mat_plot.AutoLabels(
                    title="Potential vs Radius",
                    ylabel="Potential ",
                    xlabel="Radius",
                    legend=self.lensing_obj.__class__.__name__,
                    filename="potential_1d",
                ),
            )

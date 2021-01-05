from autoarray.structures import grids
from autoarray.plot.plotters import abstract_plotters
from autogalaxy.plot.plotters import lensing_obj_plotter
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.plane import plane as pl
from autoarray.plot.plotters import structure_plotters


class PlanePlotter(lensing_obj_plotter.LensingObjPlotter):
    def __init__(
        self,
        plane: pl.Plane,
        grid: grids.Grid,
        mat_plot_1d: lensing_mat_plot.MatPlot1D = lensing_mat_plot.MatPlot1D(),
        visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
        include_1d: lensing_include.Include1D = lensing_include.Include1D(),
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):
        super().__init__(
            lensing_obj=plane,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
        )

    @property
    def visuals_with_include_2d(self) -> "vis.Visuals2D":
        """
        Extracts from a `Structure` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From an `AbstractStructure` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        structure : abstract_structure.AbstractStructure
            The structure whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        visuals_2d = super(PlanePlotter, self).visuals_with_include_2d

        visuals_2d.mask = None

        return visuals_2d + lensing_visuals.Visuals2D(
            light_profile_centres=self.extract_2d(
                "light_profile_centres", self.lensing_obj.light_profile_centres
            )
        )

    @property
    def plane(self):
        return self.lensing_obj

    @abstract_plotters.for_figure
    def figure_image(self):
        self.mat_plot_2d.plot_array(
            array=self.plane.image_from_grid(grid=self.grid),
            visuals_2d=self.visuals_with_include_2d,
        )

    @abstract_plotters.for_figure
    def figure_plane_image(self):

        self.mat_plot_2d.plot_array(
            array=self.plane.plane_image_from_grid(grid=self.grid).array,
            visuals_2d=self.visuals_with_include_2d,
        )

    @abstract_plotters.for_figure
    def figure_contribution_map(self):

        self.mat_plot_2d.plot_array(
            array=self.plane.contribution_map, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_figure
    def subplot_image_and_source_plane(
        self, image_plane, source_plane, grid, indexes=None, axis_limits=None
    ):

        number_subplots = 2

        mat_plot_2d = self.mat_plot_2d.mat_plot_for_subplot_from(
            func=self.subplot_image_and_source_plane
        )

        mat_plot_2d.open_subplot_figure(number_subplots=number_subplots)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.figure_plane_grid(
            plane=image_plane, grid=grid, indexes=indexes, axis_limits=axis_limits
        )

        source_plane_grid = image_plane.traced_grid_from_grid(grid=grid)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        self.figure_plane_grid(
            plane=source_plane,
            grid=source_plane_grid,
            indexes=indexes,
            axis_limits=axis_limits,
        )

        mat_plot_2d.output.subplot_to_figure()
        mat_plot_2d.figure.close()

from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.structures import grids
from autogalaxy.plot.plotters import lensing_obj_plotter
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.plane import plane as pl


class PlanePlotter(lensing_obj_plotter.LensingObjPlotter):
    def __init__(
        self,
        plane: pl.Plane,
        grid: grids.Grid2D,
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

        return visuals_2d + visuals_2d.__class__(
            grid=self.extract_2d("grid", self.grid),
            light_profile_centres=self.extract_2d(
                "light_profile_centres", self.lensing_obj.light_profile_centres
            ),
        )

    @property
    def plane(self):
        return self.lensing_obj

    def figures(
        self,
        image=False,
        plane_image=False,
        plane_grid=False,
        convergence=False,
        potential=False,
        deflections_y=False,
        deflections_x=False,
        magnification=False,
        contribution_map=False,
        title_suffix="",
        filename_suffix="",
    ):

        if image:

            self.mat_plot_2d.plot_array(
                array=self.plane.image_from_grid(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title=f"Image{title_suffix}", filename=f"image{filename_suffix}"
                ),
            )

        if plane_image:

            self.mat_plot_2d.plot_array(
                array=self.plane.plane_image_from_grid(grid=self.grid).array,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title=f"Plane Image{title_suffix}",
                    filename=f"plane_image{filename_suffix}",
                ),
            )

        if plane_grid:

            self.mat_plot_2d.plot_grid(
                grid=self.grid,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title=f"Plane Grid2D{title_suffix}",
                    filename=f"plane_grid{filename_suffix}",
                ),
            )

        super().figures(
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
        )

        if contribution_map:

            self.mat_plot_2d.plot_array(
                array=self.plane.contribution_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Contribution Map", filename="contribution_map"
                ),
            )

    def subplot(
        self,
        image=False,
        plane_image=False,
        plane_grid=False,
        convergence=False,
        potential=False,
        deflections_y=False,
        deflections_x=False,
        magnification=False,
        contribution_map=False,
        auto_filename="subplot_plane",
    ):

        self._subplot_custom_plot(
            image=image,
            plane_image=plane_image,
            plane_grid=plane_grid,
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
            contribution_map=contribution_map,
            auto_labels=mp.AutoLabels(filename=auto_filename),
        )

    def subplot_with_source_grid(self):

        self.open_subplot_figure(number_subplots=2)

        self.figures()
        self.mat_plot_2d.plot_grid(
            grid=self.plane.traced_grid_from_grid(grid=self.grid),
            visuals_2d=self.visuals_with_include_2d,
            auto_labels=mp.AutoLabels(),
        )

        self.mat_plot_2d.output.subplot_to_figure()
        self.close_subplot_figure()

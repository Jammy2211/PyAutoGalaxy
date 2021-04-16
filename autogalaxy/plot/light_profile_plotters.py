from autoarray.structures.grids.one_d import grid_1d
from autoarray.structures.grids.two_d import abstract_grid_2d, grid_2d_irregular
from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot import abstract_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.profiles import light_profiles as lp


class LightProfilePlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        light_profile: lp.LightProfile,
        grid: abstract_grid_2d.AbstractGrid2D,
        mat_plot_1d: lensing_mat_plot.MatPlot1D = lensing_mat_plot.MatPlot1D(),
        visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
        include_1d: lensing_include.Include1D = lensing_include.Include1D(),
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):

        self.light_profile = light_profile
        self.grid = grid

        super().__init__(
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
        )

    @property
    def visuals_with_include_1d(self) -> lensing_visuals.Visuals1D:
        """
        Extracts from the `LightProfile` attributes that can be plotted and return them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `LightProfilePlotter` the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the `LightProfile`'s total integrated luminosity.

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """
        return self.visuals_1d + self.visuals_1d.__class__(
            half_light_radius=self.light_profile.half_light_radius
        )

    @property
    def visuals_with_include_2d(self) -> lensing_visuals.Visuals2D:
        """
        Extracts from the `LightProfile` attributes that can be plotted and return them in a `Visuals2D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `LightProfilePlotter` the following 2D attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin",
                value=grid_2d_irregular.Grid2DIrregular(grid=[self.grid.origin]),
            ),
            mask=self.extract_2d("mask", value=self.grid.mask),
            border=self.extract_2d(
                "border", value=self.grid.mask.border_grid_sub_1.binned
            ),
            light_profile_centres=self.extract_2d(
                "light_profile_centres",
                grid_2d_irregular.Grid2DIrregular(grid=[self.light_profile.centre]),
            ),
        )

    def figures_1d(self, image=False):

        if self.mat_plot_1d.yx_plot.plot_axis_type is None:
            plot_axis_type_override = "semilogy"
        else:
            plot_axis_type_override = None

        if image:

            image_1d = self.light_profile.image_1d_from_grid(grid=self.grid)

            self.mat_plot_1d.plot_yx(
                y=image_1d,
                x=image_1d.grid_radial,
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=mat_plot.AutoLabels(
                    title="Image vs Radius",
                    ylabel="Image",
                    xlabel="Radius",
                    legend=self.light_profile.__class__.__name__,
                    filename="image_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

    def figures_2d(self, image=False):

        if image:

            self.mat_plot_2d.plot_array(
                array=self.light_profile.image_2d_from_grid(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mat_plot.AutoLabels(title="Image", filename="image_2d"),
            )

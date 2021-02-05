from autoarray.structures import grids
from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.util import plotter_util
from autoarray.plot.plotters import abstract_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals


class LightProfilePlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        light_profile,
        grid,
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
    def visuals_with_include_2d(self) -> lensing_visuals.Visuals2D:
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

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", value=grids.Grid2DIrregular(grid=[self.grid.origin])
            ),
            mask=self.extract_2d("mask", value=self.grid.mask),
            border=self.extract_2d(
                "border", value=self.grid.mask.border_grid_sub_1.slim_binned
            ),
            light_profile_centres=self.extract_2d(
                "light_profile_centres", self.light_profile.light_profile_centres
            ),
        )

    def figures(self, image=False):

        if image:

            self.mat_plot_2d.plot_array(
                array=self.light_profile.image_from_grid(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(title="Image", filename="image"),
            )

    def figures_1d(self, image=False):

        grid_radii = self.grid.grid_radii_from(centre=self.light_profile.centre)

        if image:

            self.mat_plot_1d.plot_line(
                y=self.light_profile.image_from_grid(grid=grid_radii),
                x=grid_radii[:, 1],
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title="Image vs Radius",
                    ylabel="Image",
                    xlabel="Radius",
                    legend=self.light_profile.__class__.__name__,
                ),
            )

    def luminosity_within_circle_in_electrons_per_second_as_function_of_radius(
        self,
        light_profile,
        minimum_radius=1.0e-4,
        maximum_radius=10.0,
        radii_bins=10,
        plot_axis_type="semilogy",
    ):

        radii = plotter_util.quantity_radii_from_minimum_and_maximum_radii_and_radii_points(
            minimum_radius=minimum_radius,
            maximum_radius=maximum_radius,
            radii_points=radii_bins,
        )

        luminosities = list(
            map(
                lambda radius: light_profile.luminosity_within_circle(radius=radius),
                radii,
            )
        )

        self.line(quantity=luminosities, radii=radii, plot_axis_type=plot_axis_type)

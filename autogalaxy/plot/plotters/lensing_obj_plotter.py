from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.grids.one_d import grid_1d
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import mat_plot as mp
from autogalaxy import lensing
from autogalaxy.profiles import mass_profiles
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals


class LensingObjPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        lensing_obj: lensing.LensingObject,
        grid: grid_2d.Grid2D,
        mat_plot_1d: lensing_mat_plot.MatPlot1D = lensing_mat_plot.MatPlot1D(),
        visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
        include_1d: lensing_include.Include1D = lensing_include.Include1D(),
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):
        super().__init__(
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
        )

        self.lensing_obj = lensing_obj
        self.grid = grid

    @property
    def visuals_with_include_1d(self) -> lensing_visuals.Visuals1D:
        """
        Extract from the `LensingObj` attributes that can be plotted and return them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `LensingObjProfilePlotter` the following 1D attributes can be extracted for plotting:

        - einstein_radius: the Einstein radius of the `MassProfile`.

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """
        return self.visuals_1d + self.visuals_1d.__class__(
            einstein_radius=self.lensing_obj.einstein_radius_from_grid(grid=self.grid)
        )

    @property
    def visuals_with_include_2d(self) -> lensing_visuals.Visuals2D:
        """
        Extract from the `LensingObj` attributes that can be plotted and return them in a `Visuals` object.

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
                "origin",
                value=grid_2d_irregular.Grid2DIrregular(grid=[self.grid.origin]),
            ),
            mask=self.extract_2d("mask", value=self.grid.mask),
            border=self.extract_2d(
                "border", value=self.grid.mask.border_grid_sub_1.slim_binned
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres",
                self.lensing_obj.extract_attribute(
                    cls=mass_profiles.MassProfile, name="centre"
                ),
            ),
            critical_curves=self.extract_2d(
                "critical_curves",
                self.lensing_obj.critical_curves_from_grid(grid=self.grid),
                "critical_curves",
            ),
        )

    def figures_1d(self, convergence=False, potential=False):

        grid_2d_radial_projected = self.grid.grid_2d_radial_projected_from(
            centre=self.lensing_obj.centre, angle=self.lensing_obj.phi + 90.0
        )

        radial_distances = grid_2d_radial_projected.distances_from_coordinate()

        grid_1d_radial_distances = grid_1d.Grid1D.manual_native(
            grid=radial_distances,
            pixel_scales=abs(radial_distances[0] - radial_distances[1]),
        )

        if convergence:

            self.mat_plot_1d.plot_yx(
                y=self.lensing_obj.convergence_from_grid(grid=grid_2d_radial_projected),
                x=grid_1d_radial_distances,
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=mp.AutoLabels(
                    title="Convergence vs Radius",
                    ylabel="Convergence ",
                    xlabel="Radius",
                    legend=self.lensing_obj.__class__.__name__,
                    filename="convergence_1d",
                ),
            )

        if potential:

            self.mat_plot_1d.plot_yx(
                y=self.lensing_obj.potential_from_grid(grid=grid_2d_radial_projected),
                x=grid_1d_radial_distances,
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=mp.AutoLabels(
                    title="Potential vs Radius",
                    ylabel="Potential ",
                    xlabel="Radius",
                    legend=self.lensing_obj.__class__.__name__,
                    filename="potential_1d",
                ),
            )

    def figures(
        self,
        convergence=False,
        potential=False,
        deflections_y=False,
        deflections_x=False,
        magnification=False,
    ):

        if convergence:

            self.mat_plot_2d.plot_array(
                array=self.lensing_obj.convergence_from_grid(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(title="Convergence", filename="convergence"),
            )

        if potential:

            self.mat_plot_2d.plot_array(
                array=self.lensing_obj.potential_from_grid(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(title="Potential", filename="potential"),
            )

        if deflections_y:

            deflections = self.lensing_obj.deflections_from_grid(grid=self.grid)
            deflections_y = array_2d.Array2D.manual_mask(
                array=deflections.slim[:, 0], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_y,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Deflections Y", filename="deflections_y"
                ),
            )

        if deflections_x:

            deflections = self.lensing_obj.deflections_from_grid(grid=self.grid)
            deflections_x = array_2d.Array2D.manual_mask(
                array=deflections.slim[:, 1], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_x,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="deflections X", filename="deflections_x"
                ),
            )

        if magnification:

            self.mat_plot_2d.plot_array(
                array=self.lensing_obj.magnification_from_grid(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Magnification", filename="magnification"
                ),
            )

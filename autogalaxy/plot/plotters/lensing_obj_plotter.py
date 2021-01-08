from autoarray.structures import arrays, grids
from autoarray.plot.plotters import abstract_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals


class LensingObjPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        lensing_obj,
        grid: grids.Grid,
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
                "origin", value=grids.GridIrregular(grid=[self.grid.origin])
            ),
            mask=self.extract_2d("mask", value=self.grid.mask),
            border=self.extract_2d(
                "border", value=self.grid.mask.geometry.border_grid_sub_1.in_1d_binned
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres", self.lensing_obj.mass_profile_centres
            ),
            critical_curves=self.extract_2d(
                "critical_curves", self.lensing_obj.critical_curves, "critical_curves"
            ),
        )

    @abstract_plotters.for_figure
    def figure_convergence(self):
        """Plot the convergence of a mass profile, on a grid of (y,x) coordinates.

        Set *autogalaxy.hyper_galaxies.arrays.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        mass_profile : model.profiles.mass_profiles.MassProfile
            The mass profile whose convergence is plotted.
        grid : grid_like
            The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
        """
        self.mat_plot_2d.plot_array(
            array=self.lensing_obj.convergence_from_grid(grid=self.grid),
            visuals_2d=self.visuals_with_include_2d,
        )

    @abstract_plotters.for_figure
    def figure_potential(self):
        """Plot the potential of a mass profile, on a grid of (y,x) coordinates.

        Set *autogalaxy.hyper_galaxies.arrays.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        mass_profile : model.profiles.mass_profiles.MassProfile
            The mass profile whose potential is plotted.
        grid : grid_like
            The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
        """
        self.mat_plot_2d.plot_array(
            array=self.lensing_obj.potential_from_grid(grid=self.grid),
            visuals_2d=self.visuals_with_include_2d,
        )

    @abstract_plotters.for_figure
    def figure_deflections_y(self):
        """Plot the y component of the deflection angles of a mass profile, on a grid of (y,x) coordinates.

        Set *autogalaxy.hyper_galaxies.arrays.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        mass_profile : model.profiles.mass_profiles.MassProfile
            The mass profile whose y deflecton angles are plotted.
        grid : grid_like
            The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
        """

        deflections = self.lensing_obj.deflections_from_grid(grid=self.grid)
        deflections_y = arrays.Array.manual_mask(
            array=deflections.in_1d[:, 0], mask=self.grid.mask
        )

        self.mat_plot_2d.plot_array(
            array=deflections_y, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_figure
    def figure_deflections_x(self):
        """Plot the x component of the deflection angles of a mass profile, on a grid of (y,x) coordinates.

        Set *autogalaxy.hyper_galaxies.arrays.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        mass_profile : model.profiles.mass_profiles.MassProfile
            The mass profile whose x deflecton angles are plotted.
        grid : grid_like
            The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
        """
        deflections = self.lensing_obj.deflections_from_grid(grid=self.grid)
        deflections_x = arrays.Array.manual_mask(
            array=deflections.in_1d[:, 1], mask=self.grid.mask
        )

        self.mat_plot_2d.plot_array(
            array=deflections_x, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_figure
    def figure_magnification(self):
        """Plot the magnification of a mass profile, on a grid of (y,x) coordinates.

        Set *autogalaxy.hyper_galaxies.arrays.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        mass_profile : model.profiles.mass_profiles.MassProfile
            The mass profile whose magnification is plotted.
        grid : grid_like
            The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
        """
        self.mat_plot_2d.plot_array(
            array=self.lensing_obj.magnification_from_grid(grid=self.grid),
            visuals_2d=self.visuals_with_include_2d,
        )

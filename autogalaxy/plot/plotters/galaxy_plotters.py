from autoarray.plot.plotters import abstract_plotters
from autogalaxy.plot.plotters import lensing_obj_plotter
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.plot.plotters import light_profile_plotters, mass_profile_plotters


class GalaxyPlotter(lensing_obj_plotter.LensingObjPlotter):
    def __init__(
        self,
        galaxy,
        grid,
        mat_plot_1d: lensing_mat_plot.MatPlot1D = lensing_mat_plot.MatPlot1D(),
        visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
        include_1d: lensing_include.Include1D = lensing_include.Include1D(),
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):
        super().__init__(
            lensing_obj=galaxy,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
        )

    @property
    def galaxy(self):
        return self.lensing_obj

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

        visuals_2d = super(GalaxyPlotter, self).visuals_with_include_2d

        light_profile_centres = (
            self.galaxy.light_profile_centres
            if self.include_2d.light_profile_centres
            else None
        )

        return visuals_2d + lensing_visuals.Visuals2D(
            light_profile_centres=light_profile_centres
        )

    def light_profile_plotter_from(self, light_profile):
        return light_profile_plotters.LightProfilePlotter(
            light_profile=light_profile,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_1d,
            include_1d=self.include_1d,
        )

    def mass_profile_plotter_from(self, mass_profile):
        return mass_profile_plotters.MassProfilePlotter(
            mass_profile=mass_profile,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_1d,
            include_1d=self.include_1d,
        )

    @abstract_plotters.for_figure
    def figure_image(self):
        """Plot the image (e.g. the datas) of a galaxy, on a grid of (y,x) coordinates.

        Set *autogalaxy.datas.arrays.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        galaxy : model.galaxy.ag.Galaxy
            The galaxy whose image are plotted.
        grid : grid_like or datas.arrays.grid_stacks.Grid
            The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
        """
        self.mat_plot_2d.plot_array(
            array=self.galaxy.image_from_grid(grid=self.grid),
            visuals_2d=self.visuals_with_include_2d,
        )

    @abstract_plotters.for_figure
    def figure_contribution_map(self):
        """Plot the summed contribution maps of a hyper_galaxies-fit.

        Set *autogalaxy.datas.arrays.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The hyper_galaxies-fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
        image_index : int
            The index of the datas in the datas-set of which the contribution_maps are plotted.
        """
        self.mat_plot_2d.plot_array(
            array=self.galaxy.contribution_map, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_subplot
    def subplot_image(self):

        number_subplots = len(self.galaxy.light_profiles)

        self.open_subplot_figure(number_subplots=number_subplots)

        for i, light_profile in enumerate(self.galaxy.light_profiles):

            light_profile_plotter = self.light_profile_plotter_from(
                light_profile=light_profile
            )

            self.setup_subplot(number_subplots=number_subplots, subplot_index=i + 1)

            light_profile_plotter.figure_image()

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()

    @abstract_plotters.for_subplot
    def subplot_convergence(self):

        number_subplots = len(self.galaxy.mass_profiles)

        self.open_subplot_figure(number_subplots=number_subplots)

        for i, mass_profile in enumerate(self.galaxy.mass_profiles):

            mass_profile_plotter = self.mass_profile_plotter_from(
                mass_profile=mass_profile
            )

            self.setup_subplot(number_subplots=number_subplots, subplot_index=i + 1)

            mass_profile_plotter.figure_convergence()

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()

    @abstract_plotters.for_subplot
    def subplot_potential(self):

        number_subplots = len(self.galaxy.mass_profiles)

        self.open_subplot_figure(number_subplots=number_subplots)

        for i, mass_profile in enumerate(self.galaxy.mass_profiles):

            mass_profile_plotter = self.mass_profile_plotter_from(
                mass_profile=mass_profile
            )

            self.setup_subplot(number_subplots=number_subplots, subplot_index=i + 1)

            mass_profile_plotter.figure_potential()

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()

    @abstract_plotters.for_subplot
    def subplot_deflections_y(self):

        number_subplots = len(self.galaxy.mass_profiles)

        self.open_subplot_figure(number_subplots=number_subplots)

        for i, mass_profile in enumerate(self.galaxy.mass_profiles):

            mass_profile_plotter = self.mass_profile_plotter_from(
                mass_profile=mass_profile
            )

            self.setup_subplot(number_subplots=number_subplots, subplot_index=i + 1)

            mass_profile_plotter.figure_deflections_y()

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()

    @abstract_plotters.for_subplot
    def subplot_deflections_x(self):

        number_subplots = len(self.galaxy.mass_profiles)

        self.open_subplot_figure(number_subplots=number_subplots)

        for i, mass_profile in enumerate(self.galaxy.mass_profiles):

            mass_profile_plotter = self.mass_profile_plotter_from(
                mass_profile=mass_profile
            )

            self.setup_subplot(number_subplots=number_subplots, subplot_index=i + 1)

            mass_profile_plotter.figure_deflections_x()

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()

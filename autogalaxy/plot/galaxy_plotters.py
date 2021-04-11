from autoarray.plot.mat_wrap import mat_plot
from autogalaxy.plot import (
    lensing_obj_plotter,
    light_profile_plotters,
    mass_profile_plotters,
)
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.profiles import light_profiles as lp, mass_profiles as mp


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
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
        )

        self.galaxy = galaxy
        self.grid = grid

    @property
    def lensing_obj(self):
        return self.galaxy

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

        visuals_2d = super().visuals_with_include_2d

        return visuals_2d + visuals_2d.__class__(
            light_profile_centres=self.extract_2d(
                "light_profile_centres",
                self.galaxy.extract_attribute(cls=lp.LightProfile, attr_name="centre"),
            )
        )

    def light_profile_plotter_from(
        self, light_profile: lp.LightProfile
    ) -> light_profile_plotters.LightProfilePlotter:
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

    def mass_profile_plotter_from(
        self, mass_profile: mp.MassProfile
    ) -> mass_profile_plotters.MassProfilePlotter:
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

    # def figures_1d(self, convergence=False, potential=False):
    #
    #     if convergence:
    #
    #         galaxy_convergence = values.ValuesIrregular
    #
    #         for mass_profile in self.galaxy.mass_profiles:
    #
    #             mass_profile_plotter = self.mass_profile_plotter_from(mass_profile=mass_profile)
    #
    #             convergence = mass_profile_plotter.mass_profile.convergence_2d_from_grid(grid=mass_profile_plotter.grid_2d_radial_projected)
    #             grid_1d_radial_distances = mass_profile_plotter.grid_1d_radial_distances
    #
    #             self.mat_plot_1d.plot_yx(
    #                 y=self.galaxy.convergence_2d_from_grid(grid=self.grid_2d_radial_projected),
    #                 x=self.grid_1d_radial_distances,
    #                 visuals_1d=self.visuals_with_include_1d,
    #                 auto_labels=mat_plot.AutoLabels(
    #                     title="Convergence vs Radius",
    #                     ylabel="Convergence ",
    #                     xlabel="Radius",
    #                     legend=self.galaxy.__class__.__name__,
    #                     filename="convergence_1d",
    #                 ),
    #             )

    # if potential:
    #
    #     self.mat_plot_1d.plot_yx(
    #         y=self.galaxy.potential_2d_from_grid(grid=self.grid_2d_radial_projected),
    #         x=self.grid_1d_radial_distances,
    #         visuals_1d=self.visuals_with_include_1d,
    #         auto_labels=mat_plot.AutoLabels(
    #             title="Potential vs Radius",
    #             ylabel="Potential ",
    #             xlabel="Radius",
    #             legend=self.galaxy.__class__.__name__,
    #             filename="potential_1d",
    #         ),
    #     )

    def figures_2d(
        self,
        image=False,
        convergence=False,
        potential=False,
        deflections_y=False,
        deflections_x=False,
        magnification=False,
        contribution_map=False,
    ):

        if image:

            self.mat_plot_2d.plot_array(
                array=self.galaxy.image_2d_from_grid(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mat_plot.AutoLabels(title="Image", filename="image"),
            )

        super().figures_2d(
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
        )

        if contribution_map:

            self.mat_plot_2d.plot_array(
                array=self.galaxy.contribution_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mat_plot.AutoLabels(
                    title="Contribution Map", filename="contribution_map"
                ),
            )

    def subplot_of_light_profiles(self, image=False):

        light_profile_plotters = [
            self.light_profile_plotter_from(light_profile)
            for light_profile in self.galaxy.light_profiles
        ]

        if image:
            self.subplot_of_plotters_figure(
                plotter_list=light_profile_plotters, name="image"
            )

    def subplot_of_mass_profiles(
        self,
        convergence=False,
        potential=False,
        deflections_y=False,
        deflections_x=False,
    ):

        mass_profile_plotters = [
            self.mass_profile_plotter_from(mass_profile)
            for mass_profile in self.galaxy.mass_profiles
        ]

        if convergence:
            self.subplot_of_plotters_figure(
                plotter_list=mass_profile_plotters, name="convergence"
            )

        if potential:
            self.subplot_of_plotters_figure(
                plotter_list=mass_profile_plotters, name="potential"
            )

        if deflections_y:
            self.subplot_of_plotters_figure(
                plotter_list=mass_profile_plotters, name="deflections_y"
            )

        if deflections_x:
            self.subplot_of_plotters_figure(
                plotter_list=mass_profile_plotters, name="deflections_x"
            )

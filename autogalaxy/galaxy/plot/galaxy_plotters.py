import math
from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.lensing_obj_plotter import LensingObjPlotter
from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.profiles.mass_profiles import MassProfile
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plot.mat_wrap.lensing_mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.lensing_mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.lensing_visuals import Visuals1D
from autogalaxy.plot.mat_wrap.lensing_visuals import Visuals2D
from autogalaxy.plot.mat_wrap.lensing_include import Include1D
from autogalaxy.plot.mat_wrap.lensing_include import Include2D
from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePlotter
from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePDFPlotter
from autogalaxy.profiles.plot.mass_profile_plotters import MassProfilePlotter
from autogalaxy.profiles.plot.mass_profile_plotters import MassProfilePDFPlotter

from autogalaxy.util import error_util


class GalaxyPlotter(LensingObjPlotter):
    def __init__(
        self,
        galaxy: Galaxy,
        grid: aa.Grid2D,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
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
    def lensing_obj(self) -> Galaxy:
        return self.galaxy

    @property
    def visuals_with_include_2d(self) -> Visuals2D:
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
                self.galaxy.extract_attribute(cls=LightProfile, attr_name="centre"),
            )
        )

    def light_profile_plotter_from(
        self, light_profile: LightProfile
    ) -> LightProfilePlotter:
        return LightProfilePlotter(
            light_profile=light_profile,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_with_include_1d_light,
            include_1d=self.include_1d,
        )

    def mass_profile_plotter_from(
        self, mass_profile: MassProfile
    ) -> MassProfilePlotter:
        return MassProfilePlotter(
            mass_profile=mass_profile,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_1d,
            include_1d=self.include_1d,
        )

    @property
    def visuals_with_include_1d_light(self) -> Visuals1D:
        """
        Extracts from the `Galaxy` attributes that can be plotted which are associated with light profiles and returns
        them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `GalaxyPlotter` the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the `LightProfile`'s total integrated luminosity.

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """
        return self.visuals_1d

    @property
    def visuals_with_include_1d_mass(self) -> Visuals1D:
        """
        Extracts from the `Galaxy` attributes that can be plotted which are associated with mass profiles and returns
        them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `GalaxyPlotter` the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the `LightProfile`'s total integrated luminosity.

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """
        if self.include_1d.einstein_radius:
            einstein_radius = self.lensing_obj.einstein_radius_from(grid=self.grid)
        else:
            einstein_radius = None

        return self.visuals_1d + self.visuals_1d.__class__(
            einstein_radius=einstein_radius
        )

    def figures_1d(
        self, image: bool = False, convergence: bool = False, potential: bool = False
    ):

        if self.mat_plot_1d.yx_plot.plot_axis_type is None:
            plot_axis_type_override = "semilogy"
        else:
            plot_axis_type_override = None

        if image:

            image_1d = self.galaxy.image_1d_from(grid=self.grid)

            self.mat_plot_1d.plot_yx(
                y=image_1d,
                x=image_1d.grid_radial,
                visuals_1d=self.visuals_with_include_1d_light,
                auto_labels=aplt.AutoLabels(
                    title="Image vs Radius",
                    ylabel="Image ",
                    xlabel="Radius",
                    legend=self.lensing_obj.__class__.__name__,
                    filename="image_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

        if convergence:

            convergence_1d = self.galaxy.convergence_1d_from(grid=self.grid)

            self.mat_plot_1d.plot_yx(
                y=convergence_1d,
                x=convergence_1d.grid_radial,
                visuals_1d=self.visuals_with_include_1d_mass,
                auto_labels=aplt.AutoLabels(
                    title="Convergence vs Radius",
                    ylabel="Convergence ",
                    xlabel="Radius",
                    legend=self.lensing_obj.__class__.__name__,
                    filename="convergence_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

        if potential:

            potential_1d = self.galaxy.potential_1d_from(grid=self.grid)

            self.mat_plot_1d.plot_yx(
                y=potential_1d,
                x=potential_1d.grid_radial,
                visuals_1d=self.visuals_with_include_1d_mass,
                auto_labels=aplt.AutoLabels(
                    title="Potential vs Radius",
                    ylabel="Potential ",
                    xlabel="Radius",
                    legend=self.lensing_obj.__class__.__name__,
                    filename="potential_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

    def figures_1d_decomposed(
        self,
        image: bool = False,
        convergence: bool = False,
        potential: bool = False,
        legend_labels: List[str] = None,
    ):

        plotter_list = [self]

        for i, light_profile in enumerate(self.galaxy.light_profiles):

            light_profile_plotter = self.light_profile_plotter_from(
                light_profile=light_profile
            )

            plotter_list.append(light_profile_plotter)

        multi_plotter = aplt.MultiYX1DPlotter(
            plotter_list=plotter_list, legend_labels=legend_labels
        )

        if image:

            change_filename = False

            if multi_plotter.plotter_list[0].mat_plot_1d.output.filename is None:
                multi_plotter.plotter_list[0].set_filename(
                    filename="image_1d_decomposed"
                )
                change_filename = True

            multi_plotter.figure_1d(func_name="figures_1d", figure_name="image")

            if change_filename:
                multi_plotter.plotter_list[0].set_filename(filename=None)

        plotter_list = [self] + [
            self.mass_profile_plotter_from(mass_profile=mass_profile)
            for mass_profile in self.galaxy.mass_profiles
        ]

        multi_plotter = aplt.MultiYX1DPlotter(
            plotter_list=plotter_list, legend_labels=legend_labels
        )

        if convergence:

            change_filename = False

            if multi_plotter.plotter_list[0].mat_plot_1d.output.filename is None:
                multi_plotter.plotter_list[0].set_filename(
                    filename="convergence_1d_decomposed"
                )
                change_filename = True

            multi_plotter.figure_1d(func_name="figures_1d", figure_name="convergence")

            if change_filename:
                multi_plotter.plotter_list[0].set_filename(filename=None)

        if potential:

            change_filename = False

            if multi_plotter.plotter_list[0].mat_plot_1d.output.filename is None:
                multi_plotter.plotter_list[0].set_filename(
                    filename="potential_1d_decomposed"
                )
                change_filename = True

            multi_plotter.figure_1d(func_name="figures_1d", figure_name="potential")

            if change_filename:
                multi_plotter.plotter_list[0].set_filename(filename=None)

    def figures_2d(
        self,
        image: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        contribution_map: bool = False,
    ):

        if image:

            self.mat_plot_2d.plot_array(
                array=self.galaxy.image_2d_from(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(title="Image", filename="image_2d"),
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
                auto_labels=aplt.AutoLabels(
                    title="Contribution Map", filename="contribution_map_2d"
                ),
            )

    def subplot_of_light_profiles(self, image: bool = False):

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
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
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


class GalaxyPDFPlotter(GalaxyPlotter):
    def __init__(
        self,
        galaxy_pdf_list: List[Galaxy],
        grid: aa.Grid2D,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
        sigma: Optional[float] = 3.0,
    ):
        super().__init__(
            galaxy=None,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
        )

        self.galaxy_pdf_list = galaxy_pdf_list
        self.sigma = sigma
        self.low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    @property
    def light_profile_pdf_plotter_list(self) -> List[LightProfilePDFPlotter]:
        return [
            self.light_profile_pdf_plotter_from(index=index)
            for index in range(len(self.galaxy_pdf_list[0].light_profiles))
        ]

    def light_profile_pdf_plotter_from(self, index) -> LightProfilePDFPlotter:

        light_profile_pdf_list = [
            galaxy.light_profiles[index] for galaxy in self.galaxy_pdf_list
        ]

        return LightProfilePDFPlotter(
            light_profile_pdf_list=light_profile_pdf_list,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_with_include_1d_light,
            include_1d=self.include_1d,
        )

    @property
    def mass_profile_pdf_plotter_list(self) -> List[MassProfilePDFPlotter]:
        return [
            self.mass_profile_pdf_plotter_from(index=index)
            for index in range(len(self.galaxy_pdf_list[0].mass_profiles))
        ]

    def mass_profile_pdf_plotter_from(self, index) -> MassProfilePDFPlotter:

        mass_profile_pdf_list = [
            galaxy.mass_profiles[index] for galaxy in self.galaxy_pdf_list
        ]

        return MassProfilePDFPlotter(
            mass_profile_pdf_list=mass_profile_pdf_list,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_with_include_1d_mass,
            include_1d=self.include_1d,
        )

    @property
    def visuals_with_include_1d_light(self) -> Visuals1D:
        """
        Extracts from the `Galaxy` attributes that can be plotted which are associated with light profiles and returns
        them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `GalaxyPlotter` the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the `LightProfile`'s total integrated luminosity.

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """
        return self.visuals_1d

    @property
    def visuals_with_include_1d_mass(self) -> Visuals1D:
        """
        Extracts from the `Galaxy` attributes that can be plotted which are associated with mass profiles and returns
        them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `GalaxyPlotter` the following 1D attributes can be extracted for plotting:

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """
        if self.include_1d.einstein_radius:

            einstein_radius_list = [
                galaxy.einstein_radius_from(grid=self.grid)
                for galaxy in self.galaxy_pdf_list
            ]

            einstein_radius, einstein_radius_errors = error_util.value_median_and_error_region_via_quantile(
                value_list=einstein_radius_list, low_limit=self.low_limit
            )

        else:

            einstein_radius = None
            einstein_radius_errors = None

        return self.visuals_1d + self.visuals_1d.__class__(
            self.extract_1d("einstein_radius", value=einstein_radius),
            self.extract_1d("einstein_radius", value=einstein_radius_errors),
        )

    def figures_1d(
        self, image: bool = False, convergence: bool = False, potential: bool = False
    ):

        if self.mat_plot_1d.yx_plot.plot_axis_type is None:
            plot_axis_type_override = "semilogy"
        else:
            plot_axis_type_override = None

        if image:

            grid_radial = (
                self.galaxy_pdf_list[0].image_1d_from(grid=self.grid).grid_radial
            )

            image_1d_list = [
                light_profile.image_1d_from(grid=self.grid)
                for light_profile in self.galaxy_pdf_list
            ]

            median_image_1d, errors_image_1d = error_util.profile_1d_median_and_error_region_via_quantile(
                profile_1d_list=image_1d_list, low_limit=self.low_limit
            )

            visuals_1d = self.visuals_with_include_1d_light + self.visuals_1d.__class__(
                shaded_region=errors_image_1d
            )

            self.mat_plot_1d.plot_yx(
                y=median_image_1d,
                x=grid_radial,
                visuals_1d=visuals_1d,
                auto_labels=aplt.AutoLabels(
                    title="Image vs Radius",
                    ylabel="Image ",
                    xlabel="Radius",
                    legend=self.galaxy_pdf_list[0].__class__.__name__,
                    filename="image_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

        if convergence:

            grid_radial = (
                self.galaxy_pdf_list[0].convergence_1d_from(grid=self.grid).grid_radial
            )

            convergence_1d_list = [
                light_profile.convergence_1d_from(grid=self.grid)
                for light_profile in self.galaxy_pdf_list
            ]

            median_convergence_1d, errors_convergence_1d = error_util.profile_1d_median_and_error_region_via_quantile(
                profile_1d_list=convergence_1d_list, low_limit=self.low_limit
            )

            visuals_1d = self.visuals_with_include_1d_mass + self.visuals_1d.__class__(
                shaded_region=errors_convergence_1d
            )

            self.mat_plot_1d.plot_yx(
                y=median_convergence_1d,
                x=grid_radial,
                visuals_1d=visuals_1d,
                auto_labels=aplt.AutoLabels(
                    title="Convergence vs Radius",
                    ylabel="Convergence ",
                    xlabel="Radius",
                    legend=self.galaxy_pdf_list[0].__class__.__name__,
                    filename="convergence_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

        if potential:

            grid_radial = (
                self.galaxy_pdf_list[0].potential_1d_from(grid=self.grid).grid_radial
            )

            potential_1d_list = [
                light_profile.potential_1d_from(grid=self.grid)
                for light_profile in self.galaxy_pdf_list
            ]

            median_potential_1d, errors_potential_1d = error_util.profile_1d_median_and_error_region_via_quantile(
                profile_1d_list=potential_1d_list, low_limit=self.low_limit
            )

            visuals_1d = self.visuals_with_include_1d_mass + self.visuals_1d.__class__(
                shaded_region=errors_potential_1d
            )

            self.mat_plot_1d.plot_yx(
                y=median_potential_1d,
                x=grid_radial,
                visuals_1d=visuals_1d,
                auto_labels=aplt.AutoLabels(
                    title="Potential vs Radius",
                    ylabel="Potential ",
                    xlabel="Radius",
                    legend=self.galaxy_pdf_list[0].__class__.__name__,
                    filename="potential_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

    def figures_1d_decomposed(
        self,
        image: bool = False,
        convergence: bool = False,
        potential: bool = False,
        legend_labels: List[str] = None,
    ):

        if image:

            multi_plotter = aplt.MultiYX1DPlotter(
                plotter_list=[self] + self.light_profile_pdf_plotter_list,
                legend_labels=legend_labels,
            )

            change_filename = False

            if multi_plotter.plotter_list[0].mat_plot_1d.output.filename is None:
                multi_plotter.plotter_list[0].set_filename(
                    filename="image_1d_decomposed"
                )

                change_filename = True

            multi_plotter.figure_1d(func_name="figures_1d", figure_name="image")

            if change_filename:
                multi_plotter.plotter_list[0].set_filename(filename=None)

        if convergence:

            multi_plotter = aplt.MultiYX1DPlotter(
                plotter_list=[self] + self.mass_profile_pdf_plotter_list,
                legend_labels=legend_labels,
            )

            change_filename = False

            if multi_plotter.plotter_list[0].mat_plot_1d.output.filename is None:
                multi_plotter.plotter_list[0].set_filename(
                    filename="convergence_1d_decomposed"
                )

                change_filename = True

            multi_plotter.figure_1d(func_name="figures_1d", figure_name="convergence")

            if change_filename:
                multi_plotter.plotter_list[0].set_filename(filename=None)

        if potential:

            multi_plotter = aplt.MultiYX1DPlotter(
                plotter_list=[self] + self.mass_profile_pdf_plotter_list,
                legend_labels=legend_labels,
            )

            change_filename = False

            if multi_plotter.plotter_list[0].mat_plot_1d.output.filename is None:
                multi_plotter.plotter_list[0].set_filename(
                    filename="potential_1d_decomposed"
                )

                change_filename = True

            multi_plotter.figure_1d(func_name="figures_1d", figure_name="potential")

            if change_filename:
                multi_plotter.plotter_list[0].set_filename(filename=None)

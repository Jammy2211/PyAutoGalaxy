import math
from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.lensing_obj_plotter import LensingObjPlotter
from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.profiles.mass_profiles import MassProfile
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals1D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include1D
from autogalaxy.plot.mat_wrap.include import Include2D
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

    def light_profile_plotter_from(
        self, light_profile: LightProfile
    ) -> LightProfilePlotter:
        return LightProfilePlotter(
            light_profile=light_profile,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_2d.via_light_obj_from(
                light_obj=light_profile, grid=self.grid
            ),
            include_2d=self.include_2d,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.get_1d.via_light_obj_from(light_obj=light_profile),
            include_1d=self.include_1d,
        )

    def mass_profile_plotter_from(
        self, mass_profile: MassProfile
    ) -> MassProfilePlotter:
        return MassProfilePlotter(
            mass_profile=mass_profile,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_2d.via_lensing_obj_from(
                lensing_obj=mass_profile, grid=self.grid
            ),
            include_2d=self.include_2d,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.get_1d.via_lensing_obj_from(
                lensing_obj=mass_profile, grid=self.grid
            ),
            include_1d=self.include_1d,
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
                visuals_1d=self.get_1d.via_light_obj_from(light_obj=self.galaxy),
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
                visuals_1d=self.get_1d.via_light_lensing_obj_from(
                    light_lensing_obj=self.galaxy, grid=self.grid
                ),
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
                visuals_1d=self.get_1d.via_light_lensing_obj_from(
                    light_lensing_obj=self.galaxy, grid=self.grid
                ),
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
                visuals_2d=self.get_2d.via_light_obj_from(
                    light_obj=self.galaxy, grid=self.grid
                ),
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
                visuals_2d=self.visuals_2d,
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
            visuals_1d=self.get_1d.via_light_obj_from(light_obj=self.galaxy),
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
            visuals_1d=self.get_1d.via_light_lensing_obj_from(
                light_lensing_obj=self.galaxy, grid=self.grid
            ),
            include_1d=self.include_1d,
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

            visuals_1d_via_light_obj_list = self.get_1d.via_light_obj_list_from(
                light_obj_list=self.galaxy_pdf_list, low_limit=self.low_limit
            )
            visuals_1d_with_shaded_region = self.visuals_1d.__class__(
                shaded_region=errors_image_1d
            )

            visuals_1d = visuals_1d_via_light_obj_list + visuals_1d_with_shaded_region

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

            visuals_1d_via_lensing_obj_list = self.get_1d.via_lensing_obj_list_from(
                lensing_obj_list=self.galaxy_pdf_list,
                grid=self.grid,
                low_limit=self.low_limit,
            )
            visuals_1d_with_shaded_region = self.visuals_1d.__class__(
                shaded_region=errors_convergence_1d
            )

            visuals_1d = visuals_1d_via_lensing_obj_list + visuals_1d_with_shaded_region

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

            visuals_1d = self.get_1d.via_light_lensing_obj_from(
                light_lensing_obj=self.galaxy, grid=self.grid
            ) + self.visuals_1d.__class__(shaded_region=errors_potential_1d)

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

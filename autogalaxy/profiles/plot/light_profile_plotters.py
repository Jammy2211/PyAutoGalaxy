import math
from typing import List, Optional

import autoarray.plot as aplt
from autoarray.structures.grids.two_d import abstract_grid_2d

from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.plot import abstract_plotters
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals1D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include1D
from autogalaxy.plot.mat_wrap.include import Include2D
from autogalaxy.util import error_util


class LightProfilePlotter(abstract_plotters.Plotter):
    def __init__(
        self,
        light_profile: LightProfile,
        grid: abstract_grid_2d.AbstractGrid2D,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
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

    def get_visuals_1d(self) -> Visuals1D:
        return self.get_1d.via_light_obj_from(light_obj=self.light_profile)

    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_light_obj_from(
            light_obj=self.light_profile, grid=self.grid
        )


    def figures_1d(self, image: bool = False):

        if self.mat_plot_1d.yx_plot.plot_axis_type is None:
            plot_axis_type_override = "semilogy"
        else:
            plot_axis_type_override = None

        if image:

            image_1d = self.light_profile.image_1d_from(grid=self.grid)

            self.mat_plot_1d.plot_yx(
                y=image_1d,
                x=image_1d.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=aplt.AutoLabels(
                    title="Image vs Radius",
                    ylabel="Image",
                    xlabel="Radius",
                    legend=self.light_profile.__class__.__name__,
                    filename="image_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

    def figures_2d(self, image: bool = False):

        if image:

            self.mat_plot_2d.plot_array(
                array=self.light_profile.image_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(title="Image", filename="image_2d"),
            )


class LightProfilePDFPlotter(LightProfilePlotter):
    def __init__(
        self,
        light_profile_pdf_list: List[LightProfile],
        grid: abstract_grid_2d.AbstractGrid2D,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
        sigma: Optional[float] = 3.0,
    ):

        super().__init__(
            light_profile=None,
            grid=grid,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

        self.light_profile_pdf_list = light_profile_pdf_list
        self.sigma = sigma
        self.low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    def figures_1d(self, image: bool = False):

        if self.mat_plot_1d.yx_plot.plot_axis_type is None:
            plot_axis_type_override = "semilogy"
        else:
            plot_axis_type_override = None

        if image:

            grid_radial = (
                self.light_profile_pdf_list[0].image_1d_from(grid=self.grid).grid_radial
            )

            image_1d_list = [
                light_profile.image_1d_from(grid=self.grid)
                for light_profile in self.light_profile_pdf_list
            ]

            median_image_1d, errors_image_1d = error_util.profile_1d_median_and_error_region_via_quantile(
                profile_1d_list=image_1d_list, low_limit=self.low_limit
            )

            visuals_1d_via_light_obj_list = self.get_1d.via_light_obj_list_from(
                light_obj_list=self.light_profile_pdf_list, low_limit=self.low_limit
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
                    ylabel="Image",
                    xlabel="Radius",
                    legend=self.light_profile_pdf_list[0].__class__.__name__,
                    filename="image_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

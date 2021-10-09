import autoarray as aa
import autoarray.plot as aplt
from autoarray.structures.grids.two_d import abstract_grid_2d
from autoarray.plot import abstract_plotters

from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.plot.mat_wrap.lensing_mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.lensing_mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.lensing_visuals import Visuals1D
from autogalaxy.plot.mat_wrap.lensing_visuals import Visuals2D
from autogalaxy.plot.mat_wrap.lensing_include import Include1D
from autogalaxy.plot.mat_wrap.lensing_include import Include2D

from autogalaxy.util import error_util

import math

from typing import List, Optional


class LightProfilePlotter(abstract_plotters.AbstractPlotter):
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

    @property
    def visuals_with_include_1d(self) -> Visuals1D:
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
            self.extract_1d(
                "half_light_radius", value=self.light_profile.half_light_radius
            )
        )

    @property
    def visuals_with_include_2d(self) -> Visuals2D:
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
                "origin", value=aa.Grid2DIrregular(grid=[self.grid.origin])
            ),
            mask=self.extract_2d("mask", value=self.grid.mask),
            border=self.extract_2d(
                "border", value=self.grid.mask.border_grid_sub_1.binned
            ),
            light_profile_centres=self.extract_2d(
                "light_profile_centres",
                aa.Grid2DIrregular(grid=[self.light_profile.centre]),
            ),
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
                visuals_1d=self.visuals_with_include_1d,
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
                visuals_2d=self.visuals_with_include_2d,
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

    @property
    def visuals_with_include_1d(self) -> Visuals1D:
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

        if self.include_1d.half_light_radius:

            half_light_radius_list = [
                light_profile.half_light_radius
                for light_profile in self.light_profile_pdf_list
            ]

            half_light_radius, half_light_radius_errors = error_util.value_median_and_error_region_via_quantile(
                value_list=half_light_radius_list, low_limit=self.low_limit
            )

        else:

            half_light_radius = None
            half_light_radius_errors = None

        return self.visuals_1d + self.visuals_1d.__class__(
            self.extract_1d("half_light_radius", value=half_light_radius),
            self.extract_1d("half_light_radius", value=half_light_radius_errors),
        )

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

            visuals_1d = self.visuals_with_include_1d + self.visuals_1d.__class__(
                shaded_region=errors_image_1d
            )

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

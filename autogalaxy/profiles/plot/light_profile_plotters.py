import math
from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt


from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.visuals.two_d import Visuals2D

from autogalaxy.util import error_util
from autogalaxy import exc


class LightProfilePlotter(Plotter):
    def __init__(
        self,
        light_profile: LightProfile,
        grid: aa.type.Grid1D2DLike,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        """
        Plots the attributes of `LightProfile` objects using the matplotlib methods `plot()` and `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `LightProfile` and plotted via the visuals object.

        Parameters
        ----------
        light_profile
            The light profile the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the light profile quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        """

        from autogalaxy.profiles.light.linear import (
            LightProfileLinear,
        )

        if isinstance(light_profile, LightProfileLinear):
            raise exc.raise_linear_light_profile_in_plot(
                plotter_type=self.__class__.__name__,
            )

        self.light_profile = light_profile
        self.grid = grid

        super().__init__(
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
        )

    def get_visuals_1d(self) -> Visuals1D:
        return self.get_1d.via_light_obj_from(light_obj=self.light_profile)

    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_light_obj_from(
            light_obj=self.light_profile, grid=self.grid
        )

    def figures_1d(self, image: bool = False):
        """
        Plots the individual attributes of the plotter's `LightProfile` object in 1D, which are computed via the
        plotter's grid object.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from the light profile. This is performed by aligning a 1D grid with the  major-axis of the light
        profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        The API is such that every plottable attribute of the `LightProfile` object is an input parameter of type
        bool of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 1D plot (via `plot`) of the image.
        """
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
                    title=r"Image ($\mathrm{e^{-}}\,\mathrm{s^{-1}}$) vs Radius (arcsec)",
                    yunit="",
                    legend=self.light_profile.__class__.__name__,
                    filename="image_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

    def figures_2d(self, image: bool = False):
        """
        Plots the individual attributes of the plotter's `LightProfile` object in 2D, which are computed via the
        plotter's 2D grid object.

        The API is such that every plottable attribute of the `LightProfile` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 2D plot (via `imshow`) of the image.
        """
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
        grid: aa.type.Grid2DLike,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
        sigma: Optional[float] = 3.0,
    ):
        """
        Plots the attributes of a list of `LightProfile` objects using the matplotlib methods `plot()` and `imshow()`
        and many other matplotlib functions which customize the plot's appearance.

        Figures plotted by this object average over a list light profiles to computed the average value of each
        attribute with errors, where the 1D regions within the errors are plotted as a shaded region to show the range
        of plausible models. Therefore, the input list of galaxies is expected to represent the probability density
        function of an inferred model-fit.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `LightProfile` and plotted via the visuals object.

        Parameters
        ----------
        light_profile_pdf_list
            The list of light profiles whose mean and error values the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the light profile quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        sigma
            The confidence interval in terms of a sigma value at which the errors are computed (e.g. a value of
            sigma=3.0 uses confidence intevals at ~0.01 and 0.99 the PDF).
        """
        super().__init__(
            light_profile=None,
            grid=grid,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
        )

        self.light_profile_pdf_list = light_profile_pdf_list
        self.sigma = sigma
        self.low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    def figures_1d(self, image: bool = False):
        """
        Plots the individual attributes of the plotter's list of ` LightProfile` object in 1D, which are computed via
        the plotter's grid object.

        This averages over a list light profiles to compute the average value of each attribute with errors, where the
        1D regions within the errors are plotted as a shaded region to show the range of plausible models. Therefore,
        the input list of galaxies is expected to represent the probability density function of an inferred model-fit.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from each light profile. This is performed by aligning a 1D grid with the major-axis of
        each light profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        The API is such that every plottable attribute of the `LightProfile` object is an input parameter of type bool
        of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 1D plot (via `plot`) of the image.
        convergence
            Whether to make a 1D plot (via `imshow`) of the convergence.
        potential
            Whether to make a 1D plot (via `imshow`) of the potential.
        """
        if self.mat_plot_1d.yx_plot.plot_axis_type is None:
            plot_axis_type_override = "semilogy"
        else:
            plot_axis_type_override = None

        if image:
            image_1d_list = [
                light_profile.image_1d_from(grid=self.grid)
                for light_profile in self.light_profile_pdf_list
            ]

            min_index = min([image_1d.shape[0] for image_1d in image_1d_list])
            image_1d_list = [image_1d[0:min_index] for image_1d in image_1d_list]

            (
                median_image_1d,
                errors_image_1d,
            ) = error_util.profile_1d_median_and_error_region_via_quantile(
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
                x=image_1d_list[0].grid_radial,
                visuals_1d=visuals_1d,
                auto_labels=aplt.AutoLabels(
                    title=r"Image ($\mathrm{e^{-}}\,\mathrm{s^{-1}}$) vs Radius (arcsec)",
                    yunit="",
                    legend=self.light_profile_pdf_list[0].__class__.__name__,
                    filename="image_1d_pdf",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

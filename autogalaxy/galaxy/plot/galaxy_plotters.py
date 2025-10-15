from __future__ import annotations
import math
from typing import TYPE_CHECKING, List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.visuals.two_d import Visuals2D
from autogalaxy.plot.mass_plotter import MassPlotter

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.galaxy.galaxy import Galaxy

if TYPE_CHECKING:
    from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePlotter
    from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePDFPlotter
from autogalaxy.profiles.plot.mass_profile_plotters import MassProfilePlotter
from autogalaxy.profiles.plot.mass_profile_plotters import MassProfilePDFPlotter

from autogalaxy import exc
from autogalaxy.util import error_util


class GalaxyPlotter(Plotter):
    def __init__(
        self,
        galaxy: Galaxy,
        grid: aa.type.Grid1D2DLike,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        """
        Plots the attributes of `Galaxy` objects using the matplotlib methods `plot()` and `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `MassProfile` and plotted via the visuals object.

        Parameters
        ----------
        galaxy
            The galaxy the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the galaxy's light and mass quantities that are plotted.
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

        if galaxy is not None:
            if galaxy.has(cls=LightProfileLinear):
                raise exc.raise_linear_light_profile_in_plot(
                    plotter_type=self.__class__.__name__,
                )

        super().__init__(
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
        )

        self.galaxy = galaxy
        self.grid = grid

        self._mass_plotter = MassPlotter(
            mass_obj=self.galaxy,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
        )

    def light_profile_plotter_from(
        self, light_profile: LightProfile, one_d_only: bool = False
    ) -> LightProfilePlotter:
        """
        Returns a `LightProfilePlotter` given an input light profile, which is typically used for plotting the
        individual light profiles of the plotter's `Galaxy` (e.g. in the function `figures_1d_decomposed`).

        Parameters
        ----------
        light_profile
            The light profile which is used to create the `LightProfilePlotter`.

        Returns
        -------
        LightProfilePlotter
            An object that plots the light profiles, often used for plotting attributes of the galaxy.
        """

        from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePlotter

        if not one_d_only:

            return LightProfilePlotter(
                light_profile=light_profile,
                grid=self.grid,
                mat_plot_2d=self.mat_plot_2d,
                visuals_2d=self.visuals_2d,
                mat_plot_1d=self.mat_plot_1d,
                visuals_1d=self.visuals_1d
                + Visuals1D().add_half_light_radius(light_obj=light_profile),
            )

        return LightProfilePlotter(
            light_profile=light_profile,
            grid=self.grid,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_1d
            + Visuals1D().add_half_light_radius(light_obj=light_profile),
        )

    def mass_profile_plotter_from(
        self, mass_profile: MassProfile, one_d_only: bool = False
    ) -> MassProfilePlotter:
        """
        Returns a `MassProfilePlotter` given an input mass profile, which is typically used for plotting the individual
        mass profiles of the plotter's `Galaxy` (e.g. in the function `figures_1d_decomposed`).

        Parameters
        ----------
        mass_profile
            The mass profile which is used to create the `MassProfilePlotter`.

        Returns
        -------
        MassProfilePlotter
            An object that plots the mass profiles, often used for plotting attributes of the galaxy.
        """

        if not one_d_only:
            return MassProfilePlotter(
                mass_profile=mass_profile,
                grid=self.grid,
                mat_plot_2d=self.mat_plot_2d,
                visuals_2d=self._mass_plotter.visuals_2d_with_critical_curves,
                mat_plot_1d=self.mat_plot_1d,
                visuals_1d=self.visuals_1d
                + Visuals1D().add_einstein_radius(
                    mass_obj=mass_profile, grid=self.grid
                ),
            )

        return MassProfilePlotter(
            mass_profile=mass_profile,
            grid=self.grid,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_1d
            + Visuals1D().add_einstein_radius(mass_obj=mass_profile, grid=self.grid),
        )

    @property
    def decomposed_light_profile_plotter_list(self):
        plotter_list = []

        for i, light_profile in enumerate(self.galaxy.cls_list_from(cls=LightProfile)):
            light_profile_plotter = self.light_profile_plotter_from(
                light_profile=light_profile, one_d_only=True
            )

            plotter_list.append(light_profile_plotter)

        return [self] + plotter_list

    @property
    def decomposed_mass_profile_plotter_list(self):
        plotter_list = []

        for i, mass_profile in enumerate(self.galaxy.cls_list_from(cls=MassProfile)):
            mass_profile_plotter = self.mass_profile_plotter_from(
                mass_profile=mass_profile, one_d_only=True
            )

            plotter_list.append(mass_profile_plotter)

        return [self] + plotter_list

    def figures_1d(
        self, image: bool = False, convergence: bool = False, potential: bool = False
    ):
        """
        Plots the individual attributes of the plotter's `Galaxy` object in 1D, which are computed via the plotter's
        grid object.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from each light profile of the galaxy. This is performed by aligning a 1D grid with the major-axis of
        each light profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        This means that the summed 1D profile of a galaxy's quantity is the sum of each individual component aligned
        with the major-axis.

        The API is such that every plottable attribute of the `Galaxy` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 1D plot (via `plot`) of the image.
        convergence
            Whether to make a 1D plot (via `plot`) of the convergence.
        potential
            Whether to make a 1D plot (via `plot`) of the potential.
        """
        if self.mat_plot_1d.yx_plot.plot_axis_type is None:
            plot_axis_type_override = "semilogy"
        else:
            plot_axis_type_override = None

        if image:
            image_1d = self.galaxy.image_1d_from(grid=self.grid)

            self.mat_plot_1d.plot_yx(
                y=image_1d,
                x=image_1d.grid_radial,
                visuals_1d=self.visuals_1d
                + Visuals1D().add_half_light_radius(self.galaxy),
                auto_labels=aplt.AutoLabels(
                    title="Image vs Radius",
                    ylabel="Image ",
                    xlabel="Radius",
                    legend=self.galaxy.__class__.__name__,
                    filename="image_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

        if convergence:
            convergence_1d = self.galaxy.convergence_1d_from(grid=self.grid)

            self.mat_plot_1d.plot_yx(
                y=convergence_1d,
                x=convergence_1d.grid_radial,
                visuals_1d=self.visuals_1d
                + Visuals1D().add_einstein_radius(mass_obj=self.galaxy, grid=self.grid),
                auto_labels=aplt.AutoLabels(
                    title="Convergence vs Radius",
                    ylabel="Convergence ",
                    xlabel="Radius",
                    legend=self.galaxy.__class__.__name__,
                    filename="convergence_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

        if potential:
            potential_1d = self.galaxy.potential_1d_from(grid=self.grid)

            self.mat_plot_1d.plot_yx(
                y=potential_1d,
                x=potential_1d.grid_radial,
                visuals_1d=self.visuals_1d,
                auto_labels=aplt.AutoLabels(
                    title="Potential vs Radius",
                    ylabel="Potential ",
                    xlabel="Radius",
                    legend=self.galaxy.__class__.__name__,
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
        """
        Plots the individual attributes of the plotter's `Galaxy` object in 1D, which are computed via the plotter's
        grid object.

        This function makes a decomposed plot shows the 1D plot of the attribute for every light or mass profile in
        the galaxy, as well as their combined 1D plot.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from each light profile of the galaxy. This is performed by aligning a 1D grid with the major-axis of
        each light profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        This means that the summed 1D profile of a galaxy's quantity is the sum of each individual component aligned
        with the major-axis.

        The API is such that every plottable attribute of the `Galaxy` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 1D plot (via `plot`) of the image.
        convergence
            Whether to make a 1D plot (via `imshow`) of the convergence.
        potential
            Whether to make a 1D plot (via `imshow`) of the potential.
        legend_labels
            Manually overrides the labels of the plot's legend.
        """

        if self.galaxy.has(cls=LightProfile):
            multi_plotter = aplt.MultiYX1DPlotter(
                plotter_list=self.decomposed_light_profile_plotter_list,
                legend_labels=legend_labels,
            )
            multi_plotter.plotter_list[0].mat_plot_1d.output = self.mat_plot_1d.output

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

        if self.galaxy.has(cls=MassProfile):
            multi_plotter = aplt.MultiYX1DPlotter(
                plotter_list=self.decomposed_mass_profile_plotter_list,
                legend_labels=legend_labels,
            )

            if convergence:
                change_filename = False

                if multi_plotter.plotter_list[0].mat_plot_1d.output.filename is None:
                    multi_plotter.plotter_list[0].set_filename(
                        filename="convergence_1d_decomposed"
                    )
                    change_filename = True

                multi_plotter.figure_1d(
                    func_name="figures_1d", figure_name="convergence"
                )

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
        title_suffix: str = "",
        filename_suffix: str = "",
    ):
        """
        Plots the individual attributes of the plotter's `Galaxy` object in 2D, which are computed via the plotter's 2D
        grid object.

        The API is such that every plottable attribute of the `Galaxy` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 2D plot (via `imshow`) of the image.
        convergence
            Whether to make a 2D plot (via `imshow`) of the convergence.
        potential
            Whether to make a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether to make a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether to make a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether to make a 2D plot (via `imshow`) of the magnification.
        """
        if image:
            self.mat_plot_2d.plot_array(
                array=self.galaxy.image_2d_from(grid=self.grid),
                visuals_2d=self.visuals_2d,
                auto_labels=aplt.AutoLabels(
                    title=f"Image{title_suffix}", filename=f"image_2d{filename_suffix}"
                ),
            )

        self._mass_plotter.figures_2d(
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
            title_suffix=title_suffix,
            filename_suffix=filename_suffix,
        )

    def subplot_of_light_profiles(self, image: bool = False):
        """
        Output a subplot of attributes of every individual light profile in 1D of the `Galaxy` object.

        For example, a 1D plot showing how the image (e.g. luminosity) of each component varies radially outwards.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from the light profile. This is performed by aligning a 1D grid with the  major-axis of the light
        profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        Parameters
        ----------
        image
            Whether to make a 1D subplot (via `plot`) of the image.
        """
        light_profile_plotters = [
            self.light_profile_plotter_from(light_profile)
            for light_profile in self.galaxy.cls_list_from(cls=LightProfile)
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
        """
        Output a subplot of attributes of every individual mass profile in 1D of the `Galaxy` object.

        For example, a 1D plot showing how the convergence of each component varies radially outwards.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from the light profile. This is performed by aligning a 1D grid with the  major-axis of the light
        profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        Parameters
        ----------
        image
            Whether to make a 1D subplot (via `plot`) of the image.
        convergence
            Whether to make a 1D plot (via `plot`) of the convergence.
        potential
            Whether to make a 1D plot (via `plot`) of the potential.
        """
        mass_profile_plotters = [
            self.mass_profile_plotter_from(mass_profile)
            for mass_profile in self.galaxy.cls_list_from(cls=MassProfile)
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
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
        sigma: Optional[float] = 3.0,
    ):
        """
        Plots the attributes of a list of `GalaxyProfile` objects using the matplotlib methods `plot()` and `imshow()`
        and many other matplotlib functions which customize the plot's appearance.

        Figures plotted by this object average over a list galaxy profiles to computed the average value of each
        attribute with errors, where the 1D regions within the errors are plotted as a shaded region to show the range
        of plausible models. Therefore, the input list of galaxies is expected to represent the probability density
        function of an inferred model-fit.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `GalaxyProfile` and plotted via the visuals object.

        Parameters
        ----------
        galaxy_profile_pdf_list
            The list of galaxy profiles whose mean and error values the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the galaxy profile quantities that are plotted.
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
            galaxy=None,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
        )

        self.galaxy_pdf_list = galaxy_pdf_list
        self.sigma = sigma
        self.low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    @property
    def light_profile_pdf_plotter_list(self) -> List[LightProfilePDFPlotter]:
        """
        Returns a list of `LightProfilePDFPlotter` objects from the list of galaxies in this object. These are
        typically used for plotting the individual average value plus errors of the light profiles of the
        plotter's `Galaxy` (e.g. in the function `figures_1d_decomposed`).

        Returns
        -------
        List[LightProfilePDFPlotter]
            An object that plots the average value and errors of a list of light profiles, often used for plotting
            attributes of the galaxy.
        """
        return [
            self.light_profile_pdf_plotter_from(index=index)
            for index in range(
                len(self.galaxy_pdf_list[0].cls_list_from(cls=LightProfile))
            )
        ]

    def light_profile_pdf_plotter_from(self, index) -> LightProfilePDFPlotter:
        """
        Returns the `LightProfilePDFPlotter` of a specific light profile in this plotter's list of galaxies. This is
        typically used for plotting the individual average value plus errors of a light profile in plotter's galaxy
        list (e.g. in the function `figures_1d_decomposed`).

        Returns
        -------
        LightProfilePDFPlotter
            An object that plots the average value and errors of a list of light profiles, often used for plotting
            attributes of the galaxy.
        """

        from autogalaxy.profiles.plot.light_profile_plotters import (
            LightProfilePDFPlotter,
        )

        light_profile_pdf_list = [
            galaxy.cls_list_from(cls=LightProfile)[index]
            for galaxy in self.galaxy_pdf_list
        ]

        return LightProfilePDFPlotter(
            light_profile_pdf_list=light_profile_pdf_list,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_1d,
        )

    @property
    def mass_profile_pdf_plotter_list(self) -> List[MassProfilePDFPlotter]:
        """
        Returns a list of `MassProfilePDFPlotter` objects from the list of galaxies in this object. These are
        typically used for plotting the individual average value plus errors of the mass profiles of the
        plotter's `Galaxy` (e.g. in the function `figures_1d_decomposed`).

        Returns
        -------
        List[MassProfilePDFPlotter]
            An object that plots the average value and errors of a list of mass profiles, often used for plotting
            attributes of the galaxy.
        """
        return [
            self.mass_profile_pdf_plotter_from(index=index)
            for index in range(
                len(self.galaxy_pdf_list[0].cls_list_from(cls=MassProfile))
            )
        ]

    def mass_profile_pdf_plotter_from(self, index) -> MassProfilePDFPlotter:
        """
        Returns the `MassProfilePDFPlotter` of a specific mass profile in this plotter's list of galaxies. This is
        typically used for plotting the individual average value plus errors of a mass profile in plotter's galaxy
        list (e.g. in the function `figures_1d_decomposed`).

        Returns
        -------
        MassProfilePDFPlotter
            An object that plots the average value and errors of a list of mass profiles, often used for plotting
            attributes of the galaxy.
        """
        mass_profile_pdf_list = [
            galaxy.cls_list_from(cls=MassProfile)[index]
            for galaxy in self.galaxy_pdf_list
        ]

        return MassProfilePDFPlotter(
            mass_profile_pdf_list=mass_profile_pdf_list,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_1d,
        )

    def figures_1d(
        self, image: bool = False, convergence: bool = False, potential: bool = False
    ):
        """
        Plots the individual attributes of the plotter's list of `Galaxy` object in 1D, which are computed via the
        plotter's grid object.

        This averages over a list galaxies to compute the average value of each attribute with errors, where the
        1D regions within the errors are plotted as a shaded region to show the range of plausible models. Therefore,
        the input list of galaxies is expected to represent the probability density function of an inferred model-fit.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from each light profile of the galaxy. This is performed by aligning a 1D grid with the major-axis of
        each light profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        This means that the summed 1D profile of a galaxy's quantity is the sum of each individual component aligned
        with the major-axis.

        The API is such that every plottable attribute of the `Galaxy` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

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
                galaxy.image_1d_from(grid=self.grid) for galaxy in self.galaxy_pdf_list
            ]

            min_index = min([image_1d.shape[0] for image_1d in image_1d_list])
            image_1d_list = [image_1d[0:min_index] for image_1d in image_1d_list]

            (
                median_image_1d,
                errors_image_1d,
            ) = error_util.profile_1d_median_and_error_region_via_quantile(
                profile_1d_list=image_1d_list, low_limit=self.low_limit
            )

            visuals_1d_via_light_obj_list = Visuals1D().add_half_light_radius_errors(
                light_obj_list=self.galaxy_pdf_list, low_limit=self.low_limit
            )
            visuals_1d_with_shaded_region = self.visuals_1d.__class__(
                shaded_region=errors_image_1d
            )

            visuals_1d = visuals_1d_via_light_obj_list + visuals_1d_with_shaded_region

            median_image_1d = aa.Array1D.no_mask(
                values=median_image_1d, pixel_scales=self.grid.pixel_scale
            )

            self.mat_plot_1d.plot_yx(
                y=median_image_1d,
                x=median_image_1d.grid_radial,
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
            convergence_1d_list = [
                galaxy.convergence_1d_from(grid=self.grid)
                for galaxy in self.galaxy_pdf_list
            ]

            min_index = min(
                [convergence_1d.shape[0] for convergence_1d in convergence_1d_list]
            )
            convergence_1d_list = [
                convergence_1d[0:min_index] for convergence_1d in convergence_1d_list
            ]

            (
                median_convergence_1d,
                errors_convergence_1d,
            ) = error_util.profile_1d_median_and_error_region_via_quantile(
                profile_1d_list=convergence_1d_list, low_limit=self.low_limit
            )

            visuals_1d_via_lensing_obj_list = Visuals1D().add_einstein_radius_errors(
                mass_obj_list=self.galaxy_pdf_list,
                grid=self.grid,
                low_limit=self.low_limit,
            )
            visuals_1d_with_shaded_region = self.visuals_1d.__class__(
                shaded_region=errors_convergence_1d
            )

            visuals_1d = visuals_1d_via_lensing_obj_list + visuals_1d_with_shaded_region

            median_convergence_1d = aa.Array1D.no_mask(
                values=median_convergence_1d, pixel_scales=self.grid.pixel_scale
            )

            self.mat_plot_1d.plot_yx(
                y=median_convergence_1d,
                x=median_convergence_1d.grid_radial,
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
            potential_1d_list = [
                galaxy.potential_1d_from(grid=self.grid)
                for galaxy in self.galaxy_pdf_list
            ]

            min_index = min(
                [potential_1d.shape[0] for potential_1d in potential_1d_list]
            )
            potential_1d_list = [
                potential_1d[0:min_index] for potential_1d in potential_1d_list
            ]

            (
                median_potential_1d,
                errors_potential_1d,
            ) = error_util.profile_1d_median_and_error_region_via_quantile(
                profile_1d_list=potential_1d_list, low_limit=self.low_limit
            )

            visuals_1d_via_lensing_obj_list = Visuals1D().add_einstein_radius_errors(
                mass_obj_list=self.galaxy_pdf_list,
                grid=self.grid,
                low_limit=self.low_limit,
            )
            visuals_1d_with_shaded_region = self.visuals_1d.__class__(
                shaded_region=errors_potential_1d
            )

            visuals_1d = visuals_1d_via_lensing_obj_list + visuals_1d_with_shaded_region

            median_potential_1d = aa.Array1D.no_mask(
                values=median_potential_1d, pixel_scales=self.grid.pixel_scale
            )

            self.mat_plot_1d.plot_yx(
                y=median_potential_1d,
                x=median_potential_1d.grid_radial,
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
        """
        Plots the individual attributes of the plotter's `Galaxy` object in 1D, which are computed via the plotter's
        grid object.

        This averages over a list galaxies to compute the average value of each attribute with errors, where the
        1D regions within the errors are plotted as a shaded region to show the range of plausible models. Therefore,
        the input list of galaxies is expected to represent the probability density function of an inferred model-fit.

        This function makes a decomposed plot showing the 1D plot of each attribute for every light or mass profile in
        the galaxy, as well as their combined 1D plot. By plotting the attribute of each profile on the same figure,
        one can see how much each profile contributes to the galaxy overall.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from each light profile of the galaxy. This is performed by aligning a 1D grid with the major-axis of
        each light profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        This means that the summed 1D profile of a galaxy's quantity is the sum of each individual component aligned
        with the major-axis.

        The API is such that every plottable attribute of the `Galaxy` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 1D plot (via `plot`) of the image.
        convergence
            Whether to make a 1D plot (via `imshow`) of the convergence.
        potential
            Whether to make a 1D plot (via `imshow`) of the potential.
        legend_labels
            Manually overrides the labels of the plot's legend.
        """
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

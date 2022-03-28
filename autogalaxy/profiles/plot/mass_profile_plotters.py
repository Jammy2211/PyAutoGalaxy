import math
from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.mass_plotter import MassPlotter
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.profiles.mass_profiles import MassProfile
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals1D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include1D
from autogalaxy.plot.mat_wrap.include import Include2D

from autogalaxy.util import error_util


class MassProfilePlotter(Plotter):
    def __init__(
        self,
        mass_profile: MassProfile,
        grid: aa.type.Grid2DLike,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        """
        Plots the attributes of `MassProfile` objects using the matplotlib methods `plot()` and `imshow()` and many 
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default, 
        the settings passed to every matplotlib function called are those specified in 
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to 
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be 
        extracted from the `MassProfile` and plotted via the visuals object, if the corresponding entry is `True` in 
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        mass_profile
            The mass profile the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the mass profile quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 2D plots.
        """
        super().__init__(
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
        )

        self.mass_profile = mass_profile
        self.grid = grid

        self._mass_plotter = MassPlotter(
            mass_obj=self.mass_profile,
            grid=self.grid,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

        self.figures_2d = self._mass_plotter.figures_2d

    def get_visuals_1d(self) -> Visuals1D:
        return self.get_1d.via_mass_obj_from(mass_obj=self.mass_profile, grid=self.grid)

    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_mass_obj_from(mass_obj=self.mass_profile, grid=self.grid)

    def figures_1d(self, convergence: bool = False, potential: bool = False):
        """
        Plots the individual attributes of the plotter's `MassProfile` object in 1D, which are computed via the 
        plotter's grid object. 

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is 
        computed from the mass profile. This is performed by aligning a 1D grid with the  major-axis of the mass 
        profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        The API is such that every plottable attribute of the  `MassProfile` object is an input parameter of type
        bool of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        convergence
            Whether or not to make a 1D plot (via `imshow`) of the convergence.
        potential
            Whether or not to make a 1D plot (via `imshow`) of the potential.
        """
        if self.mat_plot_1d.yx_plot.plot_axis_type is None:
            plot_axis_type_override = "semilogy"
        else:
            plot_axis_type_override = None

        if convergence:

            convergence_1d = self.mass_profile.convergence_1d_from(grid=self.grid)

            self.mat_plot_1d.plot_yx(
                y=convergence_1d,
                x=convergence_1d.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=aplt.AutoLabels(
                    title="Convergence vs Radius",
                    ylabel="Convergence ",
                    xlabel="Radius",
                    legend=self.mass_profile.__class__.__name__,
                    filename="convergence_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

        if potential:

            potential_1d = self.mass_profile.potential_1d_from(grid=self.grid)

            self.mat_plot_1d.plot_yx(
                y=potential_1d,
                x=potential_1d.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=aplt.AutoLabels(
                    title="Potential vs Radius",
                    ylabel="Potential ",
                    xlabel="Radius",
                    legend=self.mass_profile.__class__.__name__,
                    filename="potential_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )


class MassProfilePDFPlotter(MassProfilePlotter):
    def __init__(
        self,
        mass_profile_pdf_list: List[MassProfile],
        grid: aa.Grid2D,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
        sigma: Optional[float] = 3.0,
    ):
        """
        Plots the attributes of a list of `MassProfile` objects using the matplotlib methods `plot()` and `imshow()` 
        and many other matplotlib functions which customize the plot's appearance.

        Figures plotted by this object average over a list mass profiles to computed the average value of each attribute
        with errors, where the 1D regions within the errors are plotted as a shaded region to show the range of
        plausible models. Therefore, the input list of galaxies is expected to represent the probability density
        function of an inferred model-fit.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default, 
        the settings passed to every matplotlib function called are those specified in 
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to 
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be 
        extracted from the `MassProfile` and plotted via the visuals object, if the corresponding entry is `True` in 
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        mass_profile_pdf_list
            The list of mass profiles whose mean and error values the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the mass profile quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 2D plots.
        sigma
            The confidence interval in terms of a sigma value at which the errors are computed (e.g. a value of 
            sigma=3.0 uses confidence intevals at ~0.01 and 0.99 the PDF).            
        """
        super().__init__(
            mass_profile=None,
            grid=grid,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

        self.mass_profile_pdf_list = mass_profile_pdf_list
        self.sigma = sigma
        self.low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    def figures_1d(self, convergence=False, potential=False):
        """
        Plots the individual attributes of the plotter's list of ` MassProfile` object in 1D, which are computed via
        the plotter's grid object.

        This averages over a list mass profiles to compute the average value of each attribute with errors, where the
        1D regions within the errors are plotted as a shaded region to show the range of plausible models. Therefore,
        the input list of galaxies is expected to represent the probability density function of an inferred model-fit.

        If the plotter has a 1D grid object this is used to evaluate each quantity. If it has a 2D grid, a 1D grid is
        computed from each mass profile. This is performed by aligning a 1D grid with the major-axis of
        each mass profile in projection, uniformly computing 1D values based on the 2D grid's size and pixel-scale.

        The API is such that every plottable attribute of the `MassProfile` object is an input parameter of type bool
        of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        convergence
            Whether or not to make a 1D plot (via `imshow`) of the convergence.
        potential
            Whether or not to make a 1D plot (via `imshow`) of the potential.
        """
        if self.mat_plot_1d.yx_plot.plot_axis_type is None:
            plot_axis_type_override = "semilogy"
        else:
            plot_axis_type_override = None

        if convergence:

            convergence_1d_list = [
                mass_profile.convergence_1d_from(grid=self.grid)
                for mass_profile in self.mass_profile_pdf_list
            ]

            min_index = min(
                [convergence_1d.shape[0] for convergence_1d in convergence_1d_list]
            )
            convergence_1d_list = [
                convergence_1d[0:min_index] for convergence_1d in convergence_1d_list
            ]

            median_convergence_1d, errors_convergence_1d = error_util.profile_1d_median_and_error_region_via_quantile(
                profile_1d_list=convergence_1d_list, low_limit=self.low_limit
            )

            visuals_1d_via_lensing_obj_list = self.get_1d.via_mass_obj_list_from(
                mass_obj_list=self.mass_profile_pdf_list,
                grid=self.grid,
                low_limit=self.low_limit,
            )
            visuals_1d_with_shaded_region = self.visuals_1d.__class__(
                shaded_region=errors_convergence_1d
            )

            visuals_1d = visuals_1d_via_lensing_obj_list + visuals_1d_with_shaded_region

            self.mat_plot_1d.plot_yx(
                y=median_convergence_1d,
                x=convergence_1d_list[0].grid_radial,
                visuals_1d=visuals_1d,
                auto_labels=aplt.AutoLabels(
                    title="Convergence vs Radius",
                    ylabel="Convergence ",
                    xlabel="Radius",
                    legend=self.mass_profile_pdf_list[0].__class__.__name__,
                    filename="convergence_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

        if potential:

            potential_1d_list = [
                mass_profile.potential_1d_from(grid=self.grid)
                for mass_profile in self.mass_profile_pdf_list
            ]

            min_index = min(
                [potential_1d.shape[0] for potential_1d in potential_1d_list]
            )
            potential_1d_list = [
                potential_1d[0:min_index] for potential_1d in potential_1d_list
            ]

            median_potential_1d, errors_potential_1d = error_util.profile_1d_median_and_error_region_via_quantile(
                profile_1d_list=potential_1d_list, low_limit=self.low_limit
            )

            visuals_1d_via_lensing_obj_list = self.get_1d.via_mass_obj_list_from(
                mass_obj_list=self.mass_profile_pdf_list,
                grid=self.grid,
                low_limit=self.low_limit,
            )
            visuals_1d_with_shaded_region = self.visuals_1d.__class__(
                shaded_region=errors_potential_1d
            )

            visuals_1d = visuals_1d_via_lensing_obj_list + visuals_1d_with_shaded_region

            self.mat_plot_1d.plot_yx(
                y=median_potential_1d,
                x=potential_1d_list[0].grid_radial,
                visuals_1d=visuals_1d,
                auto_labels=aplt.AutoLabels(
                    title="Potential vs Radius",
                    ylabel="Potential ",
                    xlabel="Radius",
                    legend=self.mass_profile_pdf_list[0].__class__.__name__,
                    filename="potential_1d",
                ),
                plot_axis_type_override=plot_axis_type_override,
            )

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.basis import Basis
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.visuals.two_d import Visuals2D

from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePlotter

from autogalaxy import exc


class BasisPlotter(Plotter):
    def __init__(
        self,
        basis: Basis,
        grid: aa.type.Grid1D2DLike,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        """
        Plots the attributes of `Basis` objects using the matplotlib methods `plot()` and `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `LightProfile` and plotted via the visuals object.

        Parameters
        ----------
        basis
            The basis the plotter plots.
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

        for light_profile in basis.light_profile_list:
            if isinstance(light_profile, LightProfileLinear):
                raise exc.raise_linear_light_profile_in_plot(
                    plotter_type=self.__class__.__name__,
                )

        self.basis = basis
        self.grid = grid

        super().__init__(
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
        )

    def light_profile_plotter_from(
        self,
        light_profile: LightProfile,
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

        return LightProfilePlotter(
            light_profile=light_profile,
            grid=self.grid,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.get_1d.via_light_obj_from(light_obj=light_profile),
        )

    def subplot_image(self):
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

        self.open_subplot_figure(number_subplots=len(self.basis.light_profile_list))

        for light_profile in self.basis.light_profile_list:
            self.mat_plot_2d.plot_array(
                array=light_profile.image_2d_from(grid=self.grid),
                visuals_2d=self.visuals_2d,
                auto_labels=aplt.AutoLabels(title=light_profile.coefficient_tag),
            )

        self.mat_plot_2d.output.subplot_to_figure(auto_filename=f"subplot_basis_image")

        self.close_subplot_figure()

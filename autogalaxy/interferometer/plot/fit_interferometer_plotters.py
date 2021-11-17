import autoarray.plot as aplt

from autoarray.fit.plot.fit_interferometer_plotters import FitInterferometerPlotterMeta

from autogalaxy.plane.plane import Plane
from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals1D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include1D
from autogalaxy.plot.mat_wrap.include import Include2D

from autogalaxy.plane.plot.plane_plotters import PlanePlotter


class FitInterferometerPlotter(Plotter):
    def __init__(
        self,
        fit: FitInterferometer,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        """
        Plots the attributes of `FitInterferometer` objects using the matplotlib method `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2d` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `FitInterferometer` and plotted via the visuals object, if the corresponding entry is `True` in
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an interferometer dataset the plotter plots.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `FitInterferometer` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `FitInterferometer` are extracted and plotted as visuals for 2D plots.
        """
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.fit = fit

        self._fit_interferometer_meta_plotter = FitInterferometerPlotterMeta(
            fit=self.fit,
            get_visuals_2d_real_space=self.get_visuals_2d_real_space,
            mat_plot_1d=self.mat_plot_1d,
            include_1d=self.include_1d,
            visuals_1d=self.visuals_1d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

        self.figures_2d = self._fit_interferometer_meta_plotter.figures_2d
        self.subplot = self._fit_interferometer_meta_plotter.subplot
        self.subplot_fit_interferometer = (
            self._fit_interferometer_meta_plotter.subplot_fit_interferometer
        )
        self.subplot_fit_dirty_images = (
            self._fit_interferometer_meta_plotter.subplot_fit_dirty_images
        )

    def get_visuals_2d_real_space(self) -> Visuals2D:
        return self.get_2d.via_mask_from(mask=self.fit.interferometer.real_space_mask)

    @property
    def plane(self) -> Plane:
        return self.fit.plane

    def plane_plotter_from(self, plane: Plane) -> PlanePlotter:
        """
        Returns an `PlanePlotter` corresponding to an input `Plane` of the fit.

        Returns
        -------
        plane
            The plane used to make the `PlanePlotter`.
        """
        return PlanePlotter(
            plane=plane,
            grid=self.fit.interferometer.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d_real_space(),
            include_2d=self.include_2d,
        )

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        """
        Returns an `InversionPlotter` corresponding to the `Inversion` of the fit.

        Returns
        -------
        InversionPlotter
            An object that plots inversions which is used for plotting attributes of the inversion.
        """
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d_real_space(),
            include_2d=self.include_2d,
        )

    def subplot_fit_real_space(self):
        """
        Standard subplot of the real-space attributes of the plotter's `FitInterferometer` object.

        Depending on whether `LightProfile`'s or an `Inversion` are used to represent galaxies in the `Plane`, different
        methods are called to create these real-space images.
        """
        if self.fit.inversion is None:

            plane_plotter = self.plane_plotter_from(plane=self.plane)

            plane_plotter.subplot(
                image=True, plane_image=True, auto_filename="subplot_fit_real_space"
            )

        elif self.fit.inversion is not None:

            self.open_subplot_figure(number_subplots=6)

            mapper_index = 0

            self.inversion_plotter.figures_2d_of_mapper(
                mapper_index=mapper_index, reconstructed_image=True
            )
            self.inversion_plotter.figures_2d_of_mapper(
                mapper_index=mapper_index, reconstruction=True
            )

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_fit_real_space"
            )

            self.close_subplot_figure()

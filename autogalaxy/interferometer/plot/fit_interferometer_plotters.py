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

    def plane_plotter_from(self, plane) -> PlanePlotter:
        return PlanePlotter(
            plane=plane,
            grid=self.fit.interferometer.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d_real_space(),
            include_2d=self.include_2d,
        )

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d_real_space(),
            include_2d=self.include_2d,
        )

    def subplot_fit_real_space(self):

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

import autoarray.plot as aplt
from autoarray.fit.plot import fit_interferometer_plotters

from autogalaxy.plane.plane import Plane
from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.plot.mat_wrap.lensing_mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.lensing_mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.lensing_visuals import Visuals1D
from autogalaxy.plot.mat_wrap.lensing_visuals import Visuals2D
from autogalaxy.plot.mat_wrap.lensing_include import Include1D
from autogalaxy.plot.mat_wrap.lensing_include import Include2D

from autogalaxy.plane.plot.plane_plotters import PlanePlotter


class FitInterferometerPlotter(
    fit_interferometer_plotters.AbstractFitInterferometerPlotter
):
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
            fit=fit,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    @property
    def plane(self) -> Plane:
        return self.fit.plane

    @property
    def visuals_with_include_2d(self) -> Visuals2D:
        visuals_2d = super(FitInterferometerPlotter, self).visuals_with_include_2d
        return visuals_2d + visuals_2d.__class__()

    def plane_plotter_from(self, plane) -> PlanePlotter:
        return PlanePlotter(
            plane=plane,
            grid=self.fit.interferometer.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_with_include_2d,
            include_2d=self.include_2d,
        )

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_with_include_2d,
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

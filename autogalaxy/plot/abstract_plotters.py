from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

from autoarray.plot.abstract_plotters import AbstractPlotter

from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.visuals.two_d import Visuals2D


class Plotter(AbstractPlotter):

    def __init__(
        self,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
        )

        self.visuals_1d = visuals_1d or Visuals1D()
        self.mat_plot_1d = mat_plot_1d or MatPlot1D()

        self.visuals_2d = visuals_2d or Visuals2D()
        self.mat_plot_2d = mat_plot_2d or MatPlot2D()

    def _plot_array(self, array, visuals_2d, auto_labels):
        """Bridge: delegates to mat_plot_2d.plot_array(), ready for future migration
        to the direct-matplotlib plot_array() from autoarray."""
        self.mat_plot_2d.plot_array(
            array=array,
            visuals_2d=visuals_2d,
            auto_labels=auto_labels,
        )

    def _plot_grid(self, grid, visuals_2d, auto_labels):
        """Bridge: delegates to mat_plot_2d.plot_grid(), ready for future migration
        to the direct-matplotlib plot_grid() from autoarray."""
        self.mat_plot_2d.plot_grid(
            grid=grid,
            visuals_2d=visuals_2d,
            auto_labels=auto_labels,
        )

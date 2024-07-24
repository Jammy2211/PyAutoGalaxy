import matplotlib.pyplot as plt
import numpy as np
from typing import List

from autoconf import conf

import autoarray as aa
from autoarray import plot as aplt

from autoarray.plot.auto_labels import AutoLabels

from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.include.one_d import Include1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.two_d import Visuals2D
from autogalaxy.plot.include.two_d import Include2D


class FitEllipsePlotter(Plotter):
    def __init__(
        self,
        fit_list: List[FitEllipse],
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

        self.fit_list = fit_list

    def get_visuals_1d(self) -> Visuals1D:
        return self.visuals_1d

    def figures_2d(
        self,
        data: bool = False,
        suffix: str = "",
    ):
        """
        Plots the individual attributes of the plotter's `FitEllipse` object in 1D.

        The API is such that every plottable attribute of the `FitEllipse` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        data
            Whether to make a 1D plot (via `imshow`) of the image data.
        """

        if data:

            # # Used for 1D plot
            #
            # # colors = [ax.cmap(ax.norm(level)) for level in levels][::-1]


            self.mat_plot_2d.contour = aplt.Contour(
                manual_levels=[float(np.mean(self.fit_list[0].data_interp))]
            )

            x = self.fit_list[0].ellipse.x_from_major_axis_from(pixel_scale=self.fit_list[0].data.pixel_scale)
            y = self.fit_list[0].ellipse.y_from_major_axis_from(pixel_scale=self.fit_list[0].data.pixel_scale)

            visuals_2d = self.visuals_2d + Visuals2D(
                lines=aa.Grid2DIrregular.from_yx_1d(y=y, x=x)
            )

            self.mat_plot_2d.plot_array(
                array=self.fit_list[0].data,
                visuals_2d=visuals_2d,
                auto_labels = aplt.AutoLabels(
                    title=f"Ellipse Fit",
                    filename=f"ellipse_fit",
                ),



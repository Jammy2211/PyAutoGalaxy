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

    def get_visuals_2d(self):
        return self.get_2d.via_mask_from(mask=self.fit_list[0].dataset.mask)

    def figures_2d(
        self,
        data: bool = False,
        disable_data_contours: bool = False,
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
        disable_data_contours
            If `True`, the data is plotted without the black data contours over the top (but the white contours
            showing the ellipses are still plotted).
        """

        filename_tag = ""

        if data:
            self.mat_plot_2d.contour = aplt.Contour(
                manual_levels=np.sort(
                    [float(np.mean(fit.data_interp)) for fit in self.fit_list]
                )
            )

            if disable_data_contours:
                contour_original = self.mat_plot_2d.contour
                self.mat_plot_2d.contour = False
                filename_tag = "_no_data_contours"

            ellipse_list = []

            for fit in self.fit_list:
                points = fit.points_from_major_axis_from(flip_y=True)

                x = points[:, 1]
                y = points[:, 0]

                ellipse_list.append(aa.Grid2DIrregular.from_yx_1d(y=y, x=x))

            visuals_2d = self.get_visuals_2d() + Visuals2D(lines=ellipse_list)

            self.mat_plot_2d.plot_array(
                array=self.fit_list[0].data,
                visuals_2d=visuals_2d,
                auto_labels=aplt.AutoLabels(
                    title=f"Ellipse Fit",
                    filename=f"ellipse_fit{filename_tag}",
                ),
            )

            if disable_data_contours:
                self.mat_plot_2d.contour = contour_original

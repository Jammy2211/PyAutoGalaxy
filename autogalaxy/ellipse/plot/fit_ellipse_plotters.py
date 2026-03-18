import numpy as np
from typing import List

import math
from typing import List, Optional

import autoarray as aa
from autoarray import plot as aplt

from autogalaxy.ellipse.plot import fit_ellipse_plot_util

from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D

from autogalaxy.util import error_util


class FitEllipsePlotter(Plotter):
    def __init__(
        self,
        fit_list: List[FitEllipse],
        mat_plot_1d: MatPlot1D = None,
        mat_plot_2d: MatPlot2D = None,
        positions=None,
        lines=None,
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        self.fit_list = fit_list
        self.positions = positions
        self.lines = lines

    def figures_2d(
        self,
        data: bool = False,
        disable_data_contours: bool = False,
        ellipse_residuals: bool = False,
        for_subplot: bool = False,
        suffix: str = "",
    ):
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
                points = fit.points_from_major_axis_from()

                x = points[:, 1]
                y = points[:, 0] * -1.0  # flip for plot

                ellipse_list.append(aa.Grid2DIrregular.from_yx_1d(y=y, x=x))

            # Convert ellipse_list to list of (N,2) numpy arrays
            lines = [np.array(e.array) for e in ellipse_list if e is not None]
            positions = lines  # same data used for both

            self._plot_array(
                array=self.fit_list[0].data,
                auto_labels=aplt.AutoLabels(
                    title=f"Ellipse Fit",
                    filename=f"ellipse_fit{filename_tag}",
                ),
                lines=lines or None,
                positions=positions or None,
            )

            if disable_data_contours:
                self.mat_plot_2d.contour = contour_original

        if ellipse_residuals:

            try:
                colors = self.mat_plot_2d.grid_plot.config_dict["c"]
            except KeyError:
                colors = "k"

            fit_ellipse_plot_util.plot_ellipse_residuals(
                array=self.fit_list[0].dataset.data.native,
                fit_list=self.fit_list,
                colors=colors,
                output=self.mat_plot_2d.output,
                for_subplot=for_subplot,
            )

    def subplot_fit_ellipse(self, disable_data_contours: bool = False):
        self.open_subplot_figure(number_subplots=2)

        self.mat_plot_2d.use_log10 = True
        self.figures_2d(data=True, disable_data_contours=disable_data_contours)
        self.figures_2d(ellipse_residuals=True, for_subplot=True)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_fit_ellipse")
        self.close_subplot_figure()


class FitEllipsePDFPlotter(Plotter):
    def __init__(
        self,
        fit_pdf_list: List[FitEllipse],
        mat_plot_1d: MatPlot1D = None,
        mat_plot_2d: MatPlot2D = None,
        sigma: Optional[float] = 3.0,
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d or MatPlot1D(),
            mat_plot_2d=mat_plot_2d or MatPlot2D(),
        )

        self.fit_pdf_list = fit_pdf_list
        self.sigma = sigma
        self.low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    def subplot_ellipse_errors(self):
        contour_original = self.mat_plot_2d.contour
        self.mat_plot_2d.contour = False

        ellipse_centre_list = []
        fit_ellipse_list = [[] for _ in range(len(self.fit_pdf_list[0]))]

        for fit_list in self.fit_pdf_list:

            ellipse_centre_list.append(fit_list[0].ellipse.centre)

            for i, fit in enumerate(fit_list):

                points = fit.points_from_major_axis_from()

                x = points[:, 1]
                y = points[:, 0] * -1.0  # flip for plot

                fit_ellipse_list[i].append(aa.Grid2DIrregular.from_yx_1d(y=y, x=x))

        self.open_subplot_figure(number_subplots=len(fit_ellipse_list))

        for i in range(len(fit_ellipse_list)):

            median_ellipse, [lower_ellipse, upper_ellipse] = (
                error_util.ellipse_median_and_error_region_in_polar(
                    fit_ellipse_list[i],
                    low_limit=self.low_limit,
                    center=ellipse_centre_list[i],
                )
            )

            # Unpack points
            y_lower, x_lower = lower_ellipse[:, 0], lower_ellipse[:, 1]
            y_upper, x_upper = upper_ellipse[:, 0], upper_ellipse[:, 1]

            # Close the contours
            x_fill = np.concatenate([x_lower, x_upper[::-1]])
            y_fill = np.concatenate([y_lower, y_upper[::-1]])

            # Convert median_ellipse to a numpy line
            try:
                median_arr = np.array(
                    median_ellipse.array
                    if hasattr(median_ellipse, "array")
                    else median_ellipse
                )
                lines = [median_arr] if median_arr.ndim == 2 else None
            except Exception:
                lines = None

            self._plot_array(
                array=self.fit_pdf_list[0][0].data,
                auto_labels=aplt.AutoLabels(
                    title=f"Ellipse Fit",
                    filename=f"subplot_ellipse_errors",
                ),
                lines=lines,
            )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_ellipse_errors"
        )
        self.close_subplot_figure()

        self.mat_plot_2d.contour = contour_original

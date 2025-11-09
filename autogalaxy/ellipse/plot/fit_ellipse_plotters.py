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
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.two_d import Visuals2D

from autogalaxy.util import error_util


class FitEllipsePlotter(Plotter):
    def __init__(
        self,
        fit_list: List[FitEllipse],
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

        self.fit_list = fit_list

    def figures_2d(
        self,
        data: bool = False,
        disable_data_contours: bool = False,
        ellipse_residuals: bool = False,
        for_subplot: bool = False,
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
                points = fit.points_from_major_axis_from()

                x = points[:, 1]
                y = points[:, 0] * -1.0  # flip for plot

                ellipse_list.append(aa.Grid2DIrregular.from_yx_1d(y=y, x=x))

            visuals_2d = self.visuals_2d + Visuals2D(
                positions=ellipse_list, lines=ellipse_list
            )

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
        """
        Standard subplot of the attributes of the plotter's `FitEllipse` object.
        """

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
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        sigma: Optional[float] = 3.0,
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
        )

        self.fit_pdf_list = fit_pdf_list
        self.sigma = sigma
        self.low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    def get_visuals_1d(self) -> Visuals1D:
        return self.visuals_1d

    def get_visuals_2d(self):
        return self.get_2d.via_mask_from(mask=self.fit_pdf_list[0][0].dataset.mask)

    def subplot_ellipse_errors(self):
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

                print(i, len(x))

            print()

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

            visuals_2d = self.get_visuals_2d() + Visuals2D(
                lines=median_ellipse, fill_region=[y_fill, x_fill]
            )

            self.mat_plot_2d.plot_array(
                array=self.fit_pdf_list[0][0].data,
                visuals_2d=visuals_2d,
                auto_labels=aplt.AutoLabels(
                    title=f"Ellipse Fit",
                    filename=f"subplot_ellipse_errors",
                ),
            )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_ellipse_errors"
        )
        self.close_subplot_figure()

        self.mat_plot_2d.contour = contour_original

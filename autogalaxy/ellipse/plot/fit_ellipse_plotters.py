import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Optional

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap

import autoarray as aa
from autoarray import plot as aplt

from autogalaxy.ellipse.plot import fit_ellipse_plot_util
from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.plot.abstract_plotters import Plotter, _save_subplot
from autogalaxy.util import error_util


class FitEllipsePlotter(Plotter):
    def __init__(
        self,
        fit_list: List[FitEllipse],
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        positions=None,
        lines=None,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

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
        ax=None,
    ):
        if data:
            ellipse_list = []
            for fit in self.fit_list:
                points = fit.points_from_major_axis_from()
                x = points[:, 1]
                y = points[:, 0] * -1.0
                ellipse_list.append(aa.Grid2DIrregular.from_yx_1d(y=y, x=x))

            lines = [np.array(e.array) for e in ellipse_list if e is not None]
            positions = lines

            self._plot_array(
                array=self.fit_list[0].data,
                auto_filename=f"ellipse_fit{suffix}",
                title="Ellipse Fit",
                lines=lines or None,
                positions=positions or None,
                ax=ax,
            )

        if ellipse_residuals:
            try:
                colors = "k"
            except KeyError:
                colors = "k"

            fit_ellipse_plot_util.plot_ellipse_residuals(
                array=self.fit_list[0].dataset.data.native,
                fit_list=self.fit_list,
                colors=colors,
                output=self.output,
                for_subplot=for_subplot,
            )

    def subplot_fit_ellipse(self, disable_data_contours: bool = False):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        self.figures_2d(data=True, disable_data_contours=disable_data_contours, ax=axes[0])
        self.figures_2d(ellipse_residuals=True, for_subplot=True, ax=axes[1])

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_fit_ellipse")


class FitEllipsePDFPlotter(Plotter):
    def __init__(
        self,
        fit_pdf_list: List[FitEllipse],
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        sigma: Optional[float] = 3.0,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.fit_pdf_list = fit_pdf_list
        self.sigma = sigma
        self.low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    def subplot_ellipse_errors(self):
        ellipse_centre_list = []
        fit_ellipse_list = [[] for _ in range(len(self.fit_pdf_list[0]))]

        for fit_list in self.fit_pdf_list:
            ellipse_centre_list.append(fit_list[0].ellipse.centre)
            for i, fit in enumerate(fit_list):
                points = fit.points_from_major_axis_from()
                x = points[:, 1]
                y = points[:, 0] * -1.0
                fit_ellipse_list[i].append(aa.Grid2DIrregular.from_yx_1d(y=y, x=x))

        n = len(fit_ellipse_list)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
        axes_flat = [axes] if n == 1 else list(axes.flatten())

        for i in range(n):
            median_ellipse, [lower_ellipse, upper_ellipse] = (
                error_util.ellipse_median_and_error_region_in_polar(
                    fit_ellipse_list[i],
                    low_limit=self.low_limit,
                    center=ellipse_centre_list[i],
                )
            )

            try:
                median_arr = np.array(
                    median_ellipse.array if hasattr(median_ellipse, "array") else median_ellipse
                )
                lines = [median_arr] if median_arr.ndim == 2 else None
            except Exception:
                lines = None

            self._plot_array(
                array=self.fit_pdf_list[0][0].data,
                auto_filename="subplot_ellipse_errors",
                title="Ellipse Fit",
                lines=lines,
                ax=axes_flat[i],
            )

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_ellipse_errors")

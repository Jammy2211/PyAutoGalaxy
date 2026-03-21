import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Optional

import autoarray as aa
from autoarray import plot as aplt

from autogalaxy.ellipse.plot import fit_ellipse_plot_util
from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.plot.plot_utils import plot_array, _save_subplot
from autogalaxy.util import error_util


def _plot_data(
    fit_list: List[FitEllipse],
    output_path=None,
    output_filename="ellipse_fit",
    output_format="png",
    colormap="default",
    use_log10=False,
    disable_data_contours: bool = False,
    suffix: str = "",
    ax=None,
):
    ellipse_list = []
    for fit in fit_list:
        points = fit.points_from_major_axis_from()
        x = points[:, 1]
        y = points[:, 0] * -1.0
        ellipse_list.append(aa.Grid2DIrregular.from_yx_1d(y=y, x=x))

    lines = [np.array(e.array) for e in ellipse_list if e is not None]
    positions = lines

    plot_array(
        array=fit_list[0].data,
        title="Ellipse Fit",
        output_path=output_path,
        output_filename=f"{output_filename}{suffix}",
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        lines=lines or None,
        positions=positions or None,
        ax=ax,
    )


def _plot_ellipse_residuals(
    fit_list: List[FitEllipse],
    output_path=None,
    output_format="png",
    for_subplot: bool = False,
    suffix: str = "",
    ax=None,
):
    output = aplt.Output(path=output_path, format=output_format) if output_path else aplt.Output()

    fit_ellipse_plot_util.plot_ellipse_residuals(
        array=fit_list[0].dataset.data.native,
        fit_list=fit_list,
        colors="k",
        output=output,
        for_subplot=for_subplot,
    )


def subplot_fit_ellipse(
    fit_list: List[FitEllipse],
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    disable_data_contours: bool = False,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    _plot_data(
        fit_list=fit_list,
        colormap=colormap,
        use_log10=use_log10,
        disable_data_contours=disable_data_contours,
        ax=axes[0],
    )
    _plot_ellipse_residuals(fit_list=fit_list, for_subplot=True, ax=axes[1])

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_fit_ellipse", output_format)


def subplot_ellipse_errors(
    fit_pdf_list: List[List[FitEllipse]],
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    sigma: Optional[float] = 3.0,
):
    low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    ellipse_centre_list = []
    fit_ellipse_list = [[] for _ in range(len(fit_pdf_list[0]))]

    for fit_list in fit_pdf_list:
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
                low_limit=low_limit,
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

        plot_array(
            array=fit_pdf_list[0][0].data,
            title="Ellipse Fit",
            colormap=colormap,
            use_log10=use_log10,
            lines=lines,
            ax=axes_flat[i],
        )

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_ellipse_errors", output_format)

from typing import List

import autoarray as aa
import autoarray.plot as aplt

from autoarray.fit.plot.fit_interferometer_plotters import FitInterferometerPlotterMeta

from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.galaxy.plot import galaxies_plots


def _make_meta(fit, output_path, output_format, colormap, use_log10, residuals_symmetric_cmap):
    from autogalaxy.plot.plot_utils import _resolve_format
    output_format = _resolve_format(output_format)
    output = aplt.Output(path=output_path, format=output_format) if output_path else aplt.Output()
    cmap = aplt.Cmap(cmap=colormap) if colormap != "default" else aplt.Cmap()
    return FitInterferometerPlotterMeta(
        fit=fit,
        output=output,
        cmap=cmap,
        use_log10=use_log10,
        residuals_symmetric_cmap=residuals_symmetric_cmap,
    )


def subplot_fit(
    fit: FitInterferometer,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    residuals_symmetric_cmap: bool = True,
):
    _make_meta(fit, output_path, output_format, colormap, use_log10, residuals_symmetric_cmap).subplot_fit()


def subplot_fit_dirty_images(
    fit: FitInterferometer,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    residuals_symmetric_cmap: bool = True,
):
    _make_meta(fit, output_path, output_format, colormap, use_log10, residuals_symmetric_cmap).subplot_fit_dirty_images()


def subplot_fit_real_space(
    fit: FitInterferometer,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
):
    galaxy_list = fit.galaxies_linear_light_profiles_to_light_profiles

    if not galaxy_list.has(cls=aa.Pixelization):
        galaxies_plots.subplot_galaxies(
            galaxies=galaxy_list,
            grid=fit.grids.lp,
            output_path=output_path,
            output_format=output_format,
            colormap=colormap,
            use_log10=use_log10,
            auto_filename="subplot_fit_real_space",
        )
    else:
        output = aplt.Output(path=output_path, format=output_format) if output_path else aplt.Output()
        inversion_plotter = aplt.InversionPlotter(inversion=fit.inversion, output=output)
        inversion_plotter.figures_2d_of_pixelization(
            pixelization_index=0, reconstructed_operated_data=True
        )
        inversion_plotter.figures_2d_of_pixelization(
            pixelization_index=0, reconstruction=True
        )

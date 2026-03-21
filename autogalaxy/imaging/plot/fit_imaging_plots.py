import matplotlib.pyplot as plt
from typing import List, Optional

import autoarray as aa
import autoarray.plot as aplt

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.plot.plot_utils import _to_positions, _save_subplot, plot_array


def _make_meta(fit, output_path, output_format, colormap, use_log10, positions, residuals_symmetric_cmap):
    from autogalaxy.plot.plot_utils import _resolve_format
    output_format = _resolve_format(output_format)
    output = aplt.Output(path=output_path, format=output_format) if output_path else aplt.Output()
    cmap = aplt.Cmap(cmap=colormap) if colormap != "default" else aplt.Cmap()
    return FitImagingPlotterMeta(
        fit=fit,
        output=output,
        cmap=cmap,
        use_log10=use_log10,
        positions=_to_positions(positions),
        residuals_symmetric_cmap=residuals_symmetric_cmap,
    )


def subplot_fit(
    fit: FitImaging,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    residuals_symmetric_cmap: bool = True,
):
    _make_meta(fit, output_path, output_format, colormap, use_log10, positions, residuals_symmetric_cmap).subplot_fit()


def subplot_of_galaxy(
    fit: FitImaging,
    galaxy_index: int,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    residuals_symmetric_cmap: bool = True,
):
    galaxies = fit.galaxies_linear_light_profiles_to_light_profiles
    meta = _make_meta(fit, output_path, output_format, colormap, use_log10, positions, residuals_symmetric_cmap)

    has_pix = galaxies.has(cls=aa.Pixelization)
    n = 4 if has_pix else 3
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes_flat = list(axes.flatten())

    meta._plot_array(fit.data, "data", "Data", ax=axes_flat[0])
    meta._plot_array(
        fit.subtracted_images_of_galaxies_list[galaxy_index],
        f"subtracted_image_of_galaxy_{galaxy_index}",
        f"Subtracted Image of Galaxy {galaxy_index}",
        ax=axes_flat[1],
    )
    meta._plot_array(
        fit.model_images_of_galaxies_list[galaxy_index],
        f"model_image_of_galaxy_{galaxy_index}",
        f"Model Image of Galaxy {galaxy_index}",
        ax=axes_flat[2],
    )

    if has_pix:
        inversion_plotter = aplt.InversionPlotter(
            inversion=fit.inversion,
            output=aplt.Output(path=output_path, format=output_format) if output_path else aplt.Output(),
        )
        inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

    plt.tight_layout()
    _save_subplot(fig, output_path, f"subplot_of_galaxy_{galaxy_index}", output_format)

import matplotlib.pyplot as plt
import numpy as np

import autoarray as aa
from autoarray.plot.plots.utils import plot_visibilities_1d

from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.galaxy.plot import galaxies_plots
from autogalaxy.plot.plot_utils import plot_array, _save_subplot


def subplot_fit(
    fit: FitInterferometer,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    residuals_symmetric_cmap: bool = True,
):
    panels = [
        (fit.residual_map, "Residual Map"),
        (fit.normalized_residual_map, "Normalized Residual Map"),
        (fit.chi_squared_map, "Chi-Squared Map"),
    ]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes_flat = list(axes.flatten())

    for i, (vis, title) in enumerate(panels):
        plot_visibilities_1d(vis, axes_flat[i], title)

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_fit", output_format)


def subplot_fit_dirty_images(
    fit: FitInterferometer,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    residuals_symmetric_cmap: bool = True,
):
    panels = [
        (fit.dirty_image, "Dirty Image"),
        (fit.dirty_signal_to_noise_map, "Dirty Signal-To-Noise Map"),
        (fit.dirty_model_image, "Dirty Model Image"),
        (fit.dirty_residual_map, "Dirty Residual Map"),
        (fit.dirty_normalized_residual_map, "Dirty Normalized Residual Map"),
        (fit.dirty_chi_squared_map, "Dirty Chi-Squared Map"),
    ]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes_flat = list(axes.flatten())

    for i, (array, title) in enumerate(panels):
        plot_array(
            array=array,
            title=title,
            colormap=colormap,
            use_log10=use_log10,
            ax=axes_flat[i],
        )

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_fit_dirty_images", output_format)


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
        panels = [
            (fit.dirty_image, "Dirty Image"),
            (fit.dirty_model_image, "Dirty Model Image"),
            (fit.dirty_residual_map, "Dirty Residual Map"),
        ]
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
        axes_flat = list(axes.flatten())
        for i, (array, title) in enumerate(panels):
            plot_array(
                array=array,
                title=title,
                colormap=colormap,
                use_log10=use_log10,
                ax=axes_flat[i],
            )
        plt.tight_layout()
        _save_subplot(fig, output_path, "subplot_fit_real_space", output_format)

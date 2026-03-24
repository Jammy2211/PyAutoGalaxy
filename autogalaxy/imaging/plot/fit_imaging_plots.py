import matplotlib.pyplot as plt

import autoarray as aa

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.plot.plot_utils import plot_array, _save_subplot


def subplot_fit(
    fit: FitImaging,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    residuals_symmetric_cmap: bool = True,
):
    """Create a six-panel subplot summarising a :class:`~autogalaxy.imaging.fit_imaging.FitImaging`.

    The panels show, in order: data, signal-to-noise map, model image,
    residual map, normalised residual map, and chi-squared map.

    Parameters
    ----------
    fit : FitImaging
        The completed imaging fit to visualise.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the plotted values.
    positions : array-like or None
        Point positions to scatter-plot over each panel.
    residuals_symmetric_cmap : bool
        Reserved for future symmetric-colormap support on residual panels
        (currently unused).
    """
    panels = [
        (fit.data, "Data"),
        (fit.signal_to_noise_map, "Signal-To-Noise Map"),
        (fit.model_data, "Model Image"),
        (fit.residual_map, "Residual Map"),
        (fit.normalized_residual_map, "Normalized Residual Map"),
        (fit.chi_squared_map, "Chi-Squared Map"),
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
            positions=positions,
            ax=axes_flat[i],
        )

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_fit", output_format)


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
    """Create a three-panel subplot focused on a single galaxy contribution.

    Shows the observed data alongside the subtracted image and model image
    for the galaxy at *galaxy_index* in the fitted galaxy list.  This is
    useful for inspecting the contribution of individual galaxies when
    multiple galaxies are being fitted simultaneously.

    Parameters
    ----------
    fit : FitImaging
        The completed imaging fit to visualise.
    galaxy_index : int
        Index into ``fit.galaxies`` selecting which galaxy to highlight.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the plotted values.
    positions : array-like or None
        Point positions to scatter-plot over each panel.
    residuals_symmetric_cmap : bool
        Reserved for future symmetric-colormap support (currently unused).
    """
    panels = [
        (fit.data, "Data"),
        (
            fit.subtracted_images_of_galaxies_list[galaxy_index],
            f"Subtracted Image of Galaxy {galaxy_index}",
        ),
        (
            fit.model_images_of_galaxies_list[galaxy_index],
            f"Model Image of Galaxy {galaxy_index}",
        ),
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
    _save_subplot(fig, output_path, f"subplot_of_galaxy_{galaxy_index}", output_format)

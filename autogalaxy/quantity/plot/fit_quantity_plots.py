import matplotlib.pyplot as plt

import autoarray as aa
from autoarray.plot.utils import conf_subplot_figsize

from autogalaxy.quantity.fit_quantity import FitQuantity
from autogalaxy.plot.plot_utils import plot_array, _save_subplot


def _subplot_fit_array(fit, output_path, output_format, colormap, use_log10, positions, filename="fit"):
    """Render a six-panel fit summary subplot for a single array-valued quantity fit.

    The panels show: data, signal-to-noise map, model image, residual map,
    normalised residual map, and chi-squared map.  This internal helper is
    shared by both the scalar-array and vector-component paths in
    :func:`subplot_fit`.

    Parameters
    ----------
    fit
        A fit object exposing ``.data``, ``.signal_to_noise_map``,
        ``.model_data``, ``.residual_map``, ``.normalized_residual_map``,
        and ``.chi_squared_map`` as ``Array2D``-like objects.
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
    filename : str
        Output filename stem (default ``"subplot_fit"``).
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
    fig, axes = plt.subplots(1, n, figsize=conf_subplot_figsize(1, n))
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
    _save_subplot(fig, output_path, filename, output_format)


def subplot_fit(
    fit: FitQuantity,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    positions=None,
):
    """Create a summary subplot for a :class:`~autogalaxy.quantity.fit_quantity.FitQuantity`.

    The output depends on the type of the dataset's data:

    - **Scalar** (``aa.Array2D``): produces a single six-panel subplot saved
      as ``subplot_fit``.
    - **Vector** (anything else, e.g. a deflection-angle grid): produces two
      six-panel subplots, one for the y-component (``subplot_fit_y``) and one
      for the x-component (``subplot_fit_x``).

    Parameters
    ----------
    fit : FitQuantity
        The completed quantity fit to visualise.
    output_path : str or None
        Directory in which to save the figure(s).  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the plotted values.
    positions : array-like or None
        Point positions to scatter-plot over each panel.
    """
    if isinstance(fit.dataset.data, aa.Array2D):
        _subplot_fit_array(fit, output_path, output_format, colormap, use_log10, positions)
    else:
        _subplot_fit_array(
            fit.y, output_path, output_format, colormap, use_log10, positions, filename="fit_y"
        )
        _subplot_fit_array(
            fit.x, output_path, output_format, colormap, use_log10, positions, filename="fit_x"
        )

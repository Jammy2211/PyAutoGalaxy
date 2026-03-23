import matplotlib.pyplot as plt

import autoarray as aa

from autogalaxy.quantity.fit_quantity import FitQuantity
from autogalaxy.plot.plot_utils import plot_array, _save_subplot


def _subplot_fit_array(fit, output_path, output_format, colormap, use_log10, positions, filename="subplot_fit"):
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
    _save_subplot(fig, output_path, filename, output_format)


def subplot_fit(
    fit: FitQuantity,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
):
    if isinstance(fit.dataset.data, aa.Array2D):
        _subplot_fit_array(fit, output_path, output_format, colormap, use_log10, positions)
    else:
        _subplot_fit_array(
            fit.y, output_path, output_format, colormap, use_log10, positions, filename="subplot_fit_y"
        )
        _subplot_fit_array(
            fit.x, output_path, output_format, colormap, use_log10, positions, filename="subplot_fit_x"
        )

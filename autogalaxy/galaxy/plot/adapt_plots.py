import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plot.plot_utils import plot_array, _save_subplot


def subplot_adapt_images(
    adapt_galaxy_name_image_dict: Dict[Galaxy, aa.Array2D],
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
):
    """Create a subplot showing the adapt (model) image for each galaxy.

    Adapt images are per-galaxy model images produced during a previous
    non-linear search.  They are used to drive adaptive mesh and
    regularisation schemes in subsequent searches.  This function lays out
    one panel per entry in *adapt_galaxy_name_image_dict*, arranged in rows
    of up to three columns.

    If *adapt_galaxy_name_image_dict* is ``None`` the function returns
    immediately without producing any output.

    Parameters
    ----------
    adapt_galaxy_name_image_dict : dict[Galaxy, aa.Array2D] or None
        Mapping from galaxy (used as a label) to its adapt image array.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the image values.
    """
    if adapt_galaxy_name_image_dict is None:
        return

    n = len(adapt_galaxy_name_image_dict)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    axes_list = [axes] if n == 1 else list(np.array(axes).flatten())

    for i, (_, galaxy_image) in enumerate(adapt_galaxy_name_image_dict.items()):
        plot_array(
            array=galaxy_image,
            title="Galaxy Image",
            colormap=colormap,
            use_log10=use_log10,
            ax=axes_list[i],
        )

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_adapt_images", output_format)

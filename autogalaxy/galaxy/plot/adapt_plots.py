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

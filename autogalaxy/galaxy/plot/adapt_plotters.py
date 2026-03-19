import matplotlib.pyplot as plt
from typing import Dict

import autoarray as aa
import autoarray.plot as aplt

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plot.abstract_plotters import Plotter, _save_subplot


class AdaptPlotter(Plotter):
    def __init__(
        self,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

    def figure_model_image(self, model_image: aa.Array2D, ax=None):
        self._plot_array(
            array=model_image,
            auto_filename="adapt_model_image",
            title="adapt image",
            ax=ax,
        )

    def figure_galaxy_image(self, galaxy_image: aa.Array2D, ax=None):
        self._plot_array(
            array=galaxy_image,
            auto_filename="adapt_galaxy_image",
            title="galaxy Image",
            ax=ax,
        )

    def subplot_adapt_images(
        self, adapt_galaxy_name_image_dict: Dict[Galaxy, aa.Array2D]
    ):
        if adapt_galaxy_name_image_dict is None:
            return

        n = len(adapt_galaxy_name_image_dict)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
        import numpy as np
        axes = [axes] if n == 1 else list(np.array(axes).flatten())

        for i, (_, galaxy_image) in enumerate(adapt_galaxy_name_image_dict.items()):
            self.figure_galaxy_image(galaxy_image=galaxy_image, ax=axes[i])

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_adapt_images")

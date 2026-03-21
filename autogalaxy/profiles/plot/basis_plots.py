import matplotlib.pyplot as plt
import numpy as np

import autoarray as aa

from autogalaxy.profiles.basis import Basis
from autogalaxy.plot.plot_utils import _to_positions, plot_array, _save_subplot
from autogalaxy import exc


def subplot_image(
    basis: Basis,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    lines=None,
):
    from autogalaxy.profiles.light.linear import LightProfileLinear

    for light_profile in basis.light_profile_list:
        if isinstance(light_profile, LightProfileLinear):
            raise exc.raise_linear_light_profile_in_plot(
                plotter_type="subplot_image",
            )

    n = len(basis.light_profile_list)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    _positions = _to_positions(positions)

    for i, light_profile in enumerate(basis.light_profile_list):
        plot_array(
            array=light_profile.image_2d_from(grid=grid),
            title=light_profile.coefficient_tag,
            colormap=colormap,
            use_log10=use_log10,
            positions=_positions,
            lines=lines,
            ax=axes_flat[i],
        )

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_basis_image", output_format)

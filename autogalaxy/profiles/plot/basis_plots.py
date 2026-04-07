import matplotlib.pyplot as plt
import numpy as np

import autoarray as aa
from autoarray.plot.utils import conf_subplot_figsize, tight_layout

from autogalaxy.profiles.basis import Basis
from autogalaxy.plot.plot_utils import _to_positions, plot_array, _save_subplot
from autogalaxy import exc


def subplot_image(
    basis: Basis,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    positions=None,
    lines=None,
):
    """Create a subplot showing the image of every light profile in a basis.

    Produces one panel per profile in
    :attr:`~autogalaxy.profiles.basis.Basis.light_profile_list`, arranged in
    rows of up to four columns.  Each panel title is taken from the profile's
    ``coefficient_tag`` attribute.

    Linear light profiles cannot be plotted directly (their intensity is
    solved via inversion), so an error is raised if any are present.

    Parameters
    ----------
    basis : Basis
        The basis (e.g. MGE or shapelet set) whose component images are to be
        plotted.
    grid : aa.type.Grid1D2DLike
        The grid on which each light profile image is evaluated.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the image values.
    positions : array-like or None
        Point positions to scatter-plot over each panel.
    lines : list or None
        Line coordinates to overlay on each panel.
    """
    from autogalaxy.profiles.light.linear import LightProfileLinear

    for light_profile in basis.light_profile_list:
        if isinstance(light_profile, LightProfileLinear):
            raise exc.raise_linear_light_profile_in_plot(
                plotter_type="subplot_image",
            )

    n = len(basis.light_profile_list)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=conf_subplot_figsize(rows, cols))
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

    tight_layout()
    _save_subplot(fig, output_path, "basis_image", output_format)

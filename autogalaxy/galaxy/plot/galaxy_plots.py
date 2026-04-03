from __future__ import annotations
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

import autoarray as aa
from autoarray.plot.utils import conf_subplot_figsize

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.plot.plot_utils import plot_array, _save_subplot


def subplot_of_light_profiles(
    galaxy: Galaxy,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    positions=None,
):
    """Create a subplot showing the image of every light profile in a galaxy.

    One panel is drawn per :class:`~autogalaxy.profiles.light.abstract.LightProfile`
    attached to *galaxy*.  If the galaxy has no light profiles the function
    returns immediately without producing any output.

    Parameters
    ----------
    galaxy : Galaxy
        The galaxy whose light profiles are to be plotted.
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
    """
    light_profiles = galaxy.cls_list_from(cls=LightProfile)
    if not light_profiles:
        return

    n = len(light_profiles)
    fig, axes = plt.subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = [axes] if n == 1 else list(axes.flatten())

    for i, lp in enumerate(light_profiles):
        plot_array(
            array=lp.image_2d_from(grid=grid),
            title="Image",
            colormap=colormap,
            use_log10=use_log10,
            ax=axes_flat[i],
        )

    plt.tight_layout()
    _save_subplot(fig, output_path, "image", output_format)


def subplot_of_mass_profiles(
    galaxy: Galaxy,
    grid: aa.type.Grid2DLike,
    convergence: bool = False,
    potential: bool = False,
    deflections_y: bool = False,
    deflections_x: bool = False,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
):
    """Create subplots showing lensing quantities for every mass profile in a galaxy.

    One figure is produced *per requested quantity*, each containing one panel
    per :class:`~autogalaxy.profiles.mass.abstract.abstract.MassProfile`
    attached to *galaxy*.  Only the quantities whose corresponding flag is
    ``True`` are plotted; if none are requested, or if the galaxy has no mass
    profiles, the function returns without producing output.

    Parameters
    ----------
    galaxy : Galaxy
        The galaxy whose mass profiles are to be plotted.
    grid : aa.type.Grid2DLike
        The grid on which lensing quantities are evaluated.
    convergence : bool
        Plot the convergence map of each mass profile.
    potential : bool
        Plot the gravitational potential map of each mass profile.
    deflections_y : bool
        Plot the y-component of the deflection-angle map.
    deflections_x : bool
        Plot the x-component of the deflection-angle map.
    output_path : str or None
        Directory in which to save figures.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the plotted values.
    """
    mass_profiles = galaxy.cls_list_from(cls=MassProfile)
    if not mass_profiles:
        return

    n = len(mass_profiles)

    def _deflections_y(mp):
        deflections = mp.deflections_yx_2d_from(grid=grid)
        return aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)

    def _deflections_x(mp):
        deflections = mp.deflections_yx_2d_from(grid=grid)
        return aa.Array2D(values=deflections.slim[:, 1], mask=grid.mask)

    for name, flag, array_fn, title in [
        ("convergence", convergence, lambda mp: mp.convergence_2d_from(grid=grid), "Convergence"),
        ("potential", potential, lambda mp: mp.potential_2d_from(grid=grid), "Potential"),
        ("deflections_y", deflections_y, _deflections_y, "Deflections Y"),
        ("deflections_x", deflections_x, _deflections_x, "Deflections X"),
    ]:
        if not flag:
            continue

        fig, axes = plt.subplots(1, n, figsize=conf_subplot_figsize(1, n))
        axes_flat = [axes] if n == 1 else list(axes.flatten())

        for i, mp in enumerate(mass_profiles):
            plot_array(
                array=array_fn(mp),
                title=title,
                colormap=colormap,
                use_log10=use_log10,
                ax=axes_flat[i],
            )

        plt.tight_layout()
        _save_subplot(fig, output_path, name, output_format)

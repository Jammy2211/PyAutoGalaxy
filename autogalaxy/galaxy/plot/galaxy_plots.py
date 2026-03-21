from __future__ import annotations
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.plot.plot_utils import plot_array, _save_subplot


def subplot_of_light_profiles(
    galaxy: Galaxy,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
):
    light_profiles = galaxy.cls_list_from(cls=LightProfile)
    if not light_profiles:
        return

    n = len(light_profiles)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
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
    _save_subplot(fig, output_path, "subplot_image", output_format)


def subplot_of_mass_profiles(
    galaxy: Galaxy,
    grid: aa.type.Grid2DLike,
    convergence: bool = False,
    potential: bool = False,
    deflections_y: bool = False,
    deflections_x: bool = False,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
):
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

        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
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
        _save_subplot(fig, output_path, f"subplot_{name}", output_format)

import matplotlib.pyplot as plt
import numpy as np

import autoarray as aa

from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.plot.plot_utils import _to_lines, _to_positions, plot_array, plot_grid, _save_subplot, _critical_curves_from
from autogalaxy import exc


def _check_no_linear(galaxies):
    from autogalaxy.profiles.light.linear import LightProfileLinear

    if Galaxies(galaxies=galaxies).has(cls=LightProfileLinear):
        raise exc.raise_linear_light_profile_in_plot(plotter_type="galaxies plot")


def _galaxies_critical_curves(galaxies, grid, tc=None, rc=None):
    return _critical_curves_from(galaxies, grid, tc=tc, rc=rc)


def subplot_galaxies(
    galaxies,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    auto_filename="subplot_galaxies",
):
    _check_no_linear(galaxies)
    gs = Galaxies(galaxies=galaxies)
    tc, rc = _galaxies_critical_curves(
        gs, grid, tc=tangential_critical_curves, rc=radial_critical_curves
    )
    lines = _to_lines(tc, rc)
    pos = _to_positions(positions, light_profile_centres, mass_profile_centres, multiple_images)
    pos_no_cc = _to_positions(positions, light_profile_centres, mass_profile_centres)

    def _defl_y():
        d = gs.deflections_yx_2d_from(grid=grid)
        return aa.Array2D(values=d.slim[:, 0], mask=grid.mask)

    def _defl_x():
        d = gs.deflections_yx_2d_from(grid=grid)
        return aa.Array2D(values=d.slim[:, 1], mask=grid.mask)

    panels = [
        ("image", gs.image_2d_from(grid=grid), "Image", pos_no_cc, None),
        ("convergence", gs.convergence_2d_from(grid=grid), "Convergence", pos, lines),
        ("potential", gs.potential_2d_from(grid=grid), "Potential", pos, lines),
        ("deflections_y", _defl_y(), "Deflections Y", pos, lines),
        ("deflections_x", _defl_x(), "Deflections X", pos, lines),
    ]

    n = len(panels)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    for i, (_, array, title, p, l) in enumerate(panels):
        plot_array(
            array=array,
            title=title,
            colormap=colormap,
            use_log10=use_log10,
            positions=p,
            lines=l,
            ax=axes_flat[i],
        )

    plt.tight_layout()
    _save_subplot(fig, output_path, auto_filename, output_format)


def subplot_galaxy_images(
    galaxies,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    tangential_critical_curves=None,
    radial_critical_curves=None,
):
    _check_no_linear(galaxies)
    gs = Galaxies(galaxies=galaxies)

    n = len(gs)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes_flat = [axes] if n == 1 else list(axes.flatten())

    for i in range(n):
        plot_array(
            array=gs[i].image_2d_from(grid=grid),
            title=f"Image Of Galaxies {i}",
            colormap=colormap,
            use_log10=use_log10,
            ax=axes_flat[i],
        )

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_galaxy_images", output_format)

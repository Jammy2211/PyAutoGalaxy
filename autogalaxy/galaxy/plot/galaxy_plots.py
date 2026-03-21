from __future__ import annotations
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Optional

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.plot.plot_utils import _to_lines, _to_positions, plot_array, _save_subplot, _critical_curves_from
from autogalaxy import exc


def _galaxy_critical_curves(galaxy, grid, tc=None, rc=None):
    return _critical_curves_from(galaxy, grid, tc=tc, rc=rc)


def plot_image_2d(
    galaxy: Galaxy,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_filename="image_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    title_suffix="",
    filename_suffix="",
    ax=None,
):
    from autogalaxy.profiles.light.linear import LightProfileLinear

    if galaxy is not None and galaxy.has(cls=LightProfileLinear):
        raise exc.raise_linear_light_profile_in_plot(plotter_type="plot_image_2d")

    pos = _to_positions(positions, light_profile_centres, mass_profile_centres)

    plot_array(
        array=galaxy.image_2d_from(grid=grid),
        title=f"Image{title_suffix}",
        output_path=output_path,
        output_filename=f"{output_filename}{filename_suffix}",
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=pos,
        ax=ax,
    )


def _plot_mass_quantity(
    galaxy,
    grid,
    array,
    output_filename,
    title,
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
    title_suffix="",
    filename_suffix="",
):
    tc, rc = _galaxy_critical_curves(
        galaxy, grid, tc=tangential_critical_curves, rc=radial_critical_curves
    )
    lines = _to_lines(tc, rc)
    pos = _to_positions(positions, light_profile_centres, mass_profile_centres, multiple_images)

    plot_array(
        array=array,
        title=f"{title}{title_suffix}",
        output_path=output_path,
        output_filename=f"{output_filename}{filename_suffix}",
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=pos,
        lines=lines,
    )


def plot_convergence_2d(
    galaxy: Galaxy,
    grid: aa.type.Grid2DLike,
    output_path=None,
    output_filename="convergence_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    title_suffix="",
    filename_suffix="",
):
    _plot_mass_quantity(
        galaxy=galaxy, grid=grid,
        array=galaxy.convergence_2d_from(grid=grid),
        output_filename=output_filename, title="Convergence",
        output_path=output_path, output_format=output_format,
        colormap=colormap, use_log10=use_log10,
        positions=positions, light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres, multiple_images=multiple_images,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        title_suffix=title_suffix, filename_suffix=filename_suffix,
    )


def plot_potential_2d(
    galaxy: Galaxy,
    grid: aa.type.Grid2DLike,
    output_path=None,
    output_filename="potential_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    title_suffix="",
    filename_suffix="",
):
    _plot_mass_quantity(
        galaxy=galaxy, grid=grid,
        array=galaxy.potential_2d_from(grid=grid),
        output_filename=output_filename, title="Potential",
        output_path=output_path, output_format=output_format,
        colormap=colormap, use_log10=use_log10,
        positions=positions, light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres, multiple_images=multiple_images,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        title_suffix=title_suffix, filename_suffix=filename_suffix,
    )


def plot_deflections_y_2d(
    galaxy: Galaxy,
    grid: aa.type.Grid2DLike,
    output_path=None,
    output_filename="deflections_y_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    title_suffix="",
    filename_suffix="",
):
    deflections = galaxy.deflections_yx_2d_from(grid=grid)
    array = aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)

    _plot_mass_quantity(
        galaxy=galaxy, grid=grid, array=array,
        output_filename=output_filename, title="Deflections Y",
        output_path=output_path, output_format=output_format,
        colormap=colormap, use_log10=use_log10,
        positions=positions, light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres, multiple_images=multiple_images,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        title_suffix=title_suffix, filename_suffix=filename_suffix,
    )


def plot_deflections_x_2d(
    galaxy: Galaxy,
    grid: aa.type.Grid2DLike,
    output_path=None,
    output_filename="deflections_x_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    title_suffix="",
    filename_suffix="",
):
    deflections = galaxy.deflections_yx_2d_from(grid=grid)
    array = aa.Array2D(values=deflections.slim[:, 1], mask=grid.mask)

    _plot_mass_quantity(
        galaxy=galaxy, grid=grid, array=array,
        output_filename=output_filename, title="Deflections X",
        output_path=output_path, output_format=output_format,
        colormap=colormap, use_log10=use_log10,
        positions=positions, light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres, multiple_images=multiple_images,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        title_suffix=title_suffix, filename_suffix=filename_suffix,
    )


def plot_magnification_2d(
    galaxy: Galaxy,
    grid: aa.type.Grid2DLike,
    output_path=None,
    output_filename="magnification_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    title_suffix="",
    filename_suffix="",
):
    from autogalaxy.operate.lens_calc import LensCalc

    array = LensCalc.from_mass_obj(galaxy).magnification_2d_from(grid=grid)

    _plot_mass_quantity(
        galaxy=galaxy, grid=grid, array=array,
        output_filename=output_filename, title="Magnification",
        output_path=output_path, output_format=output_format,
        colormap=colormap, use_log10=use_log10,
        positions=positions, light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres, multiple_images=multiple_images,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        title_suffix=title_suffix, filename_suffix=filename_suffix,
    )


def subplot_of_light_profiles(
    galaxy: Galaxy,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
):
    from autogalaxy.profiles.plot.light_profile_plots import plot_image_2d as plot_lp_image_2d

    light_profiles = galaxy.cls_list_from(cls=LightProfile)
    if not light_profiles:
        return

    n = len(light_profiles)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes_flat = [axes] if n == 1 else list(axes.flatten())

    for i, lp in enumerate(light_profiles):
        plot_lp_image_2d(
            light_profile=lp,
            grid=grid,
            colormap=colormap,
            use_log10=use_log10,
            positions=positions,
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

    from autogalaxy.plot.plot_utils import plot_array

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

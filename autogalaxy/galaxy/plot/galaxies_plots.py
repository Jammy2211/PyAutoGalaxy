import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.plot.plot_utils import _to_lines, _to_positions, plot_array, plot_grid, _save_subplot, _critical_curves_from
from autogalaxy import exc


def _galaxies_critical_curves(galaxies, grid, tc=None, rc=None):
    return _critical_curves_from(galaxies, grid, tc=tc, rc=rc)


def _check_no_linear(galaxies):
    from autogalaxy.profiles.light.linear import LightProfileLinear

    if Galaxies(galaxies=galaxies).has(cls=LightProfileLinear):
        raise exc.raise_linear_light_profile_in_plot(plotter_type="galaxies plot")


def plot_image_2d(
    galaxies,
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
    _check_no_linear(galaxies)
    gs = Galaxies(galaxies=galaxies)
    pos = _to_positions(positions, light_profile_centres, mass_profile_centres)

    plot_array(
        array=gs.image_2d_from(grid=grid),
        title=f"Image{title_suffix}",
        output_path=output_path,
        output_filename=f"{output_filename}{filename_suffix}",
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=pos,
        ax=ax,
    )


def _plot_galaxies_mass_quantity(
    galaxies,
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
    ax=None,
):
    gs = Galaxies(galaxies=galaxies)
    tc, rc = _galaxies_critical_curves(
        gs, grid, tc=tangential_critical_curves, rc=radial_critical_curves
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
        ax=ax,
    )


def plot_convergence_2d(
    galaxies,
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
    ax=None,
):
    gs = Galaxies(galaxies=galaxies)
    _plot_galaxies_mass_quantity(
        galaxies=gs, grid=grid,
        array=gs.convergence_2d_from(grid=grid),
        output_filename=output_filename, title="Convergence",
        output_path=output_path, output_format=output_format,
        colormap=colormap, use_log10=use_log10,
        positions=positions, light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres, multiple_images=multiple_images,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        title_suffix=title_suffix, filename_suffix=filename_suffix, ax=ax,
    )


def plot_potential_2d(
    galaxies,
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
    ax=None,
):
    gs = Galaxies(galaxies=galaxies)
    _plot_galaxies_mass_quantity(
        galaxies=gs, grid=grid,
        array=gs.potential_2d_from(grid=grid),
        output_filename=output_filename, title="Potential",
        output_path=output_path, output_format=output_format,
        colormap=colormap, use_log10=use_log10,
        positions=positions, light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres, multiple_images=multiple_images,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        title_suffix=title_suffix, filename_suffix=filename_suffix, ax=ax,
    )


def plot_deflections_y_2d(
    galaxies,
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
    ax=None,
):
    gs = Galaxies(galaxies=galaxies)
    deflections = gs.deflections_yx_2d_from(grid=grid)
    array = aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)
    _plot_galaxies_mass_quantity(
        galaxies=gs, grid=grid, array=array,
        output_filename=output_filename, title="Deflections Y",
        output_path=output_path, output_format=output_format,
        colormap=colormap, use_log10=use_log10,
        positions=positions, light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres, multiple_images=multiple_images,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        title_suffix=title_suffix, filename_suffix=filename_suffix, ax=ax,
    )


def plot_deflections_x_2d(
    galaxies,
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
    ax=None,
):
    gs = Galaxies(galaxies=galaxies)
    deflections = gs.deflections_yx_2d_from(grid=grid)
    array = aa.Array2D(values=deflections.slim[:, 1], mask=grid.mask)
    _plot_galaxies_mass_quantity(
        galaxies=gs, grid=grid, array=array,
        output_filename=output_filename, title="Deflections X",
        output_path=output_path, output_format=output_format,
        colormap=colormap, use_log10=use_log10,
        positions=positions, light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres, multiple_images=multiple_images,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        title_suffix=title_suffix, filename_suffix=filename_suffix, ax=ax,
    )


def plot_magnification_2d(
    galaxies,
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
    ax=None,
):
    from autogalaxy.operate.lens_calc import LensCalc

    gs = Galaxies(galaxies=galaxies)
    array = LensCalc.from_mass_obj(gs).magnification_2d_from(grid=grid)
    _plot_galaxies_mass_quantity(
        galaxies=gs, grid=grid, array=array,
        output_filename=output_filename, title="Magnification",
        output_path=output_path, output_format=output_format,
        colormap=colormap, use_log10=use_log10,
        positions=positions, light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres, multiple_images=multiple_images,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        title_suffix=title_suffix, filename_suffix=filename_suffix, ax=ax,
    )


def plot_plane_image_2d(
    galaxies,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_filename="plane_image",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    zoom_to_brightest: bool = True,
    title_suffix="",
    filename_suffix="",
    source_plane_title: bool = False,
    ax=None,
):
    _check_no_linear(galaxies)
    gs = Galaxies(galaxies=galaxies)
    title = "Source Plane Image" if source_plane_title else f"Plane Image{title_suffix}"
    pos = _to_positions(positions)

    plot_array(
        array=gs.plane_image_2d_from(grid=grid, zoom_to_brightest=zoom_to_brightest),
        title=title,
        output_path=output_path,
        output_filename=f"{output_filename}{filename_suffix}",
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=pos,
        ax=ax,
    )


def plot_plane_grid_2d(
    galaxies,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_filename="plane_grid",
    output_format="png",
    title_suffix="",
    filename_suffix="",
    source_plane_title: bool = False,
    ax=None,
):
    title = "Source Plane Grid" if source_plane_title else f"Plane Grid{title_suffix}"

    plot_grid(
        grid=grid,
        title=title,
        output_path=output_path,
        output_filename=f"{output_filename}{filename_suffix}",
        output_format=output_format,
        ax=ax,
    )


def plot_image_2d_of_galaxy(
    galaxies,
    grid: aa.type.Grid1D2DLike,
    galaxy_index: int,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    tangential_critical_curves=None,
    radial_critical_curves=None,
):
    _check_no_linear(galaxies)
    gs = Galaxies(galaxies=galaxies)
    tc, rc = _galaxies_critical_curves(
        gs, grid, tc=tangential_critical_curves, rc=radial_critical_curves
    )

    plot_image_2d(
        galaxies=[gs[galaxy_index]],
        grid=grid,
        output_path=output_path,
        output_filename=f"image_2d_of_galaxy_{galaxy_index}",
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        title_suffix=f" Of Galaxy {galaxy_index}",
    )


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

    plot_fns = [
        ("image", plot_image_2d),
        ("convergence", plot_convergence_2d),
        ("potential", plot_potential_2d),
        ("deflections_y", plot_deflections_y_2d),
        ("deflections_x", plot_deflections_x_2d),
    ]

    n = len(plot_fns)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    for i, (name, fn) in enumerate(plot_fns):
        kwargs = dict(
            galaxies=gs,
            grid=grid,
            colormap=colormap,
            use_log10=use_log10,
            positions=positions,
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            multiple_images=multiple_images,
            tangential_critical_curves=tc,
            radial_critical_curves=rc,
            ax=axes_flat[i],
        )
        if name == "image":
            del kwargs["tangential_critical_curves"]
            del kwargs["radial_critical_curves"]
            del kwargs["multiple_images"]
        fn(**kwargs)

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
    tc, rc = _galaxies_critical_curves(
        gs, grid, tc=tangential_critical_curves, rc=radial_critical_curves
    )

    n = len(gs)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes_flat = [axes] if n == 1 else list(axes.flatten())

    for i in range(n):
        plot_image_2d(
            galaxies=[gs[i]],
            grid=grid,
            colormap=colormap,
            use_log10=use_log10,
            title_suffix=f" Of Galaxies {i}",
            ax=axes_flat[i],
        )

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_galaxy_images", output_format)

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.plot.plot_utils import _to_lines, _to_positions, plot_array, _critical_curves_from


def _mass_plot(
    mass_profile: MassProfile,
    grid: aa.type.Grid2DLike,
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
    tangential_critical_curves=None,
    radial_critical_curves=None,
    ax=None,
):
    tc, rc = _critical_curves_from(
        mass_profile, grid, tc=tangential_critical_curves, rc=radial_critical_curves
    )
    lines = _to_lines(tc, rc)
    pos = _to_positions(positions, light_profile_centres, mass_profile_centres)

    plot_array(
        array=array,
        title=title,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=pos,
        lines=lines,
        ax=ax,
    )


def plot_convergence_2d(
    mass_profile: MassProfile,
    grid: aa.type.Grid2DLike,
    output_path=None,
    output_filename="convergence_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    ax=None,
):
    _mass_plot(
        mass_profile=mass_profile,
        grid=grid,
        array=mass_profile.convergence_2d_from(grid=grid),
        output_filename=output_filename,
        title="Convergence",
        output_path=output_path,
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=positions,
        light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        ax=ax,
    )


def plot_potential_2d(
    mass_profile: MassProfile,
    grid: aa.type.Grid2DLike,
    output_path=None,
    output_filename="potential_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    ax=None,
):
    _mass_plot(
        mass_profile=mass_profile,
        grid=grid,
        array=mass_profile.potential_2d_from(grid=grid),
        output_filename=output_filename,
        title="Potential",
        output_path=output_path,
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=positions,
        light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        ax=ax,
    )


def plot_deflections_y_2d(
    mass_profile: MassProfile,
    grid: aa.type.Grid2DLike,
    output_path=None,
    output_filename="deflections_y_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    ax=None,
):
    deflections = mass_profile.deflections_yx_2d_from(grid=grid)
    array = aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)

    _mass_plot(
        mass_profile=mass_profile,
        grid=grid,
        array=array,
        output_filename=output_filename,
        title="Deflections Y",
        output_path=output_path,
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=positions,
        light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        ax=ax,
    )


def plot_deflections_x_2d(
    mass_profile: MassProfile,
    grid: aa.type.Grid2DLike,
    output_path=None,
    output_filename="deflections_x_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    ax=None,
):
    deflections = mass_profile.deflections_yx_2d_from(grid=grid)
    array = aa.Array2D(values=deflections.slim[:, 1], mask=grid.mask)

    _mass_plot(
        mass_profile=mass_profile,
        grid=grid,
        array=array,
        output_filename=output_filename,
        title="Deflections X",
        output_path=output_path,
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=positions,
        light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        ax=ax,
    )


def plot_magnification_2d(
    mass_profile: MassProfile,
    grid: aa.type.Grid2DLike,
    output_path=None,
    output_filename="magnification_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    ax=None,
):
    from autogalaxy.operate.lens_calc import LensCalc

    array = LensCalc.from_mass_obj(mass_profile).magnification_2d_from(grid=grid)

    _mass_plot(
        mass_profile=mass_profile,
        grid=grid,
        array=array,
        output_filename=output_filename,
        title="Magnification",
        output_path=output_path,
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=positions,
        light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres,
        tangential_critical_curves=tangential_critical_curves,
        radial_critical_curves=radial_critical_curves,
        ax=ax,
    )

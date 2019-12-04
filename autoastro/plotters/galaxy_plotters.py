import autoarray as aa
import matplotlib

backend = aa.conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.util import plotter_util
from autoastro.plotters import profile_plotters, lens_plotter_util


def profile_image(
    galaxy,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
    figsize=(7, 7),
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Galaxy Profile Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_profile_image",
):
    """Plot the image (e.g. the datas) of a galaxy, on a grid of (y,x) coordinates.

    Set *autoastro.datas.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    galaxy : model.galaxy.aast.Galaxy
        The galaxy whose image are plotted.
    grid : ndarray or datas.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    image = galaxy.profile_image_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=galaxy,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=image,
        mask=mask,
        points=positions,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        point_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def convergence(
    galaxy,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
    figsize=(7, 7),
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Galaxy Convergence",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_convergence",
):
    """Plot the convergence of a galaxy, on a grid of (y,x) coordinates.

    Set *autoastro.datas.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    galaxy : model.galaxy.aast.Galaxy
        The galaxy whose convergence is plotted.
    grid : ndarray or datas.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    convergence = galaxy.convergence_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=galaxy,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=convergence,
        mask=mask,
        points=positions,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        point_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def potential(
    galaxy,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
    figsize=(7, 7),
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Galaxy Potential",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_potential",
):
    """Plot the potential of a galaxy, on a grid of (y,x) coordinates.

     Set *autoastro.datas.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
    galaxy : model.galaxy.aast.Galaxy
         The galaxy whose potential is plotted.
    grid : ndarray or datas.arrays.grid_stacks.Grid
         The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
     """
    potential = galaxy.potential_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=galaxy,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=potential,
        mask=mask,
        points=positions,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        point_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def deflections_y(
    galaxy,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
    figsize=(7, 7),
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Galaxy Deflections (y)",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_deflections_y",
):
    """Plot the y component of the deflection angles of a galaxy, on a grid of (y,x) coordinates.

    Set *autoastro.datas.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    galaxy : model.galaxy.aast.Galaxy
        The galaxy whose y deflecton angles are plotted.
    grid : ndarray or datas.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    deflections = galaxy.deflections_from_grid(grid=grid)
    deflections_y = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 0]
    )

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=galaxy,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=deflections_y,
        mask=mask,
        points=positions,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        point_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def deflections_x(
    galaxy,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
    figsize=(7, 7),
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Galaxy Deflections (x)",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_deflections_x",
):
    """Plot the x component of the deflection angles of a galaxy, on a grid of (y,x) coordinates.

     Set *autoastro.datas.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
    galaxy : model.galaxy.aast.Galaxy
         The galaxy whose x deflecton angles are plotted.
     grid : ndarray or datas.arrays.grid_stacks.Grid
         The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
     """
    deflections = galaxy.deflections_from_grid(grid=grid)
    deflections_x = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 1]
    )

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=galaxy,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=deflections_x,
        mask=mask,
        points=positions,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        point_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def magnification(
    galaxy,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
    figsize=(7, 7),
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Galaxy Magnification",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_magnification",
):
    """Plot the magnification of a galaxy, on a grid of (y,x) coordinates.

     Set *autoastro.datas.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
    galaxy : model.galaxy.aast.Galaxy
         The galaxy whose magnification is plotted.
    grid : ndarray or datas.arrays.grid_stacks.Grid
         The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
     """
    magnification = galaxy.magnification_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=galaxy,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=magnification,
        mask=mask,
        points=positions,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        point_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def profile_image_subplot(
    galaxy,
    grid,
    mask=None,
    positions=None,
    plot_in_kpc=False,
    figsize=None,
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_individual_image",
):

    total_light_profiles = len(galaxy.light_profiles)
    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=total_light_profiles
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    for i, light_profile in enumerate(galaxy.light_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.image(
            light_profile=light_profile,
            mask=mask,
            positions=positions,
            grid=grid,
            as_subplot=True,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            title="Galaxy Component",
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            mask_pointsize=mask_pointsize,
            position_pointsize=position_pointsize,
            grid_pointsize=grid_pointsize,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()


def convergence_subplot(
    galaxy,
    grid,
    mask=None,
    positions=None,
    plot_in_kpc=False,
    figsize=None,
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_individual_convergence",
):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=total_mass_profiles
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.convergence(
            mass_profile=mass_profile,
            grid=grid,
            mask=mask,
            positions=positions,
            as_subplot=True,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            title="Galaxy Component",
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            mask_pointsize=mask_pointsize,
            position_pointsize=position_pointsize,
            grid_pointsize=grid_pointsize,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()


def potential_subplot(
    galaxy,
    grid,
    mask=None,
    positions=None,
    plot_in_kpc=False,
    figsize=None,
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_individual_potential",
):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=total_mass_profiles
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.potential(
            mass_profile=mass_profile,
            grid=grid,
            mask=mask,
            positions=positions,
            as_subplot=True,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            title="Galaxy Component",
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            mask_pointsize=mask_pointsize,
            position_pointsize=position_pointsize,
            grid_pointsize=grid_pointsize,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()


def deflections_y_subplot(
    galaxy,
    grid,
    mask=None,
    positions=None,
    plot_in_kpc=False,
    figsize=None,
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_individual_deflections_y",
):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=total_mass_profiles
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.deflections_y(
            mass_profile=mass_profile,
            grid=grid,
            mask=mask,
            positions=positions,
            as_subplot=True,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            title="Galaxy Component",
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            mask_pointsize=mask_pointsize,
            position_pointsize=position_pointsize,
            grid_pointsize=grid_pointsize,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()


def deflections_x_subplot(
    galaxy,
    grid,
    mask=None,
    positions=None,
    plot_in_kpc=False,
    figsize=None,
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_individual_deflections_x",
):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=total_mass_profiles
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=galaxy, plot_in_kpc=plot_in_kpc
    )

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.deflections_x(
            mass_profile=mass_profile,
            grid=grid,
            mask=mask,
            positions=positions,
            as_subplot=True,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            title="Galaxy Component",
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            mask_pointsize=mask_pointsize,
            position_pointsize=position_pointsize,
            grid_pointsize=grid_pointsize,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()

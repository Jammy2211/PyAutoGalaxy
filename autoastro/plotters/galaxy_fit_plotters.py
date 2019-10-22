import autoarray as aa
import matplotlib

backend = aa.conf.instance.visualize.get("figures", "backend", str)
matplotlib.use(backend)
from matplotlib import pyplot as plt

import autoarray as aa
from autoarray.plotters import plotter_util
from autoastro import exc


def subplot(
    fit,
    positions=None,
    units="arcsec",
    kpc_per_arcsec=None,
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
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_filename="galaxy_fit",
    output_format="show",
):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=4
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    galaxy_data_array(
        galaxy_data=fit.galaxy_data,
        positions=positions,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        grid_pointsize=grid_pointsize,
        position_pointsize=position_pointsize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    aa.plot.fit.model_image(
        fit=fit,
        mask_overlay=fit.mask,
        positions=positions,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        title="Model Galaxy",
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        position_pointsize=position_pointsize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 3)

    aa.plot.fit.residual_map(
        fit=fit,
        mask_overlay=fit.mask,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        position_pointsize=position_pointsize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 4)

    aa.plot.fit.chi_squared_map(
        fit=fit,
        mask_overlay=fit.mask,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        position_pointsize=position_pointsize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def individuals(
    fit,
    should_plot_mask_overlay=True,
    positions=None,
    should_plot_image=False,
    should_plot_noise_map=False,
    should_plot_model_image=False,
    should_plot_residual_map=False,
    should_plot_chi_squared_map=False,
    units="kpc",
    output_path=None,
    output_format="show",
):

    mask = aa.plot.fit.get_mask_overlay(fit=fit, should_plot_mask_overlay=should_plot_mask_overlay)

    kpc_per_arcsec = None

    if should_plot_image:

        galaxy_data_array(
            galaxy_data=fit.galaxy_data,
            mask_overlay=fit.mask,
            positions=positions,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_noise_map:

        aa.plot.fit.noise_map(
            fit=fit,
            mask_overlay=fit.mask,
            positions=positions,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_model_image:

        aa.plot.fit.model_image(
            fit=fit,
            mask_overlay=fit.mask,
            positions=positions,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_residual_map:

        aa.plot.fit.residual_map(
            fit=fit,
            mask_overlay=fit.mask,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_chi_squared_map:

        aa.plot.fit.chi_squared_map(
            fit=fit,
            mask_overlay=fit.mask,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )


def galaxy_data_array(
    galaxy_data,
    positions=None,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
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
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_filename="galaxy_data",
    output_format="show",
):

    if galaxy_data.use_image:
        title = "Galaxy Data Image"
    elif galaxy_data.use_convergence:
        title = "Galaxy Data Convergence"
    elif galaxy_data.use_potential:
        title = "Galaxy Data Potential"
    elif galaxy_data.use_deflections_y:
        title = "Galaxy Data Deflections (y)"
    elif galaxy_data.use_deflections_x:
        title = "Galaxy Data Deflections (x)"
    else:
        raise exc.PlottingException(
            "The galaxy data_type array does not have a True use_profile_type"
        )

    aa.plot.array(
        array=galaxy_data.image,
        mask_overlay=galaxy_data.mask,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        position_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


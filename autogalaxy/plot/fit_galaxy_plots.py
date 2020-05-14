import autoarray as aa
from autoarray.plot import plotters
from autogalaxy import exc
from autogalaxy.plot import lensing_plotters


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_fit_galaxy(fit, positions=None, include=None, sub_plotter=None):

    number_subplots = 4

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    galaxy_data_array(
        galaxy_data=fit.masked_galaxy_dataset, positions=positions, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    aa.plot.FitImaging.model_image(
        fit=fit, include=include, positions=positions, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    aa.plot.FitImaging.residual_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    aa.plot.FitImaging.chi_squared_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def individuals(
    fit,
    positions=None,
    plot_image=False,
    plot_noise_map=False,
    plot_model_image=False,
    plot_residual_map=False,
    plot_chi_squared_map=False,
    include=None,
    plotter=None,
):

    if plot_image:

        galaxy_data_array(
            galaxy_data=fit.masked_galaxy_dataset,
            mask=fit.mask,
            positions=positions,
            include=include,
            plotter=plotter,
        )

    if plot_noise_map:

        aa.plot.FitImaging.noise_map(
            fit=fit,
            mask=fit.mask,
            positions=positions,
            include=include,
            plotter=plotter,
        )

    if plot_model_image:

        aa.plot.FitImaging.model_image(
            fit=fit,
            mask=fit.mask,
            positions=positions,
            include=include,
            plotter=plotter,
        )

    if plot_residual_map:

        aa.plot.FitImaging.residual_map(
            fit=fit, mask=fit.mask, include=include, plotter=plotter
        )

    if plot_chi_squared_map:

        aa.plot.FitImaging.chi_squared_map(
            fit=fit, mask=fit.mask, include=include, plotter=plotter
        )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def galaxy_data_array(galaxy_data, positions=None, include=None, plotter=None):

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
            "The galaxy data arrays does not have a True use_profile_type"
        )

    plotter.plot_array(
        array=galaxy_data.image,
        mask=galaxy_data.mask,
        positions=positions,
        include_origin=include.origin,
    )

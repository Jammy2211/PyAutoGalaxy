from autoarray.plot.plots import fit_imaging_plots
from autoarray.plot.plots import structure_plots
from autogalaxy import exc
from autoarray.plot.mat_wrap import mat_decorators
from autogalaxy.plot.mat_wrap import lensing_plotter, lensing_include, lensing_visuals


def subplot_fit_galaxy(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    plotter_2d = plotter_2d.plotter_for_subplot_from(func=subplot_fit_galaxy)

    number_subplots = 4

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    galaxy_data_array(
        galaxy_data=fit.masked_galaxy_dataset,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    fit_imaging_plots.model_image(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    fit_imaging_plots.residual_map(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    fit_imaging_plots.chi_squared_map(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


def individuals(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    plot_image=False,
    plot_noise_map=False,
    plot_model_image=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
):

    if plot_image:

        galaxy_data_array(
            galaxy_data=fit.masked_galaxy_dataset,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    fit_imaging_plots.individuals(
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plot_noise_map=plot_noise_map,
        plot_model_image=plot_model_image,
        plot_residual_map=plot_residual_map,
        plot_normalized_residual_map=plot_normalized_residual_map,
        plot_chi_squared_map=plot_chi_squared_map,
    )


@mat_decorators.set_labels
def galaxy_data_array(
    galaxy_data,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
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
            "The galaxy data arrays does not have a `True` use_profile_type"
        )

    structure_plots.plot_array(
        array=galaxy_data.image,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )

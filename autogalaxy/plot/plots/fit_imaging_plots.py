import numpy as np
from autoarray.plot.plots import structure_plots, fit_imaging_plots
from autogalaxy.plot.plots import inversion_plots
from autoarray.plot.mat_wrap import mat_decorators
from autogalaxy.plot.mat_wrap import lensing_plotter, lensing_include, lensing_visuals


def subplot_fit_imaging(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    fit_imaging_plots.subplot_fit_imaging(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )


def individuals(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_image=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_subtracted_images_of_galaxies=False,
    plot_model_images_of_galaxies=False,
):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autogalaxy.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    fit_imaging_plots.individuals(
        fit=fit,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plot_image=plot_image,
        plot_noise_map=plot_noise_map,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_model_image=plot_model_image,
        plot_residual_map=plot_residual_map,
        plot_normalized_residual_map=plot_normalized_residual_map,
        plot_chi_squared_map=plot_chi_squared_map,
    )

    if plot_subtracted_images_of_galaxies:

        for galaxy_index in range(len(fit.galaxies)):

            subtracted_image_of_galaxy(
                fit=fit,
                galaxy_index=galaxy_index,
                include_2d=include_2d,
                plotter_2d=plotter_2d,
            )

    if plot_model_images_of_galaxies:

        for galaxy_index in range(len(fit.galaxies)):

            model_image_of_galaxy(
                fit=fit,
                galaxy_index=galaxy_index,
                include_2d=include_2d,
                plotter_2d=plotter_2d,
            )


@mat_decorators.set_labels
def image(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the image of a lens fit.

    Set *autogalaxy.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    fit_imaging_plots.image(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )


@mat_decorators.set_labels
def noise_map(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the noise-map of a lens fit.

    Set *autogalaxy.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    fit_imaging_plots.noise_map(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )


@mat_decorators.set_labels
def signal_to_noise_map(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the noise-map of a lens fit.

    Set *autogalaxy.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
    The datas-datas, which include_2d the observed datas, signal_to_noise_map, PSF, signal-to-signal_to_noise_map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    fit_imaging_plots.signal_to_noise_map(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )


@mat_decorators.set_labels
def model_image(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the model image of a fit.

    Set *autogalaxy.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    """
    fit_imaging_plots.model_image(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )


@mat_decorators.set_labels
def residual_map(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the residual-map of a lens fit.

    Set *autogalaxy.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """
    fit_imaging_plots.residual_map(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )


@mat_decorators.set_labels
def normalized_residual_map(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the residual-map of a lens fit.

    Set *autogalaxy.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model image, normalized_residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the normalized_residual_map are plotted.
    """
    fit_imaging_plots.normalized_residual_map(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )


@mat_decorators.set_labels
def chi_squared_map(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the chi-squared-map of a lens fit.

    Set *autogalaxy.datas.array.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the chi-squareds are plotted.
    """
    fit_imaging_plots.chi_squared_map(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )


def subplots_of_all_galaxies(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    for galaxy_index in range(len(fit.galaxies)):

        if (
            fit.galaxies[galaxy_index].has_light_profile
            or fit.galaxies[galaxy_index].has_pixelization
        ):

            subplot_of_galaxy(
                fit=fit,
                galaxy_index=galaxy_index,
                plotter_2d=plotter_2d,
                visuals_2d=visuals_2d,
                include_2d=include_2d,
            )


def subplot_of_galaxy(
    fit,
    galaxy_index,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autogalaxy.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    plotter_2d = plotter_2d.plotter_for_subplot_from(func=subplot_of_galaxy)

    number_subplots = 4

    plotter_2d = plotter_2d.plotter_with_new_output(
        filename=f"{plotter_2d.output.filename}_galaxy_index"
    )

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    fit_imaging_plots.image(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    subtracted_image_of_galaxy(
        fit=fit,
        galaxy_index=galaxy_index,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    model_image_of_galaxy(
        fit=fit,
        galaxy_index=galaxy_index,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )

    if fit.plane.has_pixelization:

        aspect_inv = plotter_2d.figure.aspect_for_subplot_from_grid(
            grid=fit.inversion.mapper.source_full_grid
        )

        plotter_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=4, aspect=float(aspect_inv)
        )

        inversion_plots.reconstruction(
            inversion=fit.inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


@mat_decorators.set_labels
def subtracted_image_of_galaxy(
    fit,
    galaxy_index,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the model image of a specific plane of a lens fit.

    Set *autogalaxy.datas.arrays.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    galaxy_indexes : int
        The plane from which the model image is generated.
    """

    if not plotter_2d.for_subplot:
        plotter_2d = plotter_2d.plotter_with_new_output(
            filename=f"{plotter_2d.output.filename}_{galaxy_index}"
        )

    if len(fit.galaxies) > 1:

        other_galaxies_model_images = [
            model_image
            for i, model_image in enumerate(fit.model_images_of_galaxies)
            if i != galaxy_index
        ]

        subtracted_image = fit.image - sum(other_galaxies_model_images)

    else:

        subtracted_image = fit.image

    plotter_2d_norm = plotter_2d.plotter_with_new_cmap(
        vmax=np.max(fit.model_images_of_galaxies[galaxy_index]),
        vmin=np.min(fit.model_images_of_galaxies[galaxy_index]),
    )

    structure_plots.plot_array(
        array=subtracted_image,
        plotter_2d=plotter_2d_norm,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )


@mat_decorators.set_labels
def model_image_of_galaxy(
    fit,
    galaxy_index,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the model image of a specific plane of a lens fit.

    Set *autogalaxy.datas.arrays.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    galaxy_indexes : [int]
        The plane from which the model image is generated.
    """

    if not plotter_2d.for_subplot:
        plotter_2d = plotter_2d.plotter_with_new_output(
            filename=f"{plotter_2d.output.filename}_{galaxy_index}"
        )

    structure_plots.plot_array(
        array=fit.model_images_of_galaxies[galaxy_index],
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )

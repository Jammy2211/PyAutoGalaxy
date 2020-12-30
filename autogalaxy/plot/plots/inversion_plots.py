from autoarray.plot.plots import inversion_plots
from autoarray.plot.mat_wrap import mat_decorators
from autogalaxy.plot.mat_wrap import lensing_plotter, lensing_include, lensing_visuals


def subplot_inversion(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    full_indexes=None,
    pixelization_indexes=None,
):

    inversion_plots.subplot_inversion(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


def individuals(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    plot_reconstructed_image=False,
    plot_reconstruction=False,
    plot_errors=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_regularization_weight_map=False,
    plot_interpolated_reconstruction=False,
    plot_interpolated_errors=False,
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

    inversion_plots.individuals(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plot_reconstructed_image=plot_reconstructed_image,
        plot_reconstruction=plot_reconstruction,
        plot_errors=plot_errors,
        plot_residual_map=plot_residual_map,
        plot_normalized_residual_map=plot_normalized_residual_map,
        plot_chi_squared_map=plot_chi_squared_map,
        plot_regularization_weight_map=plot_regularization_weight_map,
        plot_interpolated_reconstruction=plot_interpolated_reconstruction,
        plot_interpolated_errors=plot_interpolated_errors,
    )


@mat_decorators.set_labels
def reconstructed_image(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    inversion_plots.reconstructed_image(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )


@mat_decorators.set_labels
def reconstruction(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    full_indexes=None,
    pixelization_indexes=None,
):

    inversion_plots.reconstruction(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_labels
def errors(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    full_indexes=None,
    pixelization_indexes=None,
):

    inversion_plots.errors(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_labels
def residual_map(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    full_indexes=None,
    pixelization_indexes=None,
):

    inversion_plots.residual_map(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_labels
def normalized_residual_map(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    full_indexes=None,
    pixelization_indexes=None,
):

    inversion_plots.normalized_residual_map(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_labels
def chi_squared_map(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    full_indexes=None,
    pixelization_indexes=None,
):

    inversion_plots.chi_squared_map(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_labels
def regularization_weights(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    full_indexes=None,
    pixelization_indexes=None,
):

    inversion_plots.regularization_weights(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_labels
def interpolated_reconstruction(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    inversion_plots.interpolated_reconstruction(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )


@mat_decorators.set_labels
def interpolated_errors(
    inversion,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    inversion_plots.interpolated_errors(
        inversion=inversion,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )

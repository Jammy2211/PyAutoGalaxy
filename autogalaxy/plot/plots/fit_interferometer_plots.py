from autoarray.plot.mat_wrap import mat_decorators
from autoarray.plot.plots import fit_interferometer_plots
from autogalaxy.plot.mat_wrap import lensing_plotter, lensing_include, lensing_visuals
from autogalaxy.plot.plots import plane_plots, inversion_plots


def subplot_fit_interferometer(
    fit,
    plotter_1d: lensing_plotter.Plotter1D = lensing_plotter.Plotter1D(),
    visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
    include_1d: lensing_include.Include1D = lensing_include.Include1D(),
):

    fit_interferometer_plots.subplot_fit_interferometer(
        fit=fit, plotter_1d=plotter_1d, visuals_1d=visuals_1d, include_1d=include_1d
    )


def subplot_fit_real_space(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    plotter_2d = plotter_2d.plotter_for_subplot_from(func=subplot_fit_real_space)

    number_subplots = 2

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    if fit.inversion is None:

        plane_plots.image(
            plane=fit.plane,
            grid=fit.masked_interferometer.grid,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

        plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        plane_plots.plane_image(
            plane=fit.plane,
            grid=fit.masked_interferometer.grid,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    elif fit.inversion is not None:

        inversion_plots.reconstructed_image(
            inversion=fit.inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

        aspect_inv = plotter_2d.figure.aspect_for_subplot_from_grid(
            grid=fit.inversion.mapper.source_full_grid
        )

        plotter_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=2, aspect=float(aspect_inv)
        )

        inversion_plots.reconstruction(
            inversion=fit.inversion,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


def individuals(
    fit,
    plotter_1d: lensing_plotter.Plotter1D = lensing_plotter.Plotter1D(),
    visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
    include_1d: lensing_include.Include1D = lensing_include.Include1D(),
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    plot_visibilities=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_visibilities=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
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

    fit_interferometer_plots.individuals(
        fit=fit,
        plot_visibilities=plot_visibilities,
        plot_noise_map=plot_noise_map,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_model_visibilities=plot_model_visibilities,
        plot_residual_map=plot_residual_map,
        plot_normalized_residual_map=plot_normalized_residual_map,
        plot_chi_squared_map=plot_chi_squared_map,
    )


@mat_decorators.set_labels
def visibilities(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the visibilities of a lens fit.

    Set *autogalaxy.datas.grid.lensing_plotter.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    fit_interferometer_plots.visibilities(
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

    Set *autogalaxy.datas.grid.lensing_plotter.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    fit_interferometer_plots.noise_map(
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

    Set *autogalaxy.datas.grid.lensing_plotter.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
    The datas-datas, which include_2d the observed datas, signal_to_noise_map, PSF, signal-to-signal_to_noise_map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    fit_interferometer_plots.signal_to_noise_map(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )


@mat_decorators.set_labels
def model_visibilities(
    fit,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the model visibilities of a fit.

    Set *autogalaxy.datas.grid.lensing_plotter.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the model visibilities is plotted.
    """
    fit_interferometer_plots.model_visibilities(
        fit=fit, plotter_2d=plotter_2d, visuals_2d=visuals_2d, include_2d=include_2d
    )


@mat_decorators.set_labels
def residual_map_vs_uv_distances(
    fit,
    plotter_1d: lensing_plotter.Plotter1D = lensing_plotter.Plotter1D(),
    visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
    include_1d: lensing_include.Include1D = lensing_include.Include1D(),
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
):
    """Plot the residual-map of a lens fit.

    Set *autogalaxy.datas.grid.lensing_plotter.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    fit_interferometer_plots.residual_map_vs_uv_distances(
        fit=fit,
        plotter_1d=plotter_1d,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plot_real=plot_real,
        label_yunits=label_yunits,
        label_xunits=label_xunits,
    )


@mat_decorators.set_labels
def normalized_residual_map_vs_uv_distances(
    fit,
    plotter_1d: lensing_plotter.Plotter1D = lensing_plotter.Plotter1D(),
    visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
    include_1d: lensing_include.Include1D = lensing_include.Include1D(),
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
):
    """Plot the residual-map of a lens fit.

    Set *autogalaxy.datas.grid.lensing_plotter.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    fit_interferometer_plots.normalized_residual_map_vs_uv_distances(
        fit=fit,
        plotter_1d=plotter_1d,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plot_real=plot_real,
        label_yunits=label_yunits,
        label_xunits=label_xunits,
    )


@mat_decorators.set_labels
def chi_squared_map_vs_uv_distances(
    fit,
    plotter_1d: lensing_plotter.Plotter1D = lensing_plotter.Plotter1D(),
    visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
    include_1d: lensing_include.Include1D = lensing_include.Include1D(),
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
):
    """Plot the residual-map of a lens fit.

    Set *autogalaxy.datas.grid.lensing_plotter.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    fit_interferometer_plots.chi_squared_map_vs_uv_distances(
        fit=fit,
        plotter_1d=plotter_1d,
        visuals_1d=visuals_1d,
        include_1d=include_1d,
        plot_real=plot_real,
        label_yunits=label_yunits,
        label_xunits=label_xunits,
    )

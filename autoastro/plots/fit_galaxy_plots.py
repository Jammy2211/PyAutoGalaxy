import autoarray as aa
import matplotlib

backend = aa.conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

import autoarray as aa
from autoarray.plotters import plotters, array_plotters
from autoarray.util import plotter_util
from autoastro import exc


def subplot(
    fit,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
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
    )

    plt.subplot(rows, columns, 2)

    aa.plot.fit_imaging.model_image(
        fit=fit,
        mask=fit.mask,
        points=positions,
    )

    plt.subplot(rows, columns, 3)

    aa.plot.fit_imaging.residual_map(
        fit=fit,
        mask=fit.mask,
    )

    plt.subplot(rows, columns, 4)

    aa.plot.fit_imaging.chi_squared_map(
        fit=fit,
        mask=fit.mask,
    )

    array_plotter.output_subplot_array(
    )

    plt.close()


def individuals(
    fit,
    positions=None,
    plot_image=False,
    plot_noise_map=False,
    plot_model_image=False,
    plot_residual_map=False,
    plot_chi_squared_map=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    unit_conversion_factor = None

    if plot_image:

        galaxy_data_array(
            galaxy_data=fit.galaxy_data,
            mask=fit.mask,
            positions=positions,
            array_plotter=array_plotter
        )

    if plot_noise_map:

        aa.plot.fit_imaging.noise_map(
            fit=fit,
            mask=fit.mask,
            points=positions,
            array_plotter=array_plotter
        )

    if plot_model_image:

        aa.plot.fit_imaging.model_image(
            fit=fit,
            mask=fit.mask,
            points=positions,
            array_plotter=array_plotter
        )

    if plot_residual_map:

        aa.plot.fit_imaging.residual_map(
            fit=fit,
            mask=fit.mask,
            array_plotter=array_plotter
        )

    if plot_chi_squared_map:

        aa.plot.fit_imaging.chi_squared_map(
            fit=fit,
            mask=fit.mask,
            array_plotter=array_plotter
        )

@plotters.set_includes
@plotters.set_labels
def galaxy_data_array(
    galaxy_data,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
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
            "The galaxy data_type arrays does not have a True use_profile_type"
        )

    array_plotter.plot_array(
        array=galaxy_data.image,
        mask=galaxy_data.mask,
        points=positions,
    )

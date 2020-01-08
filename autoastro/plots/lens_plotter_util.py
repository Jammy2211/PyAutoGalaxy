from autoarray import conf
from autoarray.plotters import plotters, plotters_util
from autoarray import exc
from autoastro import lensing

from functools import wraps

def get_unit_label_and_unit_conversion_factor(kpc_per_arcsec, plot_in_kpc):

    if plot_in_kpc:

        unit_label = "kpc"
        unit_conversion_factor = kpc_per_arcsec

    else:

        unit_label = "arcsec"
        unit_conversion_factor = None

    return unit_label, unit_conversion_factor


def get_critical_curves_and_caustics_from_lensing_object(
    obj, include_critical_curves, include_caustics
):

    if isinstance(obj, lensing.LensingObject) and obj.has_mass_profile:

        if include_critical_curves:
            try:
                critical_curves = obj.critical_curves
            except ValueError:
                print(
                    "Critical curve could not be calculated due to an unphysical mass model"
                )
                critical_curves = None
        else:
            critical_curves = []

        if include_caustics:
            try:
                caustics = obj.caustics
            except ValueError:
                print(
                    "Caustics could not be calculated due to an unphysical mass model"
                )
                caustics = None
        else:
            caustics = []

        return [critical_curves, caustics]

    else:

        return None


def get_mass_profile_centres_from_fit(include_mass_profile_centres, fit):

    if not hasattr(fit, "tracer"):
        return None

    if include_mass_profile_centres:
        return fit.tracer.image_plane.mass_profile_centres_of_galaxies
    else:
        return None


def get_positions_from_fit(fit, positions):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
    mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if positions:
        return fit.masked_dataset.positions
    else:
        return None


def get_image_plane_pix_grid_from_fit(include_image_plane_pix, fit):

    if fit.inversion is not None:
        if include_image_plane_pix:
            if fit.inversion.mapper.is_image_plane_pixelization:
                return fit.tracer.sparse_image_plane_grids_of_planes_from_grid(
                    grid=fit.grid
                )[-1]

    return None


def set_includes(func):
    """
    Decorate a profile method that accepts a coordinate grid and returns a data_type grid.

    If an interpolator attribute is associated with the input grid then that interpolator is used to down sample the
    coordinate grid prior to calling the function and up sample the result of the function.

    If no interpolator attribute is associated with the input grid then the function is called as hyper.

    Parameters
    ----------
    func
        Some method that accepts a grid

    Returns
    -------
    decorated_function
        The function with optional interpolation
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        includes = ["include_origin", "include_mask", "include_grid", "include_positions", "include_centres", "include_border",
                    "include_critical_curves", "include_caustics"]

        for include in includes:
            if include in kwargs:
                if kwargs[include] is None:

                    kwargs[include] = plotters_util.setting(
                        section="include", name=include[8:], python_type=bool)

        return func(*args, **kwargs)

    return wrapper


def label_yunits_from_plotter(plotter, plot_in_kpc):

    if plotter.label_yunits is None:
        if plot_in_kpc:
            return "kpc"
        else:
            return "arcsec"

    else:

        return plotter.label_yunits


def label_xunits_from_plotter(plotter, plot_in_kpc):

    if plotter.label_xunits is None:
        if plot_in_kpc:
            return "kpc"
        else:
            return "arcsec"

    else:

        return plotter.label_xunits


def kpc_per_arcsec_of_object_from_dictionary(dictionary):

    kpc_per_arcsec = None

    for key, value in dictionary.items():
        if isinstance(value, lensing.LensingObject):
            if hasattr(value, "kpc_per_arcsec"):
                redshift = value.kpc_per_arcsec

    return kpc_per_arcsec

def set_labels_and_unit_conversion(func):
    """
    Decorate a profile method that accepts a coordinate grid and returns a data_type grid.

    If an interpolator attribute is associated with the input grid then that interpolator is used to down sample the
    coordinate grid prior to calling the function and up sample the result of the function.

    If no interpolator attribute is associated with the input grid then the function is called as hyper.

    Parameters
    ----------
    func
        Some method that accepts a grid

    Returns
    -------
    decorated_function
        The function with optional interpolation
    """

    @wraps(func)
    def wrapper(plot_in_kpc=None, *args, **kwargs):

        plotter_key = plotters_util.plotter_key_from_dictionary(dictionary=kwargs)
        plotter = kwargs[plotter_key]

        kpc_per_arcsec = kpc_per_arcsec_of_object_from_dictionary(dictionary=kwargs)

        if kpc_per_arcsec is not None:

            plot_in_kpc = conf.instance.visualize.get("general", "plot_in_kpc", bool) if plot_in_kpc is None else plot_in_kpc

        else:

            plot_in_kpc = False

        label_units, unit_conversion_factor = get_unit_label_and_unit_conversion_factor(
            kpc_per_arcsec=kpc_per_arcsec, plot_in_kpc=plot_in_kpc
        )

        label_title = plotters_util.label_title_from_plotter(plotter=plotter, func=func)
        label_yunits = label_yunits_from_plotter(plotter=plotter, plot_in_kpc=plot_in_kpc)
        label_xunits = label_xunits_from_plotter(plotter=plotter, plot_in_kpc=plot_in_kpc)
        output_filename = plotters_util.output_filename_from_plotter_and_func(plotter=plotter, func=func)

        kwargs[plotter_key] = plotter.plotter_with_new_labels_and_filename(
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            output_filename=output_filename,
            unit_conversion_factor=unit_conversion_factor,
        )

        return func(*args, **kwargs)

    return wrapper
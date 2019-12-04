from autoastro import lensing


def get_unit_label_and_unit_conversion_factor(obj, plot_in_kpc):

    if plot_in_kpc:

        unit_label = "kpc"
        unit_conversion_factor = obj.kpc_per_arcsec

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


def get_positions_from_fit(fit, include_positions):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
    include_mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if include_positions:
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

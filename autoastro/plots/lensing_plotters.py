from autoarray.plotters import plotters
from autoastro import lensing


def positions_from_fit(fit, include_positions):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
    mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if include_positions:
        return fit.masked_dataset.positions
    else:
        return None


def image_plane_pix_grid_from_fit(include_image_plane_pix, fit):

    if fit.inversion is not None:
        if include_image_plane_pix:
            if fit.inversion.mapper.is_image_plane_pixelization:
                return fit.tracer.sparse_image_plane_grids_of_planes_from_grid(
                    grid=fit.grid
                )[-1]

    return None


class Include(plotters.Include):
    def __init__(
        self,
        origin=None,
        mask=None,
        grid=None,
        border=None,
        inversion_centres=None,
        inversion_grid=None,
        inversion_border=None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        critical_curves=None,
        caustics=None,
    ):

        super(Include, self).__init__(
            origin=origin,
            mask=mask,
            grid=grid,
            border=border,
            inversion_centres=inversion_centres,
            inversion_grid=inversion_grid,
            inversion_border=inversion_border,
        )

        self.positions = self.load_include(value=positions, name="positions")
        self.light_profile_centres = self.load_include(
            value=light_profile_centres, name="light_profile_centres"
        )
        self.mass_profile_centres = self.load_include(
            value=mass_profile_centres, name="mass_profile_centres"
        )
        self.critical_curves = self.load_include(
            value=critical_curves, name="critical_curves"
        )
        self.caustics = self.load_include(value=caustics, name="caustics")

    def mass_profile_centres_from_obj(self, obj):

        if self.mass_profile_centres:
            return obj.mass_profile_centres
        else:
            return None

    def critical_curves_and_caustics_from_obj(self, obj):

        if isinstance(obj, lensing.LensingObject) and obj.has_mass_profile:

            if self.critical_curves:
                try:
                    critical_curves = obj.critical_curves
                except ValueError:
                    print(
                        "Critical curve could not be calculated due to an unphysical mass model"
                    )
                    critical_curves = None
            else:
                critical_curves = []

            if self.caustics:
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
